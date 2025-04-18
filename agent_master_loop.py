"""
EideticEngine Agent Master Loop (AML) - v4.0 P1 - ENHANCED
======================================================================

This module implements the core orchestration logic for the EideticEngine
AI agent. It manages the primary think-act cycle, interacts with the
Unified Memory System (UMS) via MCPClient, leverages an LLM (Anthropic Claude)
for decision-making and planning, and incorporates several cognitive functions
inspired by human memory and reasoning.

** V4.0 P1 ENHANCED implements Phase 1 improvements: refined context, adaptive
thresholds, plan validation/repair, structured error handling, and robust
background task management. **

Key Functionalities:
--------------------
*   **Workflow & Context Management:**
    - Creates, manages, and tracks progress within structured workflows.
    - Supports sub-workflow execution via a workflow stack.
    - Gathers rich, multi-faceted context for the LLM decision-making process, including:
        *   Current working memory and focal points **(Prioritized)**.
        *   Core workflow context (recent actions, important memories, key thoughts).
        *   Proactively searched memories relevant to the current goal/plan step **(Limited Fetch)**.
        *   Relevant procedural memories (how-to knowledge) **(Limited Fetch)**.
        *   Summaries of memories linked to the current focus **(Limited Fetch)**.
        *   **Freshness indicators** for context components.
    - Implements structure-aware context truncation and optional LLM-based compression.

*   **Planning & Execution:**
    - Maintains an explicit, modifiable plan consisting of sequential steps with dependencies.
    - Allows the LLM to propose plan updates via a dedicated tool or text parsing.
    - Includes a heuristic fallback mechanism to update plan steps based on action outcomes if the LLM doesn't explicitly replan.
    - **Validates plan steps and detects dependency cycles.**
    - Checks action prerequisites (dependencies) before execution.
    - Executes tools via the MCPClient, handling server lookup and argument injection.
    - Records detailed action history (start, completion, arguments, results, dependencies).

*   **LLM Interaction & Reasoning:**
    - Constructs detailed prompts for the LLM, providing comprehensive context, tool schemas, and cognitive instructions.
    - **Prompts explicitly guide analysis of working memory and provide error recovery strategies.**
    - Parses LLM responses to identify tool calls, textual reasoning (recorded as thoughts), or goal completion signals.
    - Manages dedicated thought chains for recording the agent's reasoning process.

*   **Cognitive & Meta-Cognitive Processes:**
    - **Memory Interaction:** Stores, updates, searches (semantic/hybrid/keyword), and links memories in the UMS.
    - **Working Memory Management:** Retrieves, optimizes (based on relevance/diversity), and automatically focuses working memory via UMS tools.
    - **Background Cognitive Tasks:** Initiates asynchronous tasks **with timeouts and concurrency limits (semaphore)** for:
        *   Automatic semantic linking of newly created/updated memories.
        *   Checking and potentially promoting memories to higher cognitive levels (e.g., Episodic -> Semantic) based on usage/confidence.
    - **Periodic Meta-cognition:** Runs scheduled tasks based on loop intervals or success counters:
        *   **Reflection:** Generates analysis of progress, gaps, strengths, or plans using an LLM.
        *   **Consolidation:** Synthesizes information from multiple memories into summaries, insights, or procedures using an LLM.
        *   **Adaptive Thresholds:** Dynamically adjusts the frequency of reflection/consolidation based on agent performance (e.g., error rates, **memory statistics, trends**) **with enhanced heuristics and dampening**.
    - **Maintenance:** Periodically deletes expired memories.

*   **State & Error Handling:**
    - Persists the complete agent runtime state (workflow, plan, counters, thresholds) atomically to a JSON file for resumption.
    - Implements retry logic with backoff for potentially transient tool failures (especially for idempotent operations).
    - Tracks consecutive errors and halts execution if a limit is reached.
    - Provides detailed, **categorized** error information back to the LLM for recovery attempts.
    - Handles graceful shutdown via system signals (SIGINT, SIGTERM).

────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict, deque  # Added deque for cycle detection  # noqa: F401
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError  # Correct imports
from pydantic import BaseModel, Field, ValidationError

try:
    # Note: Import all potentially used enums/classes from MCPClient for clarity
    from mcp_client import (  # type: ignore
        ActionStatus,
        ActionType,
        ArtifactType,  # noqa: F401
        LinkType,
        MCPClient,
        MemoryLevel,
        MemoryType,
        MemoryUtils,
        ThoughtType,
        ToolError,
        ToolInputError,
        WorkflowStatus,  # Keep import even if unused directly here
    )

    MCP_CLIENT_AVAILABLE = True
    log = logging.getLogger("AgentMasterLoop")
    # Bootstrap logger if MCPClient didn't configure it
    if not logging.root.handlers and not log.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        log = logging.getLogger("AgentMasterLoop")
        log.warning("MCPClient did not configure logger – falling back.")
    log.info("Successfully imported MCPClient and required components.")
except ImportError as import_err:
    # Critical error if MCPClient cannot be imported
    print(f"❌ CRITICAL ERROR: Could not import MCPClient: {import_err}")
    sys.exit(1)

# --------------------------------------------------------------------------
# Runtime log‑level can be raised via AGENT_LOOP_LOG_LEVEL=DEBUG, etc.
# --------------------------------------------------------------------------
LOG_LEVEL_ENV = os.environ.get("AGENT_LOOP_LOG_LEVEL", "INFO").upper()
log.setLevel(getattr(logging, LOG_LEVEL_ENV, logging.INFO))
if log.level <= logging.DEBUG:
    log.info("Verbose logging enabled for Agent loop.")

# ==========================================================================
# CONSTANTS
# ==========================================================================
# File for saving/loading agent state, versioned for this implementation phase
AGENT_STATE_FILE = "agent_loop_state_v4.0_p1_enhanced.json" # Updated filename
# Agent identifier used in prompts and logging
AGENT_NAME = "EidenticEngine4.0-P1-Enhanced" # Updated agent name
# Default LLM model string (can be overridden by environment or config)
MASTER_LEVEL_AGENT_LLM_MODEL_STRING = "claude-3-5-sonnet-20240620" # Use the confirmed model

# ---------------- meta‑cognition thresholds ----------------
# Base thresholds for triggering reflection and consolidation, adjustable via environment
BASE_REFLECTION_THRESHOLD = int(os.environ.get("BASE_REFLECTION_THRESHOLD", "7"))
BASE_CONSOLIDATION_THRESHOLD = int(os.environ.get("BASE_CONSOLIDATION_THRESHOLD", "12"))
# Minimum and maximum bounds for adaptive thresholds to prevent extreme values
MIN_REFLECTION_THRESHOLD = 3
MAX_REFLECTION_THRESHOLD = 15
MIN_CONSOLIDATION_THRESHOLD = 5
MAX_CONSOLIDATION_THRESHOLD = 25
# Dampening factor for threshold adjustments (e.g., 0.75 means apply 75% of calculated change)
THRESHOLD_ADAPTATION_DAMPENING = float(os.environ.get("THRESHOLD_DAMPENING", "0.75"))

# ---------------- interval constants (in loop iterations) ----------------
# How often to run working memory optimization and auto-focus checks
OPTIMIZATION_LOOP_INTERVAL = int(os.environ.get("OPTIMIZATION_INTERVAL", "8"))
# How often to check recently accessed memories for potential level promotion
MEMORY_PROMOTION_LOOP_INTERVAL = int(os.environ.get("PROMOTION_INTERVAL", "15"))
# How often to compute memory statistics and adapt thresholds
STATS_ADAPTATION_INTERVAL = int(os.environ.get("STATS_ADAPTATION_INTERVAL", "10"))
# How often to run maintenance tasks like deleting expired memories
MAINTENANCE_INTERVAL = int(os.environ.get("MAINTENANCE_INTERVAL", "50"))

# ---------------- context / token sizing ----------------
# Delay range (seconds) before running background auto-linking task
AUTO_LINKING_DELAY_SECS: Tuple[float, float] = (1.5, 3.0)
# Default description for the initial plan step if none exists
DEFAULT_PLAN_STEP = "Assess goal, gather context, formulate initial plan."

# Limits for various context components included in the prompt (PRE-FETCH LIMITS)
CONTEXT_RECENT_ACTIONS_FETCH_LIMIT = 10 # Fetch slightly more than shown
CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT = 7
CONTEXT_KEY_THOUGHTS_FETCH_LIMIT = 7
CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT = 3 # Fetch limit for procedural
CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT = 5 # Fetch limit for proactive goal-relevant
CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT = 5 # Fetch limit for link traversal per direction

# Limits for items SHOWN in final prompt context (after potential truncation/summarization)
CONTEXT_RECENT_ACTIONS_SHOW_LIMIT = 7
CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT = 5
CONTEXT_KEY_THOUGHTS_SHOW_LIMIT = 5
CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT = 2 # Limit procedural memories included
CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT = 3 # Limit goal-relevant memories included
CONTEXT_WORKING_MEMORY_SHOW_LIMIT = 10 # Max working memory items shown in context
CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT = 3 # Max links shown per direction in link summary

# Token limits triggering context compression
CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD = 15_000
CONTEXT_COMPRESSION_TARGET_TOKENS = 5_000 # Target size after compression

# Maximum number of consecutive tool execution errors before aborting
MAX_CONSECUTIVE_ERRORS = 3

# ---------------- unified‑memory tool constants ----------------
# Define constants for all UMS tool names for consistency and easy updates
TOOL_GET_WORKFLOW_DETAILS = "unified_memory:get_workflow_details"
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context" # Core context retrieval tool
TOOL_CREATE_WORKFLOW = "unified_memory:create_workflow"
TOOL_UPDATE_WORKFLOW_STATUS = "unified_memory:update_workflow_status"
TOOL_RECORD_ACTION_START = "unified_memory:record_action_start"
TOOL_RECORD_ACTION_COMPLETION = "unified_memory:record_action_completion"
TOOL_GET_ACTION_DETAILS = "unified_memory:get_action_details"
TOOL_ADD_ACTION_DEPENDENCY = "unified_memory:add_action_dependency"
TOOL_GET_ACTION_DEPENDENCIES = "unified_memory:get_action_dependencies"
TOOL_RECORD_ARTIFACT = "unified_memory:record_artifact"
TOOL_GET_ARTIFACTS = "unified_memory:get_artifacts"
TOOL_GET_ARTIFACT_BY_ID = "unified_memory:get_artifact_by_id"
TOOL_HYBRID_SEARCH = "unified_memory:hybrid_search_memories"
TOOL_STORE_MEMORY = "unified_memory:store_memory"
TOOL_UPDATE_MEMORY = "unified_memory:update_memory"
TOOL_GET_WORKING_MEMORY = "unified_memory:get_working_memory"
TOOL_SEMANTIC_SEARCH = "unified_memory:search_semantic_memories"
TOOL_CREATE_THOUGHT_CHAIN = "unified_memory:create_thought_chain"
TOOL_GET_THOUGHT_CHAIN = "unified_memory:get_thought_chain"
TOOL_DELETE_EXPIRED_MEMORIES = "unified_memory:delete_expired_memories"
TOOL_COMPUTE_STATS = "unified_memory:compute_memory_statistics"
TOOL_RECORD_THOUGHT = "unified_memory:record_thought"
TOOL_REFLECTION = "unified_memory:generate_reflection"
TOOL_CONSOLIDATION = "unified_memory:consolidate_memories"
TOOL_OPTIMIZE_WM = "unified_memory:optimize_working_memory"
TOOL_AUTO_FOCUS = "unified_memory:auto_update_focus"
TOOL_PROMOTE_MEM = "unified_memory:promote_memory_level"
TOOL_QUERY_MEMORIES = "unified_memory:query_memories" # Keyword/filter-based search
TOOL_CREATE_LINK = "unified_memory:create_memory_link"
TOOL_GET_MEMORY_BY_ID = "unified_memory:get_memory_by_id"
TOOL_GET_LINKED_MEMORIES = "unified_memory:get_linked_memories"
TOOL_LIST_WORKFLOWS = "unified_memory:list_workflows"
TOOL_GENERATE_REPORT = "unified_memory:generate_workflow_report"
TOOL_SUMMARIZE_TEXT = "unified_memory:summarize_text"
# --- Agent-internal tool name constant ---
AGENT_TOOL_UPDATE_PLAN = "agent:update_plan"

# --- Background Task Management ---
BACKGROUND_TASK_TIMEOUT_SECONDS = 60.0 # Timeout for individual background tasks
MAX_CONCURRENT_BG_TASKS = 10 # Limit concurrent background tasks (linking, promotion)

# ==========================================================================
# Utility helpers
# ==========================================================================

def _fmt_id(val: Any, length: int = 8) -> str:
    """Return a short id string safe for logs."""
    s = str(val) if val is not None else "?"
    return s[:length] if len(s) >= length else s


def _utf8_safe_slice(s: str, max_len: int) -> str:
    """Return a UTF‑8 boundary‑safe slice ≤ max_len bytes."""
    # Ensure input is a string
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf‑8")[:max_len].decode("utf‑8", "ignore")


def _truncate_context(context: Dict[str, Any], max_len: int = 25_000) -> str:
    """
    Structure‑aware context truncation with UTF‑8 safe fallback.

    Attempts to intelligently reduce context size while preserving structure
    and indicating where truncation occurred. Honors CONTEXT_*_SHOW_LIMIT constants.

    1. Serialise full context – if within limit, return.
    2. Iteratively truncate known large lists (e.g., recent actions, memories)
       based on SHOW_LIMIT constants, adding a note about omissions.
    3. Remove lowest‑priority top‑level keys (e.g., procedures, links)
       until size fits.
    4. Final fallback: utf‑8 safe byte slice of the full JSON, attempting to
       make it syntactically valid-ish and adding a clear truncation marker.
    """
    try:
        # Serialize the context to JSON. Use default=str for non-serializable types like datetime.
        full = json.dumps(context, indent=2, default=str, ensure_ascii=False)
    except TypeError:
        # Handle potential non-serializable types even before dumping if necessary
        context = json.loads(json.dumps(context, default=str)) # Pre-process
        full = json.dumps(context, indent=2, default=str, ensure_ascii=False)

    # If already within the limit, return the full JSON string
    if len(full) <= max_len:
        return full

    # Log that truncation is starting
    log.debug(f"Context length {len(full)} exceeds max {max_len}. Applying structured truncation.")
    ctx_copy = copy.deepcopy(context) # Work on a copy
    ctx_copy["_truncation_applied"] = "structure‑aware" # Add metadata about truncation
    original_length = len(full)

    # Define paths to lists that can be truncated and the number of items to keep (using SHOW_LIMIT constants)
    list_paths_to_truncate = [ # (parent_key or None, key_of_list, items_to_keep_constant)
        ("core_context", "recent_actions", CONTEXT_RECENT_ACTIONS_SHOW_LIMIT),
        ("core_context", "important_memories", CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT),
        ("core_context", "key_thoughts", CONTEXT_KEY_THOUGHTS_SHOW_LIMIT),
        (None, "proactive_memories", CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT),
        # Apply limit to the list *within* the working memory dict
        ("current_working_memory", "working_memories", CONTEXT_WORKING_MEMORY_SHOW_LIMIT),
        (None, "relevant_procedures", CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT),
    ]
    # Define keys to remove entirely if context is still too large, in order of least importance
    keys_to_remove_low_priority = [
        "relevant_procedures",
        "proactive_memories",
        "contextual_links", # Link summary is lower priority than core items
        # Removing items from core_context is more impactful
        ("core_context", "key_thoughts"), # Check nested key first
        ("core_context", "important_memories"),
        ("core_context", "recent_actions"),
        "core_context", # Remove the entire core context dict last
        "current_working_memory", # Remove working memory context before byte slice
    ]

    # 1. Truncate specified lists based on SHOW_LIMIT constants
    for parent, key, keep_count in list_paths_to_truncate:
        try:
            container = ctx_copy
            # Navigate to the parent dictionary if specified
            if parent:
                if parent not in container or not isinstance(container[parent], dict):
                    continue # Parent doesn't exist or isn't a dict
                container = container[parent]

            # Check if the key exists, is a list, and needs truncation
            if key in container and isinstance(container[key], list) and len(container[key]) > keep_count:
                original_count = len(container[key])
                # Add a note indicating truncation within the list itself
                note = {"truncated_note": f"{original_count - keep_count} items omitted from '{key}'"}
                # Slice to keep the desired number of items
                container[key] = container[key][:keep_count]
                # Append the note if there's space or if it's crucial context
                if keep_count > 0: # Only add note if we kept some items
                    container[key].append(note)
                log.debug(f"Truncated list '{key}' (under '{parent or 'root'}') to {keep_count} items (+ note).")

            # Check size after each list truncation
            serial = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial) <= max_len:
                log.info(f"Context truncated successfully after list reduction (Length: {len(serial)}).")
                return serial
        except (KeyError, TypeError, IndexError) as e:
            # Log errors during this specific truncation but continue with others
            log.warning(f"Error during list truncation for key '{key}' (under '{parent}'): {e}")
            continue

    # 2. Remove low-priority keys if still too large
    for key_info in keys_to_remove_low_priority:
        removed = False
        key_to_remove = key_info
        parent_key_to_remove = None
        if isinstance(key_info, tuple): # Handle nested keys like ("core_context", "key_thoughts")
            parent_key_to_remove, key_to_remove = key_info

        # Try removing from the parent or root level
        container_to_remove = ctx_copy
        if parent_key_to_remove:
            if parent_key_to_remove in container_to_remove and isinstance(container_to_remove[parent_key_to_remove], dict):
                container_to_remove = container_to_remove[parent_key_to_remove]
            else:
                continue # Skip if parent doesn't exist or isn't a dict

        # Remove the key if it exists in the target container
        if key_to_remove in container_to_remove:
            container_to_remove.pop(key_to_remove)
            removed = True
            log.debug(f"Removed low-priority key '{key_to_remove}' from {'root' if parent_key_to_remove is None else parent_key_to_remove} for truncation.")

        # Check size after each key removal
        if removed:
            serial = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial) <= max_len:
                log.info(f"Context truncated successfully after key removal (Length: {len(serial)}).")
                return serial

    # 3. Ultimate fallback: UTF-8 safe byte slice
    log.warning(f"Structured truncation insufficient (Length still {len(serial)}). Applying final byte-slice.")
    # Slice the original full JSON string, leaving some buffer for the truncation note/closing chars
    clipped_json_str = _utf8_safe_slice(full, max_len - 50)
    # Attempt to make it look somewhat like valid JSON by finding the last closing brace/bracket
    try:
        last_brace = clipped_json_str.rfind('}')
        last_bracket = clipped_json_str.rfind(']')
        cutoff = max(last_brace, last_bracket)
        if cutoff > 0:
            # Truncate after the last complete element and add a note + closing brace
            final_str = clipped_json_str[:cutoff+1] + '\n// ... (CONTEXT TRUNCATED BY BYTE LIMIT) ...\n}'
        else:
            # If no closing elements found, just add a basic note
            final_str = clipped_json_str + '... (CONTEXT TRUNCATED)'
    except Exception:
        # Fallback if string manipulation fails
        final_str = clipped_json_str + '... (CONTEXT TRUNCATED)'

    log.error(f"Context severely truncated from {original_length} to {len(final_str)} bytes using fallback.")
    return final_str

# ==========================================================================
# Dataclass & pydantic models
# ==========================================================================

class PlanStep(BaseModel):
    """Represents a single step in the agent's plan."""
    id: str = Field(default_factory=lambda: f"step-{MemoryUtils.generate_id()[:8]}", description="Unique identifier for the plan step.")
    description: str = Field(..., description="Clear, concise description of the action or goal for this step.")
    status: str = Field(
        default="planned",
        description="Current status of the step: planned, in_progress, completed, failed, skipped.",
    )
    depends_on: List[str] = Field(default_factory=list, description="List of other plan step IDs that must be completed before this step can start.")
    assigned_tool: Optional[str] = Field(default=None, description="Specific tool designated for executing this step, if applicable.")
    tool_args: Optional[Dict[str, Any]] = Field(default=None, description="Arguments to be passed to the assigned tool.")
    result_summary: Optional[str] = Field(default=None, description="A brief summary of the outcome after the step is completed or failed.")
    is_parallel_group: Optional[str] = Field(default=None, description="Optional tag to group steps that can potentially run in parallel.")


def _default_tool_stats() -> Dict[str, Dict[str, Union[int, float]]]:
    """Factory function for initializing tool usage statistics dictionary."""
    # Use defaultdict for convenience: accessing a non-existent tool key will create its default stats entry
    return defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0})


@dataclass
class AgentState:
    """
    Represents the complete persisted runtime state of the Agent Master Loop.

    This dataclass holds all information necessary to resume the agent's operation,
    including workflow context, planning state, error tracking, meta-cognition metrics,
    and adaptive thresholds.

    Attributes:
        workflow_id: The ID of the primary workflow the agent is currently focused on.
        context_id: The specific context ID for memory operations (often matches workflow_id).
        workflow_stack: A list maintaining the hierarchy of active workflows (e.g., for sub-workflows).
        current_plan: A list of `PlanStep` objects representing the agent's current plan.
        current_sub_goal_id: ID of the current specific sub-goal being pursued (optional).
        current_thought_chain_id: ID of the active thought chain for recording reasoning.
        last_action_summary: A brief string summarizing the outcome of the last action taken.
        current_loop: The current iteration number of the main agent loop.
        goal_achieved_flag: Boolean flag indicating if the overall goal has been marked as achieved.
        consecutive_error_count: Counter for consecutive failed actions, used for error limiting.
        needs_replan: Boolean flag indicating if the agent needs to revise its plan in the next cycle.
        last_error_details: A dictionary holding structured information about the last error encountered **(Enhanced with category)**.
        successful_actions_since_reflection: Counter for successful *agent-level* actions since the last reflection.
        successful_actions_since_consolidation: Counter for successful *agent-level* actions since the last consolidation.
        loops_since_optimization: Counter for loops since the last working memory optimization/focus update.
        loops_since_promotion_check: Counter for loops since the last memory promotion check cycle.
        loops_since_stats_adaptation: Counter for loops since the last statistics check and threshold adaptation.
        loops_since_maintenance: Counter for loops since the last maintenance task (e.g., deleting expired memories).
        reflection_cycle_index: Index to cycle through different reflection types.
        last_meta_feedback: Stores the summary of the last reflection/consolidation output for the next prompt.
        current_reflection_threshold: The current dynamic threshold for triggering reflection **(Adaptive)**.
        current_consolidation_threshold: The current dynamic threshold for triggering consolidation **(Adaptive)**.
        tool_usage_stats: Dictionary tracking success/failure counts and latency for each tool used.
        background_tasks: (Transient) Set holding currently running asyncio background tasks (not saved to state file).
    """

    # --- workflow stack ---
    workflow_id: Optional[str] = None
    context_id: Optional[str] = None
    workflow_stack: List[str] = field(default_factory=list)

    # --- planning & reasoning ---
    current_plan: List[PlanStep] = field(
        default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)]
    )
    current_sub_goal_id: Optional[str] = None # For future goal stack feature
    current_thought_chain_id: Optional[str] = None # Tracks the current reasoning thread
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False # Flag to signal loop termination

    # --- error/replan ---
    consecutive_error_count: int = 0
    needs_replan: bool = False # Flag to force replanning cycle
    last_error_details: Optional[Dict[str, Any]] = None # Stores info about the last error **(Enhanced)**

    # --- meta‑cognition metrics ---
    # Counters reset when corresponding meta-task runs or on error
    successful_actions_since_reflection: float = 0.0 # Use float for potential fractional counting
    successful_actions_since_consolidation: float = 0.0
    # Loop counters reset when corresponding periodic task runs
    loops_since_optimization: int = 0
    loops_since_promotion_check: int = 0
    loops_since_stats_adaptation: int = 0
    loops_since_maintenance: int = 0
    reflection_cycle_index: int = 0 # Used to cycle through reflection types
    last_meta_feedback: Optional[str] = None # Feedback from last meta-task for next prompt

    # adaptive thresholds (dynamic) - Initialized from constants, adapted based on stats
    current_reflection_threshold: int = BASE_REFLECTION_THRESHOLD
    current_consolidation_threshold: int = BASE_CONSOLIDATION_THRESHOLD

    # tool statistics - tracks usage counts and latency
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(
        default_factory=_default_tool_stats
    )

    # background tasks (transient) - Not saved/loaded, managed at runtime
    background_tasks: Set[asyncio.Task] = field(
        default_factory=set, init=False, repr=False
    )

# =====================================================================
# Agent Master Loop
# =====================================================================
class AgentMasterLoop:
    """
    Agent Master Loop Orchestrator.

    This class orchestrates the primary think-act cycle of the AI agent.
    It manages state, interacts with the Unified Memory System via MCPClient,
    calls the LLM for decision-making, handles plan execution, and runs
    periodic meta-cognitive tasks (reflection, consolidation, optimization).
    This version integrates rich context gathering and detailed prompting
    as defined in Phase 1 of the v4.0 plan.
    """

    # Set of tool names considered internal or meta-cognitive,
    # which typically shouldn't be recorded as primary agent actions.
    _INTERNAL_OR_META_TOOLS: Set[str] = {
        # Action recording itself is meta
        TOOL_RECORD_ACTION_START,
        TOOL_RECORD_ACTION_COMPLETION,
        # Information retrieval is usually part of the agent's thought process, not a world-altering action
        TOOL_GET_CONTEXT,
        TOOL_GET_WORKING_MEMORY,
        TOOL_SEMANTIC_SEARCH,
        TOOL_HYBRID_SEARCH,
        TOOL_QUERY_MEMORIES,
        TOOL_GET_MEMORY_BY_ID,
        TOOL_GET_LINKED_MEMORIES,
        TOOL_GET_ACTION_DETAILS,
        TOOL_GET_ARTIFACTS,
        TOOL_GET_ARTIFACT_BY_ID,
        TOOL_GET_ACTION_DEPENDENCIES,
        TOOL_GET_THOUGHT_CHAIN,
        TOOL_GET_WORKFLOW_DETAILS, # Getting details is informational
        # Managing relationships is meta
        TOOL_ADD_ACTION_DEPENDENCY,
        TOOL_CREATE_LINK,
        # Admin/Utility tasks are not primary actions
        TOOL_LIST_WORKFLOWS,
        TOOL_COMPUTE_STATS,
        TOOL_SUMMARIZE_TEXT, # Summarization is a utility
        # Periodic cognitive maintenance and enhancement tasks
        TOOL_OPTIMIZE_WM,
        TOOL_AUTO_FOCUS,
        TOOL_PROMOTE_MEM,
        TOOL_REFLECTION,
        TOOL_CONSOLIDATION,
        TOOL_DELETE_EXPIRED_MEMORIES,
        # Agent's internal mechanism for plan updates
        AGENT_TOOL_UPDATE_PLAN,
    }

    # --------------------------------------------------------------- ctor --
    def __init__(
        self, mcp_client_instance: MCPClient, agent_state_file: str = AGENT_STATE_FILE
    ):
        """
        Initializes the AgentMasterLoop.

        Args:
            mcp_client_instance: An initialized instance of the MCPClient.
            agent_state_file: Path to the file for saving/loading agent state.
        """
        # Ensure MCPClient dependency is met
        if not MCP_CLIENT_AVAILABLE:
            raise RuntimeError("MCPClient unavailable. Cannot initialize AgentMasterLoop.")

        self.mcp_client = mcp_client_instance
        # Ensure the Anthropic client is available via MCPClient
        if not hasattr(mcp_client_instance, 'anthropic') or not isinstance(mcp_client_instance.anthropic, AsyncAnthropic):
             self.logger.critical("Anthropic client not found within provided MCPClient instance.")
             raise ValueError("Anthropic client required via MCPClient.")
        self.anthropic_client: AsyncAnthropic = self.mcp_client.anthropic # type: ignore
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        # Configuration parameters for cognitive processes
        self.consolidation_memory_level = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources = 10 # Max memories to feed into consolidation
        self.auto_linking_threshold = 0.7 # Similarity threshold for auto-linking
        self.auto_linking_max_links = 3 # Max links to create per auto-link trigger

        # Sequence of reflection types to cycle through
        self.reflection_type_sequence = [
            "summary", "progress", "gaps", "strengths", "plan",
        ]

        # Initialize agent state (will be overwritten by load if file exists)
        self.state = AgentState()
        # Event to signal graceful shutdown
        self._shutdown_event = asyncio.Event()
        # Lock for safely managing the background tasks set
        self._bg_tasks_lock = asyncio.Lock()
        # Semaphore to limit concurrent background tasks
        self._bg_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BG_TASKS)
        # Placeholder for loaded tool schemas
        self.tool_schemas: List[Dict[str, Any]] = []

    # ----------------------------------------------------------- shutdown --
    async def shutdown(self) -> None:
        """
        Initiates graceful shutdown of the agent loop.

        Sets the shutdown event, cancels pending background tasks,
        and saves the final agent state.
        """
        self.logger.info("Shutdown requested.")
        self._shutdown_event.set() # Signal loops and tasks to stop
        await self._cleanup_background_tasks() # Wait for background tasks
        await self._save_agent_state() # Save final state
        self.logger.info("Agent loop shutdown complete.")

    # ----------------------------------------------------------- prompt --
    def _construct_agent_prompt(
        self, goal: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Builds the system and user prompts for the agent LLM.

        Integrates the overall goal, available tools with schemas, detailed
        process instructions, cognitive guidance, and the current runtime context
        (plan, errors, feedback, memories, etc.) into the prompt structure
        expected by the Anthropic API. Includes robust context truncation.
        **Enhanced with specific guidance on WM, error recovery, and plan repair.**
        """
        # <<< Start Integration Block: Enhance _construct_agent_prompt (Phase 1, Step 1 & 4) >>>
        # ---------- system ----------
        system_blocks: List[str] = [
            f"You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System.",
            "",
            f"Overall Goal: {goal}",
            "",
            # Tool Listing and Cognitive Guidance
            "Available Unified Memory & Agent Tools (Use ONLY these):",
        ]
        if not self.tool_schemas:
            system_blocks.append("- CRITICAL WARNING: No tools loaded. Cannot function.")
        else:
            # Define key tools to highlight in the prompt
            essential_cognitive_tools = {
                TOOL_ADD_ACTION_DEPENDENCY, TOOL_RECORD_ARTIFACT, TOOL_HYBRID_SEARCH,
                TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY, TOOL_CREATE_LINK,
                TOOL_CREATE_THOUGHT_CHAIN, TOOL_GET_THOUGHT_CHAIN, TOOL_RECORD_THOUGHT,
                TOOL_REFLECTION, TOOL_CONSOLIDATION, TOOL_PROMOTE_MEM,
                TOOL_OPTIMIZE_WM, TOOL_AUTO_FOCUS, TOOL_GET_WORKING_MEMORY,
                TOOL_QUERY_MEMORIES, TOOL_SEMANTIC_SEARCH,
                AGENT_TOOL_UPDATE_PLAN # Also highlight the agent's own tool
            }
            # Format each tool schema for the prompt
            for schema in self.tool_schemas:
                sanitized = schema["name"]
                # Ensure mapping exists before using it
                original = self.mcp_client.server_manager.sanitized_to_original.get(sanitized, sanitized)
                desc = schema.get("description", "No description.")
                is_essential = original in essential_cognitive_tools
                prefix = "**" if is_essential else "" # Highlight key tools
                # Provide tool name, mapping, description, and schema
                # Handle different schema formats ('input_schema' vs 'parameters')
                input_schema_str = json.dumps(schema.get('input_schema', schema.get('parameters', {})))
                system_blocks.append(
                    f"\n- {prefix}Name: `{sanitized}` (Represents: `{original}`){prefix}\n"
                    f"  Desc: {desc}\n"
                    f"  Schema: {input_schema_str}"
                )
        system_blocks.append("")
        # --- Detailed Process Instructions (Enhanced) ---
        system_blocks.extend([
            "Your Process at each step:",
            "1.  Context Analysis: Deeply analyze 'Current Context'. Note workflow status, errors (`last_error_details` - *pay attention to error `type`*), recent actions, memories (`core_context`, `proactive_memories`), thoughts, `current_plan`, `relevant_procedures`, **`current_working_memory` (use this for immediate relevance, note `focal_memory_id` if present)**, `current_thought_chain_id`, and `meta_feedback`. Pay attention to memory `importance`/`confidence` and context component `retrieved_at` timestamps.",
            "2.  Error Handling: If `last_error_details` exists, **FIRST** reason about the error `type` and `message`. Propose a recovery strategy in your reasoning. Refer to 'Recovery Strategies' below.",
            "3.  Reasoning & Planning:",
            "    a. State step-by-step reasoning towards the Goal/Sub-goal, integrating context and feedback. Consider `current_working_memory` for immediate context. Record key thoughts using `record_thought` and specify the `thought_chain_id` if different from `current_thought_chain_id`.",
            "    b. Evaluate `current_plan`. Is it valid? Does it address errors? Are dependencies (`depends_on`) likely met? Check for cycles.",
            "    c. **Action Dependencies:** If planning Step B requires output from Step A (action ID 'a123'), include `\"depends_on\": [\"a123\"]` in Step B's plan object.",
            "    d. **Artifact Tracking:** If planning to use a tool that creates a file/data, plan a subsequent step to call `record_artifact`. If needing a previously created artifact, plan to use `get_artifacts` or `get_artifact_by_id` first.",
            "    e. **Direct Memory Management:** If you synthesize a critical new fact, insight, or procedure, plan to use `store_memory` to explicitly save it. If you find strong evidence contradicting a stored memory, plan to use `update_memory` to correct it. Provide clear `content`, `memory_type`, `importance`, and `confidence`.",
            "    f. **Custom Thought Chains:** If tackling a distinct sub-problem or exploring a complex tangent, consider creating a new reasoning thread using `create_thought_chain`. Provide a clear `title`. Subsequent related thoughts should specify the new `thought_chain_id`. The loop tracks the `current_thought_chain_id`.",
            "    g. **Linking:** Identify potential memory relationships (causal, supportive, contradictory). Plan to use `create_memory_link` with specific `link_type`s (e.g., `SUPPORTS`, `CONTRADICTS`, `CAUSAL`, `REFERENCES`).",
            "    h. **Search:** Prefer `hybrid_search_memories` for mixed queries needing relevance and keyword matching. Use `search_semantic_memories` for pure conceptual similarity.",
            "    i. **Update Plan Tool / Repair:** Use `agent:update_plan` ONLY for significant changes (error recovery, major strategy shift, multi-step insertion/removal, fixing validation issues like cycles). **If `needs_replan` is true in context, prioritize using this tool to fix the plan.** Do NOT use it for simply marking a step complete.",
            "4.  Action Decision: Choose **ONE** action based on the *first planned step* in your current plan:",
            "    *   Call Memory/Other Tool: Select the most precise tool. Provide args per schema. **Mandatory:** Call `create_workflow` if context shows 'No Active Workflow'.",
            "    *   Record Thought: Use `record_thought` for logging reasoning, questions, hypotheses etc. Specify `thought_chain_id` if not the current one.",
            "    *   Update Plan Tool: Call `agent:update_plan` with the **complete, repaired** plan structure if major changes are needed or validation failed.",
            "    *   Signal Completion: If Overall Goal is MET, respond ONLY with the text \"Goal Achieved:\" followed by a brief summary.",
            "5.  Output Format: Respond **ONLY** with the valid JSON for the chosen tool call OR the \"Goal Achieved:\" text."
        ])
        # --- Key Considerations (Enhanced) ---
        system_blocks.extend([
            "\nKey Considerations:",
            "*   Dependencies & Cycles: Ensure `depends_on` actions are likely complete. Avoid circular dependencies. Use `get_action_details` if unsure.",
            "*   Artifacts: Track outputs (`record_artifact`), retrieve inputs (`get_artifacts`/`get_artifact_by_id`).",
            "*   Memory: Store important learned info (`store_memory`). Update incorrect info (`update_memory`). Use confidence scores.",
            "*   Thought Chains: Use `create_thought_chain` for complex sub-problems. Use the correct `thought_chain_id` when recording thoughts.",
            "*   Linking: Use specific `link_type`s to build the knowledge graph.",
            "*   Focus: Leverage `current_working_memory` for immediate context. Note the `focal_memory_id`.",
            "*   Errors: Prioritize error analysis and recovery based on `last_error_details.type`."
        ])
        # --- Recovery Strategies Guidance ---
        system_blocks.extend([
            "\nRecovery Strategies based on `last_error_details.type`:",
            "*   `InvalidInputError`: Review tool schema, arguments, and context. Correct the arguments and retry OR choose a different tool/step.",
            "*   `DependencyNotMetError`: Use `get_action_details` on dependency IDs to check status. Adjust plan order using `agent:update_plan` or wait.",
            "*   `ServerUnavailable` / `NetworkError`: The tool's server might be down. Try a different tool, wait, or adjust the plan.",
            "*   `APILimitError` / `RateLimitError`: The external API (e.g., LLM) is busy. Plan to wait (record a thought) before retrying the step.",
            "*   `ToolExecutionError` / `ToolInternalError`: The tool failed internally. Analyze the error message. Maybe try different arguments, use an alternative tool, or adjust the plan.",
            "*   `PlanUpdateError`: The plan structure you proposed was invalid. Re-examine the plan steps and dependencies, then try `agent:update_plan` again with a valid list.",
            "*   `PlanValidationError`: The proposed plan has logical issues (e.g., cycles). Debug dependencies and propose a corrected plan structure using `agent:update_plan`.",
            "*   `CancelledError`: The previous action was cancelled. Re-evaluate the current step.",
            "*   `UnknownError` / `UnexpectedExecutionError`: Analyze the error message carefully. Try to understand the cause. You might need to simplify the step, use a different approach, or ask for clarification via `record_thought` if stuck."
        ])
        system_prompt = "\n".join(system_blocks)
        # <<< End Integration Block: Enhance _construct_agent_prompt >>>

        # ---------- user ----------
        # Construct the user part of the prompt, including truncated context
        context_json = _truncate_context(context) # Apply robust truncation
        user_blocks = [
            "Current Context:",
            "```json",
            context_json,
            "```",
            "",
            "Current Plan:",
            "```json",
            # Serialize current plan steps (ensure model_dump handles exclude_none)
            json.dumps(
                [step.model_dump(exclude_none=True) for step in self.state.current_plan],
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            # Include summary of the last action taken
            f"Last Action Summary:\n{self.state.last_action_summary}\n",
        ]
        # If there was an error in the previous step, include details prominently
        if self.state.last_error_details:
            user_blocks += [
                "**CRITICAL: Address Last Error Details**:", # Highlight error
                "```json",
                # Use default=str for safe serialization of potential complex error objects
                json.dumps(self.state.last_error_details, indent=2, default=str),
                "```",
                "",
            ]
        # If there's feedback from meta-cognitive tasks, include it
        if self.state.last_meta_feedback:
            user_blocks += [
                "**Meta-Cognitive Feedback**:", # Highlight feedback
                self.state.last_meta_feedback,
                "",
            ]
        # Reiterate the overall goal and the final instruction
        user_blocks += [
            f"Overall Goal: {goal}",
            "",
            # Updated instruction emphasizing error/plan repair
            "Instruction: Analyze context & errors (use recovery strategies if needed). Reason step-by-step. Evaluate and **REPAIR** the plan if `needs_replan` is true or errors indicate plan issues (use `agent:update_plan`). Otherwise, decide ONE action based on the *first planned step*: call a tool (output tool_use JSON), record a thought (`record_thought`), or signal completion (output 'Goal Achieved: ...').",
        ]
        user_prompt = "\n".join(user_blocks)

        # Return structure for Anthropic API (user prompt combines system instructions and current state)
        # Note: Anthropic recommends placing system prompts outside the 'messages' list if using their client directly.
        # Here, we combine them into the user message content as per the original structure.
        # >>>>> PRESERVED ORIGINAL PROMPT STRUCTURE <<<<<
        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]


    # ---------------------------------------------------------- bg‑task utils --
    def _background_task_done(self, task: asyncio.Task) -> None:
        """Callback attached to background tasks upon completion."""
        # Schedule the safe cleanup coroutine to avoid blocking the callback
        asyncio.create_task(self._background_task_done_safe(task))

    async def _background_task_done_safe(self, task: asyncio.Task) -> None:
        """
        Safely removes a completed task from the tracking set, releases the semaphore,
        and logs any exceptions. Ensures thread-safety using an asyncio Lock.
        """
        # <<< Start Integration Block: Enhance _background_task_done_safe (Phase 1, Step 5) >>>
        was_present = False
        async with self._bg_tasks_lock: # Acquire lock before modifying the set
            if task in self.state.background_tasks:
                 self.state.background_tasks.discard(task)
                 was_present = True

        # Release the semaphore ONLY if the task was successfully removed from the set
        # This prevents releasing the semaphore multiple times if the callback somehow fires twice
        if was_present:
            try:
                self._bg_task_semaphore.release()
                log.debug(f"Released semaphore. Count: {self._bg_task_semaphore._value}. Task: {task.get_name()}")
            except ValueError:
                 # This can happen if release is called more times than acquire (should not normally occur)
                 log.warning(f"Semaphore release attempt failed for task {task.get_name()}. Already fully released?")
            except Exception as sem_err:
                 log.error(f"Unexpected error releasing semaphore for task {task.get_name()}: {sem_err}")

        # Log cancellation or exceptions after releasing the lock and semaphore
        if task.cancelled():
            self.logger.debug(f"Background task {task.get_name()} was cancelled.")
            return
        # Check if the task encountered an exception
        exc = task.exception()
        if exc:
            # Log the exception details
            self.logger.error(
                # Provide task name/info for better debugging
                f"Background task {task.get_name()} failed: {type(exc).__name__}",
                exc_info=(type(exc), exc, exc.__traceback__), # Provide full traceback info
            )
        # <<< End Integration Block: Enhance _background_task_done_safe >>>

    def _start_background_task(self, coro_fn, *args, **kwargs) -> asyncio.Task:
        """
        Creates and starts an asyncio task for a background operation.

        Acquires a semaphore slot before starting. Includes timeout handling.
        Captures essential state (workflow_id, context_id) at the time of creation
        to ensure the background task operates on the correct context, even if the
        main agent state changes before the task runs. Adds the task to the
        tracking set for cleanup.

        Args:
            coro_fn: The async function (coroutine) to run in the background.
                     Must accept `self` as the first argument if it's an instance method.
            *args: Positional arguments to pass to `coro_fn`.
            **kwargs: Keyword arguments to pass to `coro_fn`.

        Returns:
            The created asyncio.Task object.
        """
        # <<< Start Integration Block: Enhance _start_background_task (Phase 1, Step 5) >>>
        # Snapshot critical state needed by the background task at the moment of creation
        snapshot_wf_id = self.state.workflow_id
        snapshot_ctx_id = self.state.context_id
        # Add other state variables here if specific background tasks need them

        # Define an async wrapper function to execute the coroutine with snapshotted state
        async def _wrapper():
            # Acquire semaphore before running the actual work
            log.debug(f"Waiting for semaphore... Task: {asyncio.current_task().get_name()}. Current count: {self._bg_task_semaphore._value}")
            await self._bg_task_semaphore.acquire()
            log.debug(f"Acquired semaphore. Task: {asyncio.current_task().get_name()}. New count: {self._bg_task_semaphore._value}")
            try:
                # Run the actual coroutine with timeout
                await asyncio.wait_for(
                    coro_fn(
                        self, # Pass the agent instance
                        *args,
                        workflow_id=snapshot_wf_id, # Pass snapshotted workflow_id
                        context_id=snapshot_ctx_id, # Pass snapshotted context_id
                        **kwargs,
                    ),
                    timeout=BACKGROUND_TASK_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                # Log timeout specifically
                self.logger.warning(f"Background task {asyncio.current_task().get_name()} timed out after {BACKGROUND_TASK_TIMEOUT_SECONDS}s.")
            except Exception:
                # Log other exceptions, they will also be logged by the done callback
                self.logger.debug(f"Exception caught within background task wrapper {asyncio.current_task().get_name()}. Will be logged by done callback.")
            # finally:
            #     # Ensure semaphore is released *within the task itself*
            #     # This is handled by the done callback now to simplify wrapper logic
            #     # self._bg_task_semaphore.release()
            #     # log.debug(f"Released semaphore via FINALLY. Task: {asyncio.current_task().get_name()}")
            #     pass

        # Create the asyncio task
        # Naming the task helps with debugging
        task_name = f"bg_{coro_fn.__name__}_{_fmt_id(snapshot_wf_id)}_{random.randint(100,999)}"
        task = asyncio.create_task(_wrapper(), name=task_name)

        # Schedule adding the task to the tracking set safely using another task
        # This avoids potential blocking if the lock is held
        asyncio.create_task(self._add_bg_task(task))

        # Add the completion callback to handle cleanup and logging (this callback now also releases semaphore)
        task.add_done_callback(self._background_task_done)
        self.logger.debug(f"Started background task: {task.get_name()} for WF {_fmt_id(snapshot_wf_id)}")
        return task
        # <<< End Integration Block: Enhance _start_background_task >>>


    async def _add_bg_task(self, task: asyncio.Task) -> None:
        """Safely add a task to the background task set using the lock."""
        async with self._bg_tasks_lock:
            self.state.background_tasks.add(task)

    async def _cleanup_background_tasks(self) -> None:
        """
        Cancels all pending background tasks and awaits their completion.
        Called during graceful shutdown. Ensures semaphores are released.
        """
        # <<< Start Integration Block: Enhance _cleanup_background_tasks (Phase 1, Step 5) >>>
        tasks_to_cleanup: List[asyncio.Task] = []
        async with self._bg_tasks_lock: # Acquire lock to safely get the list
            # Create a copy to iterate over, as the set might be modified by callbacks
            tasks_to_cleanup = list(self.state.background_tasks)

        if not tasks_to_cleanup:
            self.logger.debug("No background tasks to clean up.")
            return

        self.logger.info(f"Cleaning up {len(tasks_to_cleanup)} background tasks…")
        cancelled_tasks = []
        already_done_tasks = []

        # Cancel running tasks
        for t in tasks_to_cleanup:
            if not t.done():
                # Cancel any task that hasn't finished yet
                t.cancel()
                cancelled_tasks.append(t)
            else:
                 already_done_tasks.append(t)

        # Wait for all tasks (including those just cancelled) to finish
        # return_exceptions=True prevents gather from stopping on the first exception
        results = await asyncio.gather(*tasks_to_cleanup, return_exceptions=True)

        # Log the outcome of each task cleanup
        for i, res in enumerate(results):
            task = tasks_to_cleanup[i]
            task_name = task.get_name()
            if isinstance(res, asyncio.CancelledError):
                # This is expected for tasks that were cancelled above
                self.logger.debug(f"Task {task_name} successfully cancelled during cleanup.")
            elif isinstance(res, Exception):
                # Log any unexpected errors that occurred during task execution/cleanup
                self.logger.error(f"Task {task_name} raised an exception during cleanup: {res}")
            else:
                # Task completed normally before/during cleanup or was already done
                self.logger.debug(f"Task {task_name} finalized during cleanup (completed normally or already done).")

            # --- Ensure semaphore is released ---
            # The done callback should handle release, but we add a check here as a failsafe
            # during shutdown, especially if the callback didn't run or failed itself.
            # This is tricky because we don't know if the task acquired the semaphore before being cancelled.
            # A potentially safer approach is *not* to release here unless we are certain the task
            # acquired it and didn't release it. Releasing without acquiring increments the semaphore count.
            # Given the complexity, we rely on the robust done_callback for release.
            # Adding a log message if the semaphore count seems wrong at the end might be useful.

        # Clear the tracking set after all tasks are handled
        async with self._bg_tasks_lock:
            self.state.background_tasks.clear()

        # Final check on semaphore count after cleanup
        final_sem_count = self._bg_task_semaphore._value
        if final_sem_count != MAX_CONCURRENT_BG_TASKS:
             self.logger.warning(f"Semaphore count is {final_sem_count} after cleanup, expected {MAX_CONCURRENT_BG_TASKS}. Some tasks might not have released.")

        self.logger.info("Background tasks cleanup finished.")
        # <<< End Integration Block: Enhance _cleanup_background_tasks >>>


    # ------------------------------------------------------- token estimator --
    async def _estimate_tokens_anthropic(self, data: Any) -> int:
        """
        Estimates token count for given data using the Anthropic client.

        Handles serialization of non-string data and provides a fallback
        heuristic (chars/4) if the API call fails.
        """
        if data is None:
            return 0
        try:
            if not self.anthropic_client:
                # This should ideally be caught during initialization
                raise RuntimeError("Anthropic client unavailable for token estimation")

            # Convert data to string if it's not already (e.g., dict, list)
            text_to_count = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)

            # Use the actual count_tokens method from the anthropic client
            token_count = await self.anthropic_client.count_tokens(text_to_count)
            return int(token_count) # Ensure result is an integer

        except Exception as e:
            # Log the specific error from the API call
            self.logger.warning(f"Token estimation via Anthropic API failed: {e}. Using fallback.")
            # Fallback heuristic: Estimate based on character count
            try:
                text_representation = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
                return len(text_representation) // 4 # Rough approximation
            except Exception as fallback_e:
                # Log error if even the fallback fails
                self.logger.error(f"Token estimation fallback failed: {fallback_e}")
                return 0 # Return 0 if all estimation methods fail

    # --------------------------------------------------------------- retry util --
    async def _with_retries(
        self,
        coro_fun, # The async function to call
        *args,
        max_retries: int = 3,
        # Exceptions to retry on (can be customized per call)
        retry_exceptions: Tuple[type[BaseException], ...] = (
            ToolError, ToolInputError, # Specific MCP errors
            asyncio.TimeoutError, ConnectionError, # Common network issues
            APIConnectionError, RateLimitError, # Anthropic network issues
            APIStatusError, # Treat potentially transient API status errors as retryable
        ),
        retry_backoff: float = 2.0, # Exponential backoff factor
        jitter: Tuple[float, float] = (0.1, 0.5), # Random jitter range (min_sec, max_sec)
        **kwargs,
    ):
        """
        Generic retry wrapper for coroutine functions with exponential backoff and jitter.

        Args:
            coro_fun: The async function to execute and potentially retry.
            *args: Positional arguments for `coro_fun`.
            max_retries: Maximum number of total attempts (1 initial + max_retries-1 retries).
            retry_exceptions: Tuple of exception types that trigger a retry.
            retry_backoff: Multiplier for exponential backoff calculation.
            jitter: Tuple (min, max) defining the range for random delay added to backoff.
            **kwargs: Keyword arguments for `coro_fun`.

        Returns:
            The result of `coro_fun` upon successful execution.

        Raises:
            The last exception encountered if all retry attempts fail.
            asyncio.CancelledError: If cancellation occurs during the retry loop or wait.
        """
        attempt = 0
        last_exception = None # Store the last exception for re-raising
        while True:
            try:
                # Attempt to execute the coroutine
                return await coro_fun(*args, **kwargs)
            except retry_exceptions as e:
                attempt += 1
                last_exception = e # Store the exception
                # Check if max retries have been reached
                if attempt >= max_retries:
                    self.logger.error(f"{coro_fun.__name__} failed after {max_retries} attempts. Last error: {e}")
                    raise # Re-raise the last encountered exception
                # Calculate delay with exponential backoff and random jitter
                delay = (retry_backoff ** (attempt - 1)) + random.uniform(*jitter)
                self.logger.warning(
                    f"{coro_fun.__name__} failed ({type(e).__name__}: {str(e)[:100]}...); retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                # Check for shutdown signal *before* sleeping
                if self._shutdown_event.is_set():
                    self.logger.warning(f"Shutdown signaled during retry wait for {coro_fun.__name__}. Aborting retry.")
                    # Raise CancelledError to stop the process cleanly if shutdown occurs during wait
                    raise asyncio.CancelledError(f"Shutdown during retry for {coro_fun.__name__}") from last_exception
                # Wait for the calculated delay
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                 # Propagate cancellation immediately if caught
                 self.logger.info(f"Coroutine {coro_fun.__name__} was cancelled during retry loop.")
                 raise


    # ---------------------------------------------------------------- state I/O --
    async def _save_agent_state(self) -> None:
        """
        Saves the current agent state to a JSON file atomically.

        Uses a temporary file and `os.replace` for atomicity. Includes fsync
        for robustness against crashes. Serializes the AgentState dataclass,
        handling nested structures like the plan and tool stats. Excludes
        transient fields like `background_tasks`.
        """
        # Create a dictionary from the dataclass state
        state_dict = dataclasses.asdict(self.state)
        # Add a timestamp for when the state was saved
        state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Ensure background_tasks (non-serializable Set[Task]) is removed
        state_dict.pop("background_tasks", None)
        # Convert defaultdict to regular dict for saving tool stats
        state_dict["tool_usage_stats"] = {
            k: dict(v) for k, v in self.state.tool_usage_stats.items()
        }
        # Convert PlanStep Pydantic objects to dictionaries for saving
        state_dict["current_plan"] = [
            step.model_dump(exclude_none=True) for step in self.state.current_plan
        ]

        try:
            # Ensure the directory for the state file exists
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            # Define a temporary file path for atomic write
            tmp_file = self.agent_state_file.with_suffix(f".tmp_{os.getpid()}") # Process-specific avoids collisions
            # Write to the temporary file asynchronously
            async with aiofiles.open(tmp_file, "w", encoding='utf-8') as f:
                # Dump the state dictionary to JSON with indentation
                await f.write(json.dumps(state_dict, indent=2, ensure_ascii=False))
                # Ensure data is written to the OS buffer
                await f.flush()
                # Ensure data is physically written to disk (crucial for crash recovery)
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    # fsync might fail on some systems/filesystems (e.g., network drives)
                    self.logger.warning(f"os.fsync failed during state save: {e} (Continuing, but save might not be fully durable)")

            # Atomically replace the old state file with the new temporary file
            os.replace(tmp_file, self.agent_state_file)
            self.logger.debug(f"State saved atomically → {self.agent_state_file}")
        except Exception as e:
            # Log any errors during the save process
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)
            # Attempt to clean up the temporary file if it exists after an error
            if 'tmp_file' in locals() and tmp_file.exists():
                try:
                    os.remove(tmp_file)
                except OSError as rm_err:
                     self.logger.error(f"Failed to remove temporary state file {tmp_file}: {rm_err}")


    async def _load_agent_state(self) -> None:
        """
        Loads agent state from the JSON file.

        Handles file not found, JSON decoding errors, and potential mismatches
        between the saved state structure and the current `AgentState` dataclass
        (missing keys use defaults, extra keys are ignored with a warning).
        Ensures critical fields like thresholds are initialized even if loading fails.
        """
        # Check if the state file exists
        if not self.agent_state_file.exists():
            # If no file, initialize with defaults, ensuring thresholds are set
            self.state = AgentState(
                 current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                 current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
                 # Other fields will use their dataclass defaults
            )
            self.logger.info("No prior state file found. Starting fresh with default state.")
            return
        # Try loading the state file
        try:
            async with aiofiles.open(self.agent_state_file, "r", encoding='utf-8') as f:
                # Read and parse the JSON data
                data = json.loads(await f.read())

            # Prepare keyword arguments for AgentState constructor
            kwargs: Dict[str, Any] = {}
            processed_keys = set() # Track keys successfully processed from the file

            # Iterate through the fields defined in the AgentState dataclass
            for fld in dataclasses.fields(AgentState):
                # Skip fields that are not meant to be initialized (like background_tasks)
                if not fld.init:
                    continue

                name = fld.name
                processed_keys.add(name)

                # Check if the field exists in the loaded data
                if name in data:
                    value = data[name]
                    # Handle specific fields needing type conversion or validation
                    if name == "current_plan":
                        try:
                            # Validate and convert saved plan steps back to PlanStep objects
                            if isinstance(value, list):
                                kwargs["current_plan"] = [PlanStep(**d) for d in value]
                            else:
                                raise TypeError("Saved plan is not a list")
                        except (ValidationError, TypeError) as e:
                            # If plan loading fails, reset to default plan
                            self.logger.warning(f"Plan reload failed: {e}. Resetting plan.")
                            kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    elif name == "tool_usage_stats":
                        # Reconstruct defaultdict structure for tool stats
                        dd = _default_tool_stats()
                        if isinstance(value, dict):
                            for k, v_dict in value.items():
                                if isinstance(v_dict, dict):
                                    # Ensure required keys exist with correct types
                                    dd[k]["success"] = int(v_dict.get("success", 0))
                                    dd[k]["failure"] = int(v_dict.get("failure", 0))
                                    dd[k]["latency_ms_total"] = float(v_dict.get("latency_ms_total", 0.0))
                        kwargs["tool_usage_stats"] = dd
                    # Add handling for other complex types here if necessary in the future
                    else:
                        # Directly assign the loaded value if no special handling is needed
                        kwargs[name] = value
                else:
                    # Field defined in AgentState but missing in saved data
                    self.logger.debug(f"Field '{name}' not found in saved state. Using default.")
                    # Use the dataclass default factory or default value if defined
                    if fld.default_factory is not dataclasses.MISSING:
                        kwargs[name] = fld.default_factory()
                    elif fld.default is not dataclasses.MISSING:
                        kwargs[name] = fld.default
                    # Explicitly handle potentially missing thresholds if they didn't have defaults
                    elif name == "current_reflection_threshold":
                        kwargs[name] = BASE_REFLECTION_THRESHOLD
                    elif name == "current_consolidation_threshold":
                        kwargs[name] = BASE_CONSOLIDATION_THRESHOLD
                    # Otherwise, the field will be missing if it had no default and wasn't saved

            # Warn about extra keys found in the file but not defined in the current AgentState
            # This helps detect state format drift or old fields
            extra_keys = set(data.keys()) - processed_keys - {"timestamp"} # Exclude meta timestamp key
            if extra_keys:
                self.logger.warning(f"Ignoring unknown keys found in state file: {extra_keys}")

            # Create the AgentState instance using the processed keyword arguments
            # >>>>> PRESERVED ORIGINAL LOADING LOGIC FOR UNHANDLED FIELDS <<<<<
            # Note: This implicitly handles fields not explicitly checked above,
            # relying on the dataclass constructor to handle types if possible.
            # It's generally safer to handle complex types explicitly as done for plan/stats.
            temp_state = AgentState(**kwargs)

            # Ensure mandatory fields (like thresholds) have values AFTER construction,
            # using defaults if somehow missed or loading failed for them.
            if not isinstance(temp_state.current_reflection_threshold, int):
                 self.logger.warning(f"Invalid loaded reflection threshold ({temp_state.current_reflection_threshold}). Resetting to base.")
                 temp_state.current_reflection_threshold = BASE_REFLECTION_THRESHOLD
            else:
                 # Ensure loaded threshold is within bounds
                 temp_state.current_reflection_threshold = max(MIN_REFLECTION_THRESHOLD, min(MAX_REFLECTION_THRESHOLD, temp_state.current_reflection_threshold))

            if not isinstance(temp_state.current_consolidation_threshold, int):
                 self.logger.warning(f"Invalid loaded consolidation threshold ({temp_state.current_consolidation_threshold}). Resetting to base.")
                 temp_state.current_consolidation_threshold = BASE_CONSOLIDATION_THRESHOLD
            else:
                 # Ensure loaded threshold is within bounds
                 temp_state.current_consolidation_threshold = max(MIN_CONSOLIDATION_THRESHOLD, min(MAX_CONSOLIDATION_THRESHOLD, temp_state.current_consolidation_threshold))

            # Assign the potentially corrected state
            self.state = temp_state
            self.logger.info(f"Loaded state from {self.agent_state_file}; current loop {self.state.current_loop}")

        except (json.JSONDecodeError, TypeError, FileNotFoundError) as e:
            # Handle common file loading or parsing errors
            self.logger.error(f"State load failed: {e}. Resetting to default state.", exc_info=True)
            # Reset to a clean default state on failure
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
        except Exception as e:
            # Catch any other unexpected errors during state loading
            self.logger.critical(f"Unexpected error loading state: {e}. Resetting to default state.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )


    # --------------------------------------------------- tool‑lookup helper --
    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """
        Finds an active server providing the specified tool via MCPClient's Server Manager.

        Args:
            tool_name: The original, potentially sanitized, tool name (e.g., "unified_memory:store_memory").

        Returns:
            The name of the active server providing the tool, "AGENT_INTERNAL" for the plan update tool,
            or None if the tool is not found on any active server.
        """
        # Ensure MCP Client and its server manager are available
        if not self.mcp_client or not self.mcp_client.server_manager:
            self.logger.error("MCP Client or Server Manager not available for tool lookup.")
            return None

        sm = self.mcp_client.server_manager
        # Check registered tools first (uses original tool names)
        if tool_name in sm.tools:
            server_name = sm.tools[tool_name].server_name
            # Verify the server is currently connected and considered active
            if server_name in sm.active_sessions:
                self.logger.debug(f"Found tool '{tool_name}' on active server '{server_name}'.")
                return server_name
            else:
                # Tool is known but its server is not currently active
                self.logger.debug(f"Server '{server_name}' for tool '{tool_name}' is registered but not active.")
                return None

        # Handle core tools if a server named "CORE" is active
        # (Assuming core tools follow a "core:" prefix convention)
        if tool_name.startswith("core:") and "CORE" in sm.active_sessions:
            self.logger.debug(f"Found core tool '{tool_name}' on active CORE server.")
            return "CORE"

        # Handle the agent's internal plan update tool
        if tool_name == AGENT_TOOL_UPDATE_PLAN:
            self.logger.debug(f"Internal tool '{tool_name}' does not require a server.")
            return "AGENT_INTERNAL" # Return a special marker

        # If the tool is not found in registered tools or core tools
        self.logger.debug(f"Tool '{tool_name}' not found on any active server.")
        return None


    # ------------------------------------------------------------ initialization --
    async def initialize(self) -> bool:
        """
        Initializes the Agent Master Loop.

        Loads prior agent state, fetches available tool schemas from MCPClient,
        validates the presence of essential tools, checks the validity of any
        loaded workflow state, and sets the initial thought chain ID.

        Returns:
            True if initialization is successful, False otherwise.
        """
        self.logger.info("Initializing Agent loop …")
        # Load state from file first
        await self._load_agent_state()

        # Ensure context_id matches workflow_id if context_id was missing after load
        # This maintains consistency, assuming context usually maps 1:1 with workflow initially
        if self.state.workflow_id and not self.state.context_id:
            self.state.context_id = self.state.workflow_id
            self.logger.info(f"Initialized context_id from loaded workflow_id: {_fmt_id(self.state.workflow_id)}")

        try:
            # Check if MCPClient's server manager is ready
            if not self.mcp_client.server_manager:
                self.logger.error("MCP Client server manager not initialized.")
                return False

            # Fetch all available tool schemas formatted for the LLM (e.g., Anthropic format)
            all_tools = self.mcp_client.server_manager.format_tools_for_anthropic()

            # Manually inject the schema for the internal agent plan-update tool
            plan_step_schema = PlanStep.model_json_schema()
            # Remove 'title' if Pydantic adds it automatically, as it's not part of PlanStep definition
            plan_step_schema.pop('title', None)
            all_tools.append(
                {
                    "name": AGENT_TOOL_UPDATE_PLAN, # Use sanitized name expected by LLM
                    "description": "Replace the agent's current plan with a new list of plan steps. Use this for significant replanning or error recovery.",
                    "input_schema": { # Anthropic uses 'input_schema'
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "array",
                                "description": "Complete new plan as a list of PlanStep objects.",
                                "items": plan_step_schema, # Embed the generated PlanStep schema
                            }
                        },
                        "required": ["plan"],
                    },
                }
            )

            # Filter the fetched schemas to keep only those relevant to this agent
            # (Unified Memory tools and the internal agent tool)
            self.tool_schemas = []
            loaded_tool_names = set()
            for sc in all_tools:
                # Map the sanitized name back to the original for filtering
                original_name = self.mcp_client.server_manager.sanitized_to_original.get(sc["name"], sc["name"])
                # Keep if it's a unified_memory tool or the internal agent plan update tool
                if original_name.startswith("unified_memory:") or sc["name"] == AGENT_TOOL_UPDATE_PLAN:
                    self.tool_schemas.append(sc)
                    loaded_tool_names.add(original_name)

            self.logger.info(f"Loaded {len(self.tool_schemas)} relevant tool schemas: {loaded_tool_names}")

            # Verify that essential tools for core functionality are available
            essential = [
                TOOL_CREATE_WORKFLOW, TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION,
                TOOL_RECORD_THOUGHT, TOOL_STORE_MEMORY, TOOL_GET_WORKING_MEMORY,
                TOOL_HYBRID_SEARCH, # Essential for enhanced context gathering
                TOOL_GET_CONTEXT,   # Essential for core context
                TOOL_REFLECTION,    # Essential for meta-cognition loop
                TOOL_CONSOLIDATION, # Essential for meta-cognition loop
                TOOL_GET_WORKFLOW_DETAILS, # Needed for setting default chain ID on load
            ]
            # Check availability using the server lookup helper
            missing = [t for t in essential if not self._find_tool_server(t)]
            if missing:
                # Log as error because agent functionality will be significantly impaired
                self.logger.error(f"Missing essential tools: {missing}. Agent functionality WILL BE impaired.")
                # Depending on desired strictness, could return False here to halt initialization

            # Check the validity of the workflow ID loaded from the state file
            # Determine the top workflow ID from the stack or the primary ID
            top_wf = (self.state.workflow_stack[-1] if self.state.workflow_stack else None) or self.state.workflow_id
            if top_wf and not await self._check_workflow_exists(top_wf):
                # If the loaded workflow doesn't exist anymore, reset workflow-specific state
                self.logger.warning(
                    f"Stored workflow '{_fmt_id(top_wf)}' not found in UMS; resetting workflow-specific state."
                )
                # Preserve non-workflow specific state like stats and dynamic thresholds
                preserved_stats = self.state.tool_usage_stats
                pres_ref_thresh = self.state.current_reflection_threshold
                pres_con_thresh = self.state.current_consolidation_threshold
                # Reset state, keeping only preserved items
                self.state = AgentState(
                    tool_usage_stats=preserved_stats,
                    current_reflection_threshold=pres_ref_thresh,
                    current_consolidation_threshold=pres_con_thresh
                    # All other fields reset to defaults
                )
                # Save the reset state immediately
                await self._save_agent_state()

            # Initialize the current thought chain ID if a workflow exists but the chain ID is missing
            # This ensures thoughts are recorded correctly after loading state
            if self.state.workflow_id and not self.state.current_thought_chain_id:
                await self._set_default_thought_chain_id() # Attempt to find/set the primary chain

            self.logger.info("Agent loop initialization complete.")
            return True # Initialization successful
        except Exception as e:
            # Catch any unexpected errors during initialization
            self.logger.critical(f"Agent loop initialization failed: {e}", exc_info=True)
            return False # Initialization failed


    async def _set_default_thought_chain_id(self):
        """
        Sets the `current_thought_chain_id` in the agent state to the primary
        (usually first created) thought chain associated with the current workflow.
        """
        # Determine the current workflow ID from the stack or primary state
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.debug("Cannot set default thought chain ID: No active workflow.")
            return # Cannot set if no workflow is active

        get_details_tool = TOOL_GET_WORKFLOW_DETAILS # Tool needed to fetch chains

        # Check if the necessary tool is available
        if self._find_tool_server(get_details_tool):
            try:
                # Call the tool internally to get workflow details, including thought chains
                details = await self._execute_tool_call_internal(
                    get_details_tool,
                    {
                        "workflow_id": current_wf_id,
                        "include_thoughts": True, # Must include thoughts to get chain info
                        "include_actions": False, # Not needed for this task
                        "include_artifacts": False,
                        "include_memories": False
                    },
                    record_action=False # Internal setup action, don't log as agent action
                )
                # Check if the tool call was successful and returned data
                if details.get("success"):
                    thought_chains = details.get("thought_chains")
                    # Check if thought_chains is a non-empty list
                    if isinstance(thought_chains, list) and thought_chains:
                        # Assume the first chain in the list is the primary one (usually ordered by creation time)
                        first_chain = thought_chains[0]
                        chain_id = first_chain.get("thought_chain_id")
                        if chain_id:
                            # Set the state variable
                            self.state.current_thought_chain_id = chain_id
                            self.logger.info(f"Set current_thought_chain_id to primary chain: {_fmt_id(self.state.current_thought_chain_id)} for workflow {_fmt_id(current_wf_id)}")
                            return # Successfully set
                        else:
                             # Log warning if the found chain object is missing the ID
                             self.logger.warning(f"Primary thought chain found for workflow {current_wf_id}, but it lacks an ID in the details.")
                    else:
                         # Log warning if no chains were found for the workflow
                         self.logger.warning(f"Could not find any thought chains in details for workflow {current_wf_id}.")
                else:
                    # Log the error message returned by the tool if it failed
                    self.logger.error(f"Tool '{get_details_tool}' failed while trying to get default thought chain: {details.get('error')}")

            except Exception as e:
                # Log any exceptions encountered during the tool call itself
                self.logger.error(f"Error fetching workflow details for default chain: {e}", exc_info=False)
        else:
            # Log warning if the required tool is unavailable
            self.logger.warning(f"Cannot set default thought chain ID: Tool '{get_details_tool}' unavailable.")

        # Fallback message if the chain ID couldn't be set for any reason
        self.logger.info(f"Could not determine primary thought chain ID for WF {_fmt_id(current_wf_id)}. Will use default on first thought.")


    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """
        Efficiently checks if a given workflow ID exists using the UMS.

        Args:
            workflow_id: The workflow ID to check.

        Returns:
            True if the workflow exists, False otherwise.
        """
        self.logger.debug(f"Checking existence of workflow {_fmt_id(workflow_id)} using {TOOL_GET_WORKFLOW_DETAILS}.")
        tool_name = TOOL_GET_WORKFLOW_DETAILS
        # Check if the required tool is available
        if not self._find_tool_server(tool_name):
            self.logger.error(f"Cannot check workflow existence: Tool {tool_name} unavailable.")
            # If tool is unavailable, we cannot confirm existence, assume False for safety
            return False
        try:
            # Call get_workflow_details with minimal includes for efficiency
            result = await self._execute_tool_call_internal(
                tool_name,
                {
                    "workflow_id": workflow_id,
                    "include_actions": False,
                    "include_artifacts": False,
                    "include_thoughts": False,
                    "include_memories": False
                },
                record_action=False # This is an internal check
            )
            # If the tool call returns success=True, the workflow exists
            return isinstance(result, dict) and result.get("success", False)
        except ToolInputError as e:
            # A ToolInputError often indicates the ID was not found
            self.logger.debug(f"Workflow {_fmt_id(workflow_id)} likely not found (ToolInputError: {e}).")
            return False
        except Exception as e:
            # Log other errors encountered during the check
            self.logger.error(f"Error checking workflow {_fmt_id(workflow_id)} existence: {e}", exc_info=False)
            # Assume not found if an error occurs
            return False

    # <<< Start Integration Block: Plan Cycle Detection Helper (Phase 1, Step 3) >>>
    def _detect_plan_cycle(self, plan: List[PlanStep]) -> bool:
        """
        Detects cyclic dependencies in the agent's plan using Depth First Search.

        Args:
            plan: The list of PlanStep objects representing the current plan.

        Returns:
            True if a cycle is detected, False otherwise.
        """
        if not plan: return False # Empty plan has no cycles

        adj: Dict[str, Set[str]] = defaultdict(set) # Adjacency list: step_id -> set(dependency_step_ids)
        plan_step_ids = {step.id for step in plan} # Set of all valid step IDs in the current plan

        # Build the adjacency list from depends_on relationships
        for step in plan:
            for dep_id in step.depends_on:
                # Only add dependency if the target step actually exists in the current plan
                if dep_id in plan_step_ids:
                    adj[step.id].add(dep_id)
                else:
                    # Log if a dependency points to a non-existent step (potential issue)
                    self.logger.warning(f"Plan step {_fmt_id(step.id)} depends on non-existent step {_fmt_id(dep_id)} in current plan.")

        # DFS state tracking:
        # path: nodes currently in the recursion stack for the current DFS path
        # visited: nodes that have been completely explored (all descendants visited)
        path: Set[str] = set()
        visited: Set[str] = set()

        def dfs(node_id: str) -> bool:
            """Recursive DFS function. Returns True if a cycle is detected."""
            path.add(node_id) # Mark node as currently visiting
            visited.add(node_id) # Mark node as visited

            # Explore neighbors (dependencies)
            for neighbor_id in adj[node_id]:
                if neighbor_id in path: # Cycle detected! Neighbor is already in the current path.
                    self.logger.warning(f"Dependency cycle detected involving steps: {_fmt_id(node_id)} -> {_fmt_id(neighbor_id)}")
                    return True
                if neighbor_id not in visited: # If neighbor not visited yet, recurse
                    if dfs(neighbor_id):
                        return True # Propagate cycle detection signal up

            # Finished exploring node_id's descendants, remove from current path
            path.remove(node_id)
            return False # No cycle found starting from this node

        # Run DFS from each node in the plan to check all potential cycles
        for step_id in plan_step_ids:
            if step_id not in visited:
                if dfs(step_id):
                    return True # Cycle found

        # If DFS completes for all nodes without finding a cycle
        return False
    # <<< End Integration Block: Plan Cycle Detection Helper >>>

    # ------------------------------------------------ dependency check --
    async def _check_prerequisites(self, ids: List[str]) -> Tuple[bool, str]:
        """
        Checks if all specified prerequisite action IDs have status 'completed'.

        Args:
            ids: A list of action IDs to check.

        Returns:
            A tuple: (bool: True if all completed, False otherwise,
                      str: Reason for failure or "All dependencies completed.")
        """
        # If no IDs are provided, prerequisites are met by default
        if not ids:
            return True, "No dependencies listed."

        tool_name = TOOL_GET_ACTION_DETAILS # Tool needed to get action status
        # Check if the tool is available
        if not self._find_tool_server(tool_name):
            self.logger.error(f"Cannot check prerequisites: Tool {tool_name} unavailable.")
            return False, f"Tool {tool_name} unavailable."

        self.logger.debug(f"Checking prerequisites: {[_fmt_id(item_id) for item_id in ids]}")
        try:
            # Call the tool internally to get details for the specified action IDs
            res = await self._execute_tool_call_internal(
                tool_name,
                {"action_ids": ids, "include_dependencies": False}, # Don't need nested dependencies for this check
                record_action=False # Internal check
            )

            # Check if the tool call itself failed
            if not res.get("success"):
                error_msg = res.get("error", "Unknown error during dependency check.")
                self.logger.warning(f"Dependency check failed: {error_msg}")
                return False, f"Failed to check dependencies: {error_msg}"

            # Process the returned action details
            actions_found = res.get("actions", [])
            found_ids = {a.get("action_id") for a in actions_found}
            # Check if any requested dependency IDs were not found
            missing_ids = list(set(ids) - found_ids)
            if missing_ids:
                self.logger.warning(f"Dependency actions not found: {[_fmt_id(item_id) for item_id in missing_ids]}")
                return False, f"Dependency actions not found: {[_fmt_id(item_id) for item_id in missing_ids]}"

            # Check the status of each found dependency action
            incomplete_actions = []
            for action in actions_found:
                if action.get("status") != ActionStatus.COMPLETED.value:
                    # Collect details of incomplete actions for the reason message
                    incomplete_actions.append(
                        f"'{action.get('title', _fmt_id(action.get('action_id')))}' (Status: {action.get('status', 'UNKNOWN')})"
                    )

            # If any actions are not completed, prerequisites are not met
            if incomplete_actions:
                reason = f"Dependencies not completed: {', '.join(incomplete_actions)}"
                self.logger.warning(reason)
                return False, reason

            # If all checks passed, prerequisites are met
            self.logger.debug("All dependencies completed.")
            return True, "All dependencies completed."

        except Exception as e:
            # Log any exceptions during the prerequisite check
            self.logger.error(f"Error during prerequisite check: {e}", exc_info=True)
            return False, f"Exception checking prerequisites: {str(e)}"


    # ---------------------------------------------------- action recording --
    async def _record_action_start_internal(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        planned_dependencies: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Records the start of an action using the UMS tool.

        Optionally records dependencies declared for this action.

        Args:
            tool_name: The name of the tool being executed.
            tool_args: The arguments passed to the tool.
            planned_dependencies: Optional list of action IDs this action depends on.

        Returns:
            The generated `action_id` if successful, otherwise None.
        """
        start_tool = TOOL_RECORD_ACTION_START # Tool for recording action start
        # Check if the recording tool is available
        if not self._find_tool_server(start_tool):
            self.logger.error(f"Cannot record action start: Tool '{start_tool}' unavailable.")
            return None

        # Get the current workflow ID from the state
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning("Cannot record action start: No active workflow ID in state.")
            return None

        # Prepare the payload for the recording tool
        payload = {
            "workflow_id": current_wf_id,
            # Generate a basic title based on the tool name
            "title": f"Execute: {tool_name.split(':')[-1]}", # Use only the tool name part
            "action_type": ActionType.TOOL_USE.value, # Assume it's a tool use action
            "tool_name": tool_name,
            "tool_args": tool_args, # Pass the arguments being used
            "reasoning": f"Agent initiated tool call: {tool_name}", # Basic reasoning
            # Status will likely be set to IN_PROGRESS by the tool itself
        }

        action_id: Optional[str] = None
        try:
            # Call the recording tool internally (don't record this recording action)
            res = await self._execute_tool_call_internal(
                start_tool, payload, record_action=False
            )
            # Check if the recording was successful and returned an action ID
            if res.get("success"):
                action_id = res.get("action_id")
                if action_id:
                    self.logger.debug(f"Action started: {_fmt_id(action_id)} for tool {tool_name}")
                    # If dependencies were provided, record them *after* getting the action ID
                    if planned_dependencies:
                        await self._record_action_dependencies_internal(action_id, planned_dependencies)
                else:
                    # Log if the tool succeeded but didn't return an ID
                    self.logger.warning(f"Tool {start_tool} succeeded but returned no action_id.")
            else:
                # Log if the recording tool itself failed
                self.logger.error(f"Failed to record action start for {tool_name}: {res.get('error')}")

        except Exception as e:
            # Log any exceptions during the recording process
            self.logger.error(f"Exception recording action start for {tool_name}: {e}", exc_info=True)

        return action_id # Return the action ID or None


    async def _record_action_dependencies_internal(
        self,
        source_id: str, # The action being started
        target_ids: List[str], # The actions it depends on
    ) -> None:
        """
        Records dependencies (source_id REQUIRES target_id) in the UMS.

        Args:
            source_id: The ID of the action that has dependencies.
            target_ids: A list of action IDs that `source_id` depends on.
        """
        # Basic validation
        if not source_id or not target_ids:
            self.logger.debug("Skipping dependency recording: Missing source or target IDs.")
            return
        # Filter out empty/invalid target IDs and self-references
        valid_target_ids = {tid for tid in target_ids if tid and tid != source_id}
        if not valid_target_ids:
            self.logger.debug(f"No valid dependencies to record for source action {_fmt_id(source_id)}.")
            return

        dep_tool = TOOL_ADD_ACTION_DEPENDENCY # Tool for adding dependencies
        # Check tool availability
        if not self._find_tool_server(dep_tool):
            self.logger.error(f"Cannot record dependencies: Tool '{dep_tool}' unavailable.")
            return

        # Get current workflow ID (should match the source action's workflow)
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"Cannot record dependencies for action {_fmt_id(source_id)}: No active workflow ID.")
            return

        self.logger.debug(f"Recording {len(valid_target_ids)} dependencies for action {_fmt_id(source_id)}: depends on {[_fmt_id(tid) for tid in valid_target_ids]}")

        # Create tasks to record each dependency concurrently
        tasks = []
        for target_id in valid_target_ids:
            args = {
                # workflow_id might be inferred by the tool, but pass for robustness
                "workflow_id": current_wf_id,
                "source_action_id": source_id,
                "target_action_id": target_id,
                "dependency_type": "requires", # Assuming 'requires' is the default/most common type here
            }
            # Call the dependency tool internally for each target ID
            task = asyncio.create_task(
                self._execute_tool_call_internal(dep_tool, args, record_action=False)
            )
            tasks.append(task)

        # Wait for all dependency recording tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        target_list = list(valid_target_ids) # Ensure consistent ordering for results
        # Log any failures encountered during dependency recording
        for i, res in enumerate(results):
            target_id = target_list[i] # Get corresponding target ID
            if isinstance(res, Exception):
                self.logger.error(f"Error recording dependency {_fmt_id(source_id)} -> {_fmt_id(target_id)}: {res}", exc_info=False)
            elif isinstance(res, dict) and not res.get("success"):
                # Log if the tool call itself failed
                self.logger.warning(f"Failed recording dependency {_fmt_id(source_id)} -> {_fmt_id(target_id)}: {res.get('error')}")
            # else: Successfully recorded


    async def _record_action_completion_internal(
        self,
        action_id: str,
        result: Dict[str, Any], # The final result dict from the tool execution
    ) -> None:
        """
        Records the completion or failure status and result for a given action ID.

        Args:
            action_id: The ID of the action to mark as completed/failed.
            result: The result dictionary returned by `_execute_tool_call_internal`.
        """
        completion_tool = TOOL_RECORD_ACTION_COMPLETION # Tool for recording completion
        # Check tool availability
        if not self._find_tool_server(completion_tool):
            self.logger.error(f"Cannot record action completion: Tool '{completion_tool}' unavailable.")
            return

        # Determine the final status based on the 'success' key in the result dict
        status = (
            ActionStatus.COMPLETED.value
            if isinstance(result, dict) and result.get("success")
            else ActionStatus.FAILED.value
        )

        # Get current workflow ID (should match the action's workflow)
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"Cannot record completion for action {_fmt_id(action_id)}: No active workflow ID.")
            return

        # Prepare payload for the completion tool
        payload = {
            # Pass workflow_id for context, though tool might infer from action_id
            "workflow_id": current_wf_id,
            "action_id": action_id,
            "status": status,
            # Pass the entire result dictionary to be stored/summarized by the UMS tool
            "tool_result": result,
            # Optionally, extract a summary here if needed by the tool, but UMS likely handles this
            # "summary": result.get("summary") or result.get("message") or str(result.get("data"))[:100]
        }

        try:
            # Call the completion tool internally
            completion_result = await self._execute_tool_call_internal(
                completion_tool, payload, record_action=False # Don't record this meta-action
            )
            # Log success or failure of the recording itself
            if completion_result.get("success"):
                self.logger.debug(f"Action completion recorded for {_fmt_id(action_id)} (Status: {status})")
            else:
                self.logger.error(f"Failed to record action completion for {_fmt_id(action_id)}: {completion_result.get('error')}")
        except Exception as e:
            # Log any exceptions during the completion recording call
            self.logger.error(f"Exception recording action completion for {_fmt_id(action_id)}: {e}", exc_info=True)

    # ---------------------------------------------------- auto‑link helper --
    async def _run_auto_linking(
        self,
        memory_id: str, # The ID of the newly created/updated memory
        *,
        workflow_id: Optional[str], # Snapshotted workflow ID
        context_id: Optional[str], # Snapshotted context ID (unused here but passed)
    ) -> None:
        """
        Background task to find semantically similar memories and automatically
        create links to them. Uses richer link types based on memory types.
        """
        # (Keep the integrated _run_auto_linking method logic)
        # Check if the agent's current workflow matches the one snapshotted when the task was created
        # Also check if shutdown has been requested
        if workflow_id != self.state.workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping auto-linking for {_fmt_id(memory_id)}: Workflow changed ({_fmt_id(self.state.workflow_id)} vs {_fmt_id(workflow_id)}) or shutdown signaled.")
            return

        try:
            # Validate inputs
            if not memory_id or not workflow_id:
                self.logger.debug(f"Skipping auto-linking: Missing memory_id ({_fmt_id(memory_id)}) or workflow_id ({_fmt_id(workflow_id)}).")
                return

            # Introduce a small random delay to distribute load
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            # Check shutdown again after sleep
            if self._shutdown_event.is_set(): return

            self.logger.debug(f"Attempting auto-linking for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}...")

            # 1. Get details of the source memory (the one just created/updated)
            source_mem_details_result = await self._execute_tool_call_internal(
                TOOL_GET_MEMORY_BY_ID, {"memory_id": memory_id, "include_links": False}, record_action=False
            )
            # Ensure retrieval succeeded and the memory still belongs to the expected workflow
            if not source_mem_details_result.get("success") or source_mem_details_result.get("workflow_id") != workflow_id:
                self.logger.warning(f"Auto-linking failed for {_fmt_id(memory_id)}: Couldn't retrieve source memory or workflow mismatch.")
                return
            source_mem = source_mem_details_result # Result is the memory dict

            # 2. Determine text for similarity search (use description or truncated content)
            query_text = source_mem.get("description", "") or source_mem.get("content", "")[:200] # Limit content length
            if not query_text:
                self.logger.debug(f"Skipping auto-linking for {_fmt_id(memory_id)}: No description or content for query.")
                return

            # 3. Perform semantic search for similar memories within the same workflow
            # Prefer hybrid search if available, fall back to semantic
            search_tool = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
            if not self._find_tool_server(search_tool):
                self.logger.warning(f"Skipping auto-linking: Tool {search_tool} unavailable.")
                return

            search_args = {
                "workflow_id": workflow_id, # Search within the snapshotted workflow
                "query": query_text,
                "limit": self.auto_linking_max_links + 1, # Fetch one extra to filter self
                "threshold": self.auto_linking_threshold, # Use configured threshold
                "include_content": False # Don't need full content of similar items
            }
            # Adjust weights if using hybrid search (prioritize semantic similarity)
            if search_tool == TOOL_HYBRID_SEARCH:
                search_args.update({"semantic_weight": 0.8, "keyword_weight": 0.2})

            similar_results = await self._execute_tool_call_internal(
                search_tool, search_args, record_action=False
            )
            if not similar_results.get("success"):
                self.logger.warning(f"Auto-linking search failed for {_fmt_id(memory_id)}: {similar_results.get('error')}")
                return

            # 4. Process results and create links
            link_count = 0
            # Determine which key holds the similarity score based on the tool used
            score_key = "hybrid_score" if search_tool == TOOL_HYBRID_SEARCH else "similarity"

            for similar_mem_summary in similar_results.get("memories", []):
                # Check shutdown flag frequently during loop
                if self._shutdown_event.is_set(): break

                target_id = similar_mem_summary.get("memory_id")
                similarity_score = similar_mem_summary.get(score_key, 0.0)

                # Skip linking to self
                if not target_id or target_id == memory_id: continue

                # 5. Get details of the potential target memory for richer link type inference
                target_mem_details_result = await self._execute_tool_call_internal(
                    TOOL_GET_MEMORY_BY_ID, {"memory_id": target_id, "include_links": False}, record_action=False
                )
                # Ensure target retrieval succeeded and it's in the same workflow
                if not target_mem_details_result.get("success") or target_mem_details_result.get("workflow_id") != workflow_id:
                    self.logger.debug(f"Skipping link target {_fmt_id(target_id)}: Not found or workflow mismatch.")
                    continue
                target_mem = target_mem_details_result

                # 6. Infer a more specific link type based on memory types
                inferred_link_type = LinkType.RELATED.value # Default link type
                source_type = source_mem.get("memory_type")
                target_type = target_mem.get("memory_type")

                # Example inference rules (can be expanded)
                if source_type == MemoryType.INSIGHT.value and target_type == MemoryType.FACT.value: inferred_link_type = LinkType.SUPPORTS.value
                elif source_type == MemoryType.FACT.value and target_type == MemoryType.INSIGHT.value: inferred_link_type = LinkType.SUPPORTS.value # Or maybe GENERALIZES/SPECIALIZES?
                elif source_type == MemoryType.QUESTION.value and target_type == MemoryType.FACT.value: inferred_link_type = LinkType.REFERENCES.value
                # >>>>> PRESERVED ORIGINAL LINK TYPE RULES <<<<<
                # (Can add more rules here based on analysis)
                # elif source_type == MemoryType.HYPOTHESIS.value and target_type == MemoryType.EVIDENCE.value: inferred_link_type = LinkType.SUPPORTS.value # Assuming evidence supports hypothesis
                # elif source_type == MemoryType.EVIDENCE.value and target_type == MemoryType.HYPOTHESIS.value: inferred_link_type = LinkType.SUPPORTS.value
                # ... add other rules ...

                # 7. Create the link using the UMS tool
                link_tool_name = TOOL_CREATE_LINK
                if not self._find_tool_server(link_tool_name):
                    self.logger.warning(f"Cannot create link: Tool {link_tool_name} unavailable.")
                    break # Stop trying if link tool is missing

                link_args = {
                    # workflow_id is usually inferred by the tool from memory IDs
                    "source_memory_id": memory_id,
                    "target_memory_id": target_id,
                    "link_type": inferred_link_type,
                    "strength": round(similarity_score, 3), # Use similarity score as strength
                    "description": f"Auto-link ({inferred_link_type}) based on similarity ({score_key})"
                }
                link_result = await self._execute_tool_call_internal(
                    link_tool_name, link_args, record_action=False # Don't record link creation as primary action
                )

                # Log success or failure of link creation
                if link_result.get("success"):
                    link_count += 1
                    self.logger.debug(f"Auto-linked memory {_fmt_id(memory_id)} to {_fmt_id(target_id)} ({inferred_link_type}, score: {similarity_score:.2f})")
                else:
                    # Log failure, but continue trying other potential links
                    self.logger.warning(f"Failed to auto-create link {_fmt_id(memory_id)}->{_fmt_id(target_id)}: {link_result.get('error')}")

                # Stop if max links reached for this source memory
                if link_count >= self.auto_linking_max_links:
                    self.logger.debug(f"Reached auto-linking limit ({self.auto_linking_max_links}) for memory {_fmt_id(memory_id)}.")
                    break
                # Small delay between creating links if multiple are found
                await asyncio.sleep(0.1)

        except Exception as e:
            # Catch and log any errors occurring within the background task itself
            self.logger.warning(f"Error in auto-linking task for {_fmt_id(memory_id)}: {e}", exc_info=False)


    # ---------------------------------------------------- promotion helper --
    async def _check_and_trigger_promotion(
        self,
        memory_id: str, # The ID of the memory to check
        *,
        workflow_id: Optional[str], # Snapshotted workflow ID
        context_id: Optional[str], # Snapshotted context ID (unused but passed)
    ):
        """
        Checks if a specific memory meets criteria for promotion to a higher
        cognitive level and calls the UMS tool to perform the promotion if eligible.
        Intended to be run as a background task.
        """
        # (Keep the integrated _check_and_trigger_promotion method logic)
        # Abort if workflow context has changed or shutdown is signaled
        if workflow_id != self.state.workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping promotion check for {_fmt_id(memory_id)}: Workflow changed ({_fmt_id(self.state.workflow_id)} vs {_fmt_id(workflow_id)}) or shutdown.")
            return

        promotion_tool_name = TOOL_PROMOTE_MEM # Tool for checking/promoting
        # Basic validation
        if not memory_id or not self._find_tool_server(promotion_tool_name):
            self.logger.debug(f"Skipping promotion check for {_fmt_id(memory_id)}: Invalid ID or tool '{promotion_tool_name}' unavailable.")
            return

        try:
            # Optional slight delay
            await asyncio.sleep(random.uniform(0.1, 0.4))
            # Check shutdown again after sleep
            if self._shutdown_event.is_set(): return

            self.logger.debug(f"Checking promotion potential for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}...")
            # Execute the promotion check tool internally
            # Workflow ID is likely inferred by the tool from memory_id
            promotion_result = await self._execute_tool_call_internal(
                promotion_tool_name, {"memory_id": memory_id}, record_action=False # Don't record check as action
            )

            # Log the outcome of the promotion check
            if promotion_result.get("success"):
                if promotion_result.get("promoted"):
                    # Log successful promotion clearly
                    self.logger.info(f"Memory {_fmt_id(memory_id)} promoted from {promotion_result.get('previous_level')} to {promotion_result.get('new_level')}.", emoji_key="arrow_up")
                else:
                    # Log the reason if promotion didn't occur but check was successful
                    self.logger.debug(f"Memory {_fmt_id(memory_id)} not promoted: {promotion_result.get('reason')}")
            else:
                 # Log if the promotion check tool itself failed
                 self.logger.warning(f"Promotion check tool failed for {_fmt_id(memory_id)}: {promotion_result.get('error')}")

        except Exception as e:
            # Log any exceptions within the promotion check task
            self.logger.warning(f"Error in memory promotion check task for {_fmt_id(memory_id)}: {e}", exc_info=False)


    # ------------------------------------------------------ execute tool call --
    async def _execute_tool_call_internal(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        record_action: bool = True,
        planned_dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Central handler for executing tool calls via MCPClient.

        Includes:
        - Server lookup.
        - Automatic injection of workflow/context IDs.
        - Prerequisite dependency checking.
        - Handling of the internal AGENT_TOOL_UPDATE_PLAN.
        - Optional recording of action start/completion/dependencies.
        - Retry logic for idempotent tools.
        - Result parsing and standardization.
        - **Enhanced error handling/categorization** and state updates (last_error_details).
        - Triggering relevant background tasks (auto-linking, promotion check).
        - Updating last_action_summary state.
        - Handling workflow side effects (creation/completion).
        """
        # <<< Start Integration Block: Enhance _execute_tool_call_internal (Phase 1, Step 4) >>>
        # --- Step 1: Server Lookup ---
        target_server = self._find_tool_server(tool_name)
        # Handle case where tool server is not found, except for the internal agent tool
        if not target_server and tool_name != AGENT_TOOL_UPDATE_PLAN:
            err = f"Tool server unavailable for {tool_name}"
            self.logger.error(err)
            # Set error details for the main loop to see - Enhanced Category
            self.state.last_error_details = {"tool": tool_name, "error": err, "type": "ServerUnavailable", "status_code": 503}
            # Return a failure dictionary consistent with other error returns
            return {"success": False, "error": err, "status_code": 503} # 503 Service Unavailable

        # --- Step 2: Context Injection ---
        # Get current workflow context IDs from state
        current_wf_id = (self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id)
        current_ctx_id = self.state.context_id

        # Make a copy to avoid modifying the original arguments dict passed in
        final_arguments = arguments.copy()
        # Inject workflow_id if missing and relevant for the tool
        if (
            final_arguments.get("workflow_id") is None # Only if not already provided
            and current_wf_id # Only if a workflow is active
            and tool_name # Check if tool_name is valid
            not in { # Exclude tools that don't operate on a specific workflow
                TOOL_CREATE_WORKFLOW, # Creates a new one
                TOOL_LIST_WORKFLOWS, # Lists across potentially many
                "core:list_servers", # Core MCP tool
                "core:get_tool_schema", # Core MCP tool
                AGENT_TOOL_UPDATE_PLAN, # Internal agent tool
            }
        ):
            final_arguments["workflow_id"] = current_wf_id
        # Inject context_id if missing and relevant for the tool
        if (
            final_arguments.get("context_id") is None # Only if not already provided
            and current_ctx_id # Only if a context ID is set
            and tool_name # Check if tool_name is valid
            in { # Tools known to operate on a specific cognitive context ID
                TOOL_GET_WORKING_MEMORY,
                TOOL_OPTIMIZE_WM,
                TOOL_AUTO_FOCUS,
                # Add others here if they accept/require context_id
            }
        ):
            final_arguments["context_id"] = current_ctx_id
        # Inject current thought chain ID for thought recording if not specified
        if (
            final_arguments.get("thought_chain_id") is None # Only if not already provided
            and self.state.current_thought_chain_id # Only if a chain ID is set
            and tool_name == TOOL_RECORD_THOUGHT # Only for the record_thought tool
        ):
            final_arguments["thought_chain_id"] = self.state.current_thought_chain_id

        # --- Step 3: Dependency Check ---
        # If dependencies were declared for this action, check them first
        if planned_dependencies:
            ok, reason = await self._check_prerequisites(planned_dependencies)
            if not ok:
                # If dependencies not met, log warning, set error state, and return failure
                err_msg = f"Prerequisites not met for {tool_name}: {reason}"
                self.logger.warning(err_msg)
                # Store detailed error info for the LLM - Enhanced Category
                self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "DependencyNotMetError", "dependencies": planned_dependencies, "status_code": 412}
                # Signal that replanning is needed due to dependency failure
                self.state.needs_replan = True
                return {"success": False, "error": err_msg, "status_code": 412} # 412 Precondition Failed
            else:
                # Log if dependencies were checked and met
                self.logger.info(f"Prerequisites {[_fmt_id(dep) for dep in planned_dependencies]} met for {tool_name}.")

        # --- Step 4: Handle Internal Agent Tool ---
        # Directly handle the AGENT_TOOL_UPDATE_PLAN without calling MCPClient
        if tool_name == AGENT_TOOL_UPDATE_PLAN:
            try:
                new_plan_data = final_arguments.get("plan", [])
                # Validate the structure of the provided plan data
                if not isinstance(new_plan_data, list):
                    raise ValueError("`plan` argument must be a list of step objects.")
                # Convert list of dicts to list of PlanStep objects (validates structure)
                validated_plan = [PlanStep(**p) for p in new_plan_data]

                # --- Plan Cycle Detection ---
                if self._detect_plan_cycle(validated_plan):
                    err_msg = "Proposed plan contains a dependency cycle."
                    self.logger.error(err_msg)
                    self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "PlanValidationError", "proposed_plan": new_plan_data}
                    self.state.needs_replan = True # Force replan again
                    return {"success": False, "error": err_msg}

                # Replace the agent's current plan
                self.state.current_plan = validated_plan
                # Plan was explicitly updated, so replan flag can be cleared
                self.state.needs_replan = False
                self.logger.info(f"Internal plan update successful. New plan has {len(validated_plan)} steps.")
                # Clear any previous errors after a successful plan update
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                return {"success": True, "message": f"Plan updated with {len(validated_plan)} steps."}
            except (ValidationError, TypeError, ValueError) as e:
                # Handle errors during plan validation or application
                err_msg = f"Failed to validate/apply new plan: {e}"
                self.logger.error(err_msg)
                # Store error details for the LLM - Enhanced Category
                self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "PlanUpdateError", "proposed_plan": final_arguments.get("plan")}
                # Increment error count for internal failures too? Decide policy. Yes, for now.
                self.state.consecutive_error_count += 1
                # Failed plan update requires another attempt at planning
                self.state.needs_replan = True
                return {"success": False, "error": err_msg}

        # --- Step 5: Record Action Start (Optional) ---
        action_id: Optional[str] = None
        # Determine if this tool call should be recorded as a primary agent action
        # Exclude internal/meta tools and calls where record_action is explicitly False
        should_record = record_action and tool_name not in self._INTERNAL_OR_META_TOOLS
        if should_record:
            # Call internal helper to record the action start and dependencies
            action_id = await self._record_action_start_internal(
                 tool_name, final_arguments, planned_dependencies # Pass potentially modified args and dependencies
            )
            # Note: _record_action_start_internal now handles calling _record_action_dependencies_internal

        # --- Step 6: Execute Tool Call (with Retries) ---
        # Define the actual async function to call the tool via MCPClient
        async def _do_call():
            # Ensure None values are stripped *before* sending to MCPClient execute_tool
            # Although MCPClient likely handles this, this adds robustness.
            call_args = {k: v for k, v in final_arguments.items() if v is not None}
            # Target server must be valid here because AGENT_INTERNAL was handled earlier
            return await self.mcp_client.execute_tool(target_server, tool_name, call_args)

        # Get the stats dictionary for this specific tool
        record_stats = self.state.tool_usage_stats[tool_name]
        # Decide if the tool is safe to automatically retry on failure
        idempotent = tool_name in {
            # Read-only operations are generally safe to retry
            TOOL_GET_CONTEXT, TOOL_GET_MEMORY_BY_ID, TOOL_SEMANTIC_SEARCH,
            TOOL_HYBRID_SEARCH, TOOL_GET_ACTION_DETAILS, TOOL_LIST_WORKFLOWS,
            TOOL_COMPUTE_STATS, TOOL_GET_WORKING_MEMORY, TOOL_GET_LINKED_MEMORIES,
            TOOL_GET_ARTIFACTS, TOOL_GET_ARTIFACT_BY_ID, TOOL_GET_ACTION_DEPENDENCIES,
            TOOL_GET_THOUGHT_CHAIN, TOOL_GET_WORKFLOW_DETAILS,
            # Some meta operations might be considered retry-safe
            TOOL_SUMMARIZE_TEXT,
        }

        start_ts = time.time() # Record start time for latency calculation
        res = {} # Initialize result dictionary

        try:
            # Execute the tool call using the retry wrapper
            raw = await self._with_retries(
                _do_call,
                max_retries=3 if idempotent else 1, # Retry only idempotent tools (3 attempts total)
                # Specify exceptions that should trigger a retry attempt
                retry_exceptions=(
                    ToolError, ToolInputError, # Specific MCP errors
                    asyncio.TimeoutError, ConnectionError, # Common network issues
                    APIConnectionError, RateLimitError, APIStatusError, # Anthropic/LLM network/API issues
                ),
            )
            # Calculate execution latency
            latency_ms = (time.time() - start_ts) * 1000
            record_stats["latency_ms_total"] += latency_ms

            # --- Step 7: Process and Standardize Result ---
            # Handle different result formats returned by MCPClient/tools
            if isinstance(raw, dict) and ("success" in raw or "isError" in raw):
                # Assume standard MCP result format with success/isError flag
                is_error = raw.get("isError", not raw.get("success", True))
                # Extract content or error message
                content = raw.get("content", raw.get("error", raw.get("data")))
                if is_error:
                    res = {"success": False, "error": str(content), "status_code": raw.get("status_code")}
                else:
                    # If content itself has a standard structure, use it directly
                    if isinstance(content, dict) and "success" in content:
                        res = content
                    else: # Otherwise, wrap the content under a 'data' key for consistency
                        res = {"success": True, "data": content}
            elif isinstance(raw, dict): # Handle plain dictionaries without explicit success/isError
                 # Assume success if no error indicators present
                 res = {"success": True, "data": raw}
            else:
                # Handle non-dict results (e.g., simple strings, numbers, booleans)
                res = {"success": True, "data": raw}

            # --- Step 8: State Updates and Background Triggers on SUCCESS ---
            if res.get("success"):
                # Update success stats for the tool
                record_stats["success"] += 1

                # --- Background Triggers Integration ---
                # Snapshot the workflow ID *before* potentially starting background tasks
                # (This ensures tasks operate on the workflow active during the trigger event)
                current_wf_id_snapshot = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id  # noqa: F841

                # Trigger auto-linking after storing/updating a memory or recording an artifact with a linked memory
                if tool_name in [TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY] and res.get("memory_id"):
                    mem_id = res["memory_id"]
                    self.logger.debug(f"Queueing auto-link check for memory {_fmt_id(mem_id)}")
                    # Start background task, passing the memory ID
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=mem_id)
                    # Note: _start_background_task automatically snapshots workflow_id/context_id
                if tool_name == TOOL_RECORD_ARTIFACT and res.get("linked_memory_id"):
                    linked_mem_id = res["linked_memory_id"]
                    self.logger.debug(f"Queueing auto-link check for memory linked to artifact: {_fmt_id(linked_mem_id)}")
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=linked_mem_id)

                # Trigger promotion check after retrieving memories
                if tool_name in [TOOL_GET_MEMORY_BY_ID, TOOL_QUERY_MEMORIES, TOOL_HYBRID_SEARCH, TOOL_SEMANTIC_SEARCH, TOOL_GET_WORKING_MEMORY]:
                    mem_ids_to_check = set() # Use set to avoid duplicate checks
                    potential_mems = []
                    # Extract memory IDs from various possible result structures
                    if tool_name == TOOL_GET_MEMORY_BY_ID:
                        # Result might be the memory dict directly or nested under 'data'
                        mem_data = res if "memory_id" in res else res.get("data", {})
                        if isinstance(mem_data, dict): potential_mems = [mem_data]
                    elif tool_name == TOOL_GET_WORKING_MEMORY:
                        potential_mems = res.get("working_memories", [])
                        # Also check the focal memory if present
                        focus_id = res.get("focal_memory_id")
                        if focus_id: mem_ids_to_check.add(focus_id)
                    else: # Query/Search results typically under 'memories' key
                         potential_mems = res.get("memories", [])

                    # Add IDs from the list/dict structures found
                    if isinstance(potential_mems, list):
                        # Limit checks to the top few most relevant results to avoid overload
                        mem_ids_to_check.update(
                            m.get("memory_id") for m in potential_mems[:3] # Check top 3 retrieved
                            if isinstance(m, dict) and m.get("memory_id") # Ensure it's a dict with an ID
                        )

                    # Start background tasks for each unique, valid memory ID found
                    for mem_id in filter(None, mem_ids_to_check): # Filter out any None IDs
                         self.logger.debug(f"Queueing promotion check for retrieved memory {_fmt_id(mem_id)}")
                         self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id)
                # --- Background Triggers Integration End ---

                # Update current thought chain ID if a new one was just created successfully
                if tool_name == TOOL_CREATE_THOUGHT_CHAIN and res.get("success"):
                    # Find the chain ID in the result (might be root or under 'data')
                    chain_data = res if "thought_chain_id" in res else res.get("data", {})
                    if isinstance(chain_data, dict):
                        new_chain_id = chain_data.get("thought_chain_id")
                        if new_chain_id:
                            self.state.current_thought_chain_id = new_chain_id
                            self.logger.info(f"Switched current thought chain to newly created: {_fmt_id(new_chain_id)}")

            else: # Tool failed
                # Update failure stats
                record_stats["failure"] += 1
                # Ensure error details are captured from the result for the LLM context
                # Enhance error details with categorization
                error_type = "ToolExecutionError" # Default category
                status_code = res.get("status_code")
                error_message = res.get("error", "Unknown failure")
                if status_code == 412: error_type = "DependencyNotMetError"
                elif status_code == 503: error_type = "ServerUnavailable"
                elif "input" in str(error_message).lower() or "validation" in str(error_message).lower(): error_type = "InvalidInputError" # Basic keyword check
                elif "timeout" in str(error_message).lower(): error_type = "NetworkError" # Assuming timeout implies network
                # ... (can add more categorization based on status codes or keywords) ...

                self.state.last_error_details = {
                    "tool": tool_name,
                    "args": arguments, # Log the arguments that caused failure
                    "error": error_message,
                    "status_code": status_code,
                    "type": error_type # Store the categorized error type
                }
                # Log the categorized error
                self.logger.warning(f"Tool {tool_name} failed. Type: {error_type}, Error: {error_message}")


            # --- Step 9: Update Last Action Summary ---
            # Create a concise summary of the action's outcome for the next prompt
            summary = ""
            if res.get("success"):
                # Try to find a meaningful summary field in the result or its 'data' payload
                summary_keys = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "chain_id", "state_id", "report", "visualization"]
                data_payload = res.get("data", res) # Check 'data' key or root level
                if isinstance(data_payload, dict):
                    for k in summary_keys:
                        if k in data_payload and data_payload[k]:
                            # Format IDs concisely, use string value for others
                            summary = f"{k}: {_fmt_id(data_payload[k]) if 'id' in k else str(data_payload[k])}"
                            break
                    else: # Fallback if no specific key found in dict
                        data_str = str(data_payload)[:70] # Preview the dict
                        summary = f"Success. Data: {data_str}..." if len(str(data_payload)) > 70 else f"Success. Data: {data_str}"
                else: # Handle non-dict data payload
                     data_str = str(data_payload)[:70] # Preview the data
                     summary = f"Success. Data: {data_str}..." if len(str(data_payload)) > 70 else f"Success. Data: {data_str}"
            else: # If failed
                # Use the structured error details for a more informative summary
                err_type = self.state.last_error_details.get("type", "Unknown") if self.state.last_error_details else "Unknown"
                err_msg = str(res.get('error', 'Unknown Error'))[:100]
                summary = f"Failed ({err_type}): {err_msg}" # Include error type
                if res.get('status_code'): summary += f" (Code: {res['status_code']})" # Add status code if available

            # Update the state variable
            self.state.last_action_summary = f"{tool_name} -> {summary}"
            # Log the outcome
            self.logger.info(self.state.last_action_summary, emoji_key="checkered_flag" if res.get('success') else "warning")


        # --- Step 10: Exception Handling for Tool Call/Retries ---
        # Updated Exception handling to add error categorization
        except (ToolError, ToolInputError) as e:
            # Handle specific MCP exceptions caught during execution or retries
            err_str = str(e); status_code = getattr(e, 'status_code', None)
            # Determine category based on type and status code
            error_type = "InvalidInputError" if isinstance(e, ToolInputError) else "ToolInternalError"
            if status_code == 412: error_type = "DependencyNotMetError"
            self.logger.error(f"Tool Error executing {tool_name}: {err_str}", exc_info=False) # Don't need full trace for these
            res = {"success": False, "error": err_str, "status_code": status_code}
            record_stats["failure"] += 1 # Record failure
            # Store categorized error details for the LLM
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": error_type, "status_code": status_code}
            self.state.last_action_summary = f"{tool_name} -> Failed ({error_type}): {err_str[:100]}"

        except APIConnectionError as e:
            err_str = f"LLM API Connection Error: {e}"
            self.logger.error(err_str, exc_info=False)
            res = {"success": False, "error": err_str}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "NetworkError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: NetworkError"
        except RateLimitError as e:
            err_str = f"LLM Rate Limit Error: {e}"
            self.logger.error(err_str, exc_info=False)
            res = {"success": False, "error": err_str}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "APILimitError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: APILimitError"
        except APIStatusError as e:
            err_str = f"LLM API Error {e.status_code}: {e.message}"
            self.logger.error(f"Anthropic API status error: {e.status_code} - {e.response}", exc_info=False)
            res = {"success": False, "error": err_str, "status_code": e.status_code}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "APIError", "status_code": e.status_code}
            self.state.last_action_summary = f"{tool_name} -> Failed: APIError ({e.status_code})"
        except asyncio.TimeoutError as e: # Catch timeouts from retry wrapper or internal calls
            err_str = f"Operation timed out: {e}"
            self.logger.error(f"Timeout executing {tool_name}: {err_str}", exc_info=False)
            res = {"success": False, "error": err_str}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "TimeoutError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: Timeout"

        except asyncio.CancelledError:
             # Handle task cancellation gracefully (e.g., due to shutdown signal)
             err_str = "Tool execution cancelled."
             self.logger.warning(f"{tool_name} execution was cancelled.")
             res = {"success": False, "error": err_str, "status_code": 499} # Use 499 Client Closed Request
             record_stats["failure"] += 1 # Count cancellation as failure for stats
             self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "CancelledError"}
             self.state.last_action_summary = f"{tool_name} -> Cancelled"
             # Re-raise cancellation to potentially stop the loop if needed
             raise

        except Exception as e:
            # Catch any other unexpected errors during execution or retries
            err_str = str(e)
            self.logger.error(f"Unexpected Error executing {tool_name}: {err_str}", exc_info=True) # Log full traceback
            res = {"success": False, "error": f"Unexpected error: {err_str}"}
            record_stats["failure"] += 1 # Record failure
            # Store categorized error
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "UnexpectedExecutionError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: Unexpected error."

        # --- Step 11: Record Action Completion (if start was recorded) ---
        if action_id:
            # Pass the final result 'res' (success or failure dict) to the completion recorder
            await self._record_action_completion_internal(action_id, res)

        # --- Step 12: Handle Workflow Side Effects ---
        # Call this *after* execution and completion recording, using the final 'res'
        await self._handle_workflow_side_effects(tool_name, final_arguments, res)

        # Return the final processed result dictionary
        return res
        # <<< End Integration Block: Enhance _execute_tool_call_internal >>>


    async def _handle_workflow_side_effects(self, tool_name: str, arguments: Dict, result_content: Dict):
        """
        Handles agent state changes triggered by specific tool outcomes,
        primarily workflow creation and termination.
        """
        # --- Side effects for Workflow Creation ---
        if tool_name == TOOL_CREATE_WORKFLOW and result_content.get("success"):
            # Extract details from the successful creation result
            new_wf_id = result_content.get("workflow_id")
            primary_chain_id = result_content.get("primary_thought_chain_id")
            parent_id = arguments.get("parent_workflow_id") # Get parent from original args

            if new_wf_id:
                # Set the agent's primary workflow ID and context ID to the new one
                self.state.workflow_id = new_wf_id
                self.state.context_id = new_wf_id # Align context ID

                # Manage the workflow stack for sub-workflows
                if parent_id and parent_id in self.state.workflow_stack:
                    # If created as a sub-workflow of an existing one on the stack, push it
                    self.state.workflow_stack.append(new_wf_id)
                    log_prefix = "sub-"
                else:
                    # If it's a new root workflow or parent isn't on stack, reset stack
                    self.state.workflow_stack = [new_wf_id]
                    log_prefix = "new "

                # Set the current thought chain ID for the new workflow
                self.state.current_thought_chain_id = primary_chain_id
                self.logger.info(f"Switched to {log_prefix}workflow: {_fmt_id(new_wf_id)}. Current chain: {_fmt_id(primary_chain_id)}", emoji_key="label")

                # Reset plan, errors, and replan flag for the new workflow context
                self.state.current_plan = [PlanStep(description=f"Start {log_prefix}workflow: '{result_content.get('title', 'Untitled')}'. Goal: {result_content.get('goal', 'Not specified')}.")]
                self.state.consecutive_error_count = 0
                self.state.needs_replan = False
                self.state.last_error_details = None
                # Optionally reset meta-counters for the new workflow
                # self.state.successful_actions_since_reflection = 0
                # self.state.successful_actions_since_consolidation = 0

        # --- Side effects for Workflow Status Update (Completion/Failure/Abandonment) ---
        elif tool_name == TOOL_UPDATE_WORKFLOW_STATUS and result_content.get("success"):
            status = arguments.get("status") # Status requested in the tool call
            wf_id_updated = arguments.get("workflow_id") # Workflow that was updated

            # Check if the *currently active* workflow (top of stack) was the one updated
            if wf_id_updated and self.state.workflow_stack and wf_id_updated == self.state.workflow_stack[-1]:
                # Check if the new status is a terminal one
                is_terminal = status in [
                    WorkflowStatus.COMPLETED.value,
                    WorkflowStatus.FAILED.value,
                    WorkflowStatus.ABANDONED.value
                ]

                if is_terminal:
                    # Remove the finished workflow from the stack
                    finished_wf = self.state.workflow_stack.pop()
                    if self.state.workflow_stack:
                        # If there's a parent workflow remaining on the stack, return to it
                        self.state.workflow_id = self.state.workflow_stack[-1]
                        self.state.context_id = self.state.workflow_id # Realign context ID
                        # Fetch the parent's primary thought chain ID
                        await self._set_default_thought_chain_id()
                        self.logger.info(f"Sub-workflow {_fmt_id(finished_wf)} finished ({status}). Returning to parent {_fmt_id(self.state.workflow_id)}. Current chain: {_fmt_id(self.state.current_thought_chain_id)}", emoji_key="arrow_left")
                        # Force replan in the parent context as the sub-task outcome needs consideration
                        self.state.needs_replan = True
                        self.state.current_plan = [PlanStep(description=f"Returned from sub-workflow {_fmt_id(finished_wf)} (status: {status}). Re-assess parent goal.")]
                        self.state.last_error_details = None # Clear error details from the sub-task
                    else:
                        # If the stack is empty, the root workflow finished
                        self.logger.info(f"Root workflow {_fmt_id(finished_wf)} finished with status: {status}.")
                        # Clear active workflow state
                        self.state.workflow_id = None
                        self.state.context_id = None
                        self.state.current_thought_chain_id = None
                        # Set goal achieved flag only if completed successfully
                        if status == WorkflowStatus.COMPLETED.value:
                             self.state.goal_achieved_flag = True
                        else:
                             # Mark as not achieved if failed or abandoned
                             self.state.goal_achieved_flag = False
                        # Clear the plan as the workflow is over
                        self.state.current_plan = []


    async def _apply_heuristic_plan_update(self, last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        """
        Applies heuristic updates to the plan based on the last action's outcome
        when the LLM doesn't explicitly call `agent:update_plan`.

        This acts as a default progression mechanism. It marks steps completed
        on success, handles failures by marking the step failed and inserting
        an analysis step, and updates meta-cognitive counters.
        """
        # <<< Start Integration Block: Heuristic Plan Update Method (Phase 1, Step 3) >>>
        self.logger.info("Applying heuristic plan update (fallback)...", emoji_key="clipboard")

        # Handle case where plan might be empty (shouldn't usually happen)
        if not self.state.current_plan:
            self.logger.warning("Plan is empty during heuristic update, adding default re-evaluation step.")
            self.state.current_plan = [PlanStep(description="Fallback: Re-evaluate situation.")]
            self.state.needs_replan = True # Force replan if plan was empty
            return

        # Get the step the agent was working on (assumed to be the first)
        current_step = self.state.current_plan[0]
        decision_type = last_decision.get("decision") # What the LLM decided to do

        action_successful = False # Flag to track if the action succeeded for counter updates
        tool_name_executed = last_decision.get("tool_name") # Tool name if a tool was called

        # --- Update plan based on decision type and success ---
        # Case 1: LLM called a tool (and it wasn't AGENT_TOOL_UPDATE_PLAN)
        if decision_type == "call_tool" and tool_name_executed != AGENT_TOOL_UPDATE_PLAN:
            # Check the success status from the tool execution result
            tool_success = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            action_successful = tool_success

            if tool_success:
                # On success, mark step completed and remove from plan
                current_step.status = ActionStatus.COMPLETED.value # Use enum value
                # Generate a concise summary for the plan step
                summary = "Success."
                if isinstance(last_tool_result_content, dict):
                     # Prioritize specific meaningful keys from the result
                     summary_keys = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "chain_id", "state_id", "report", "visualization"]
                     data_payload = last_tool_result_content.get("data", last_tool_result_content) # Look in 'data' or root
                     if isinstance(data_payload, dict):
                         for k in summary_keys:
                              if k in data_payload and data_payload[k]:
                                   # Format IDs or use string representation
                                   summary = f"{k}: {_fmt_id(data_payload[k]) if 'id' in k else str(data_payload[k])}"
                                   break
                         else: # Fallback preview for dicts
                              data_str = str(data_payload)[:70]
                              summary = f"Success. Data: {data_str}..." if len(str(data_payload)) > 70 else f"Success. Data: {data_str}"
                     else: # Handle non-dict data payload
                          data_str = str(data_payload)[:70]
                          summary = f"Success. Data: {data_str}..." if len(str(data_payload)) > 70 else f"Success. Data: {data_str}"

                current_step.result_summary = summary[:150] # Add summary to step, truncated
                self.state.current_plan.pop(0) # Remove completed step from the front
                # If the plan is now empty, add a final analysis step
                if not self.state.current_plan:
                    self.logger.info("Plan completed. Adding final analysis step.")
                    self.state.current_plan.append(PlanStep(description="Plan finished. Analyze overall result and decide if goal is met."))
                self.state.needs_replan = False # Success usually doesn't require immediate replan
            else: # Tool failed
                current_step.status = ActionStatus.FAILED.value # Mark step as failed
                # Extract error message for summary
                error_msg = "Unknown failure"
                if isinstance(last_tool_result_content, dict):
                     # Use the enhanced error details if available
                     error_details = self.state.last_error_details
                     if error_details:
                         error_msg = f"Type: {error_details.get('type', 'Unknown')}, Msg: {error_details.get('error', 'Unknown')}"
                     else: # Fallback to basic error message
                         error_msg = str(last_tool_result_content.get('error', 'Unknown failure'))

                current_step.result_summary = f"Failure: {error_msg[:150]}" # Add error summary
                # Keep the failed step in the plan for context.
                # Insert an analysis step *after* the failed step, if one isn't already there.
                if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Analyze failure of step"):
                    self.state.current_plan.insert(1, PlanStep(description=f"Analyze failure of step '{current_step.description[:30]}...' and replan."))
                self.state.needs_replan = True # Failure always requires replanning

        # Case 2: LLM decided to record a thought
        elif decision_type == "thought_process":
            action_successful = True # Recording a thought is considered a successful step completion heuristically
            current_step.status = ActionStatus.COMPLETED.value
            current_step.result_summary = f"Thought Recorded: {last_decision.get('content','')[:50]}..." # Summary based on thought
            self.state.current_plan.pop(0) # Remove completed step
            # If plan is empty after thought, add next step prompt
            if not self.state.current_plan:
                self.logger.info("Plan completed after thought. Adding next action step.")
                self.state.current_plan.append(PlanStep(description="Decide next action based on recorded thought and overall goal."))
            self.state.needs_replan = False # Recording a thought doesn't force replan

        # Case 3: LLM signaled completion
        elif decision_type == "complete":
            action_successful = True # Achieving goal is success
            # Overwrite plan with a final step (status handled in main loop)
            self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing.", status="completed")]
            self.state.needs_replan = False

        # Case 4: Handle errors or unexpected decisions (including AGENT_TOOL_UPDATE_PLAN failure)
        else:
            action_successful = False # Mark as failure for counter updates
            # Only mark the *current plan step* as failed if it wasn't the plan update tool itself that caused the error state
            if tool_name_executed != AGENT_TOOL_UPDATE_PLAN:
                current_step.status = ActionStatus.FAILED.value
                # Use the last action summary (which should contain the error) for the result summary
                err_summary = self.state.last_action_summary or "Unknown agent error"
                current_step.result_summary = f"Agent/Tool Error: {err_summary[:100]}..."
                # Insert re-evaluation step if not already present
                if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Re-evaluate due to agent error"):
                     self.state.current_plan.insert(1, PlanStep(description="Re-evaluate due to agent error or unclear decision."))
            # Always set needs_replan if an error occurred or the decision was unexpected
            self.state.needs_replan = True

        # --- Update Meta-Cognitive Counters ---
        if action_successful:
            # Reset error counter on any successful progression
            self.state.consecutive_error_count = 0

            # Increment success counters *only* if the action wasn't internal/meta
            # Check if a tool was executed and if it's not in the excluded set
            if tool_name_executed and tool_name_executed not in self._INTERNAL_OR_META_TOOLS:
                 # Use float increments for flexibility
                 self.state.successful_actions_since_reflection += 1.0
                 self.state.successful_actions_since_consolidation += 1.0
                 self.logger.debug(f"Incremented success counters R:{self.state.successful_actions_since_reflection:.1f}, C:{self.state.successful_actions_since_consolidation:.1f} after successful action: {tool_name_executed}")
            elif decision_type == "thought_process":
                 # Option: Count thoughts as partial progress (e.g., 0.5)
                 self.state.successful_actions_since_reflection += 0.5 # Example: count thought as half an action
                 self.state.successful_actions_since_consolidation += 0.5
                 self.logger.debug(f"Incremented success counters R:{self.state.successful_actions_since_reflection:.1f}, C:{self.state.successful_actions_since_consolidation:.1f} after thought recorded.")

        else: # Action failed or was an error condition handled above
            # Increment consecutive error count
            self.state.consecutive_error_count += 1
            self.logger.warning(f"Consecutive error count increased to: {self.state.consecutive_error_count}")
            # Reset reflection counter immediately on error to encourage faster reflection
            if self.state.successful_actions_since_reflection > 0:
                 self.logger.info(f"Resetting reflection counter due to error (was {self.state.successful_actions_since_reflection:.1f}).")
                 self.state.successful_actions_since_reflection = 0
                 # Policy Decision: Keep consolidation counter running unless error rate is very high (handled in _adapt_thresholds)

        # --- Log Final Plan State ---
        log_plan_msg = f"Plan updated heuristically. Steps remaining: {len(self.state.current_plan)}. "
        if self.state.current_plan:
            next_step = self.state.current_plan[0]
            depends_str = f"Depends: {[_fmt_id(d) for d in next_step.depends_on]}" if next_step.depends_on else "Depends: None"
            log_plan_msg += f"Next: '{next_step.description[:60]}...' (Status: {next_step.status}, {depends_str})"
        else:
            log_plan_msg += "Plan is now empty."
        self.logger.info(log_plan_msg, emoji_key="clipboard")
        # <<< End Integration Block: Heuristic Plan Update Method >>>


    # ------------------------------------------------ adaptive thresholds --
    def _adapt_thresholds(self, stats: Dict[str, Any]) -> None:
        """
        Adjusts reflection and consolidation thresholds based on memory statistics
        and tool usage patterns to dynamically control meta-cognition frequency.
        **Enhanced with more stats, nuanced calculation, and dampening.**
        """
        # <<< Start Integration Block: Enhance _adapt_thresholds (Phase 1, Step 2) >>>
        # Validate stats input
        if not stats or not stats.get("success"):
             self.logger.warning("Cannot adapt thresholds: Invalid or failed stats received.")
             return

        self.logger.debug(f"Adapting thresholds based on stats: {stats}")
        # Use dampening factor from constant
        adjustment_dampening = THRESHOLD_ADAPTATION_DAMPENING
        changed = False # Flag to log if any threshold changed

        # --- Consolidation Threshold Adaptation ---
        episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
        total_memories = stats.get("total_memories", 1) # Avoid division by zero
        episodic_ratio = episodic_count / total_memories if total_memories > 0 else 0
        # Target range for episodic ratio (e.g., ideally keep it below 30%)
        target_episodic_ratio_upper = 0.30
        target_episodic_ratio_lower = 0.10
        # Calculate deviation from the middle of the target range
        mid_target_ratio = (target_episodic_ratio_upper + target_episodic_ratio_lower) / 2
        ratio_deviation = episodic_ratio - mid_target_ratio
        # Calculate adjustment: more negative deviation (low ratio) -> increase threshold
        # More positive deviation (high ratio) -> decrease threshold
        # Scale adjustment based on current threshold (larger thresholds allow bigger steps)
        consolidation_adjustment = -math.ceil(ratio_deviation * self.state.current_consolidation_threshold * 2.0) # Factor of 2 controls sensitivity

        # Apply dampening to the adjustment
        dampened_adjustment = int(consolidation_adjustment * adjustment_dampening)
        if dampened_adjustment != 0:
            old_threshold = self.state.current_consolidation_threshold
            # Calculate potential new threshold, enforcing MIN/MAX bounds
            potential_new = max(MIN_CONSOLIDATION_THRESHOLD, min(MAX_CONSOLIDATION_THRESHOLD, old_threshold + dampened_adjustment))
            # Apply change only if it's different from the current threshold
            if potential_new != old_threshold:
                change_direction = "Lowering" if dampened_adjustment < 0 else "Raising"
                self.logger.info(
                    f"{change_direction} consolidation threshold: {old_threshold} -> {potential_new} "
                    f"(Episodic Ratio: {episodic_ratio:.1%}, Deviation: {ratio_deviation:+.1%}, Adjustment: {dampened_adjustment})"
                )
                self.state.current_consolidation_threshold = potential_new
                changed = True

        # --- Reflection Threshold Adaptation ---
        # Use tool usage stats accumulated in agent state
        total_calls = sum(v.get("success", 0) + v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        total_failures = sum(v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        # Calculate failure rate, avoid division by zero, require minimum calls for statistical significance
        min_calls_for_rate = 5 # Need at least 5 calls to calculate a meaningful rate
        failure_rate = (total_failures / total_calls) if total_calls >= min_calls_for_rate else 0.0
        # Target failure rate (e.g., aim for below 10%)
        target_failure_rate = 0.10
        failure_deviation = failure_rate - target_failure_rate
        # Calculate adjustment: more positive deviation (high failure) -> decrease threshold
        # More negative deviation (low failure) -> increase threshold
        reflection_adjustment = -math.ceil(failure_deviation * self.state.current_reflection_threshold * 3.0) # Factor of 3 controls sensitivity

        # Apply dampening
        dampened_adjustment = int(reflection_adjustment * adjustment_dampening)
        if dampened_adjustment != 0 and total_calls >= min_calls_for_rate: # Only adjust if enough calls
            old_threshold = self.state.current_reflection_threshold
            # Calculate potential new threshold, enforcing MIN/MAX bounds
            potential_new = max(MIN_REFLECTION_THRESHOLD, min(MAX_REFLECTION_THRESHOLD, old_threshold + dampened_adjustment))
            # Apply change only if different
            if potential_new != old_threshold:
                change_direction = "Lowering" if dampened_adjustment < 0 else "Raising"
                self.logger.info(
                    f"{change_direction} reflection threshold: {old_threshold} -> {potential_new} "
                    f"(Failure Rate: {failure_rate:.1%}, Deviation: {failure_deviation:+.1%}, Adjustment: {dampened_adjustment})"
                )
                self.state.current_reflection_threshold = potential_new
                changed = True

        # Log if no changes were made
        if not changed:
             self.logger.debug("No threshold adjustments triggered based on current stats/heuristics.")
        # <<< End Integration Block: Enhance _adapt_thresholds >>>


    # ------------------------------------------------ periodic task runner --
    async def _run_periodic_tasks(self):
        """
        Runs scheduled cognitive maintenance and enhancement tasks periodically.

        Checks intervals and thresholds to trigger tasks like reflection,
        consolidation, working memory optimization, focus updates, promotion checks,
        statistics computation, threshold adaptation, and memory maintenance.
        Executes triggered tasks sequentially within the loop cycle.
        """
        # Prevent running if no workflow or shutting down
        if not self.state.workflow_id or not self.state.context_id or self._shutdown_event.is_set():
            return

        # List to hold tasks scheduled for this cycle: (tool_name, args_dict)
        tasks_to_run: List[Tuple[str, Dict]] = []
        # List to track reasons for triggering tasks (for logging)
        trigger_reasons: List[str] = []

        # Check tool availability once per cycle for efficiency
        reflection_tool_available = self._find_tool_server(TOOL_REFLECTION) is not None
        consolidation_tool_available = self._find_tool_server(TOOL_CONSOLIDATION) is not None
        optimize_wm_tool_available = self._find_tool_server(TOOL_OPTIMIZE_WM) is not None
        auto_focus_tool_available = self._find_tool_server(TOOL_AUTO_FOCUS) is not None
        promote_mem_tool_available = self._find_tool_server(TOOL_PROMOTE_MEM) is not None
        stats_tool_available = self._find_tool_server(TOOL_COMPUTE_STATS) is not None
        maintenance_tool_available = self._find_tool_server(TOOL_DELETE_EXPIRED_MEMORIES) is not None

        # --- Tier 1: Highest Priority - Stats Check & Adaptation ---
        # Increment counter for stats adaptation interval
        self.state.loops_since_stats_adaptation += 1
        # Check if interval reached
        if self.state.loops_since_stats_adaptation >= STATS_ADAPTATION_INTERVAL:
            if stats_tool_available:
                trigger_reasons.append("StatsInterval")
                try:
                    # Fetch current statistics for the workflow
                    stats = await self._execute_tool_call_internal(
                        TOOL_COMPUTE_STATS, {"workflow_id": self.state.workflow_id}, record_action=False
                    )
                    if stats.get("success"):
                        # Adapt thresholds based on the fetched stats
                        self._adapt_thresholds(stats)
                        # Example: Potentially trigger consolidation *now* if stats show high episodic count
                        episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
                        # Trigger if count significantly exceeds the *current dynamic* threshold
                        if episodic_count > (self.state.current_consolidation_threshold * 2.0) and consolidation_tool_available:
                            # Check if consolidation isn't already scheduled for other reasons this cycle
                            if not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run):
                                self.logger.info(f"High episodic count ({episodic_count}) detected via stats, scheduling consolidation.")
                                # Schedule consolidation task
                                tasks_to_run.append((TOOL_CONSOLIDATION, {
                                    "workflow_id": self.state.workflow_id,
                                    "consolidation_type": "summary", # Default to summary
                                    # Filter consolidation sources to episodic memories
                                    "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                                    "max_source_memories": self.consolidation_max_sources
                                }))
                                trigger_reasons.append(f"HighEpisodic({episodic_count})")
                                # Reset consolidation counter as we're triggering it now based on stats
                                self.state.successful_actions_since_consolidation = 0
                    else:
                        # Log if stats computation failed
                        self.logger.warning(f"Failed to compute stats for adaptation: {stats.get('error')}")
                except Exception as e:
                    # Log errors during the stats/adaptation process
                    self.logger.error(f"Error during stats computation/adaptation: {e}", exc_info=False)
                finally:
                     # Reset the interval counter regardless of success/failure
                     self.state.loops_since_stats_adaptation = 0
            else:
                # Log if stats tool is unavailable
                self.logger.warning(f"Skipping stats/adaptation: Tool {TOOL_COMPUTE_STATS} not available")

        # --- Tier 2: Reflection & Consolidation (Based on dynamic thresholds) ---
        # Reflection Trigger (Check replan flag OR success counter vs. *dynamic* threshold)
        needs_reflection = self.state.needs_replan or self.state.successful_actions_since_reflection >= self.state.current_reflection_threshold
        if needs_reflection:
            if reflection_tool_available:
                # Check if reflection isn't already scheduled this cycle
                if not any(task[0] == TOOL_REFLECTION for task in tasks_to_run):
                    # Cycle through different reflection types for variety
                    reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
                    # Schedule reflection task
                    tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
                    # Log the specific reason for triggering
                    reason_str = f"ReplanNeeded({self.state.needs_replan})" if self.state.needs_replan else f"SuccessCount({self.state.successful_actions_since_reflection:.1f}>={self.state.current_reflection_threshold})"
                    trigger_reasons.append(f"Reflect({reason_str})")
                    # Reset the success counter and advance the cycle index
                    self.state.successful_actions_since_reflection = 0
                    self.state.reflection_cycle_index += 1
            else:
                # Log if tool unavailable, still reset counter to prevent immediate re-trigger
                self.logger.warning(f"Skipping reflection: Tool {TOOL_REFLECTION} not available")
                self.state.successful_actions_since_reflection = 0

        # Consolidation Trigger (Check success counter vs. *dynamic* threshold)
        needs_consolidation = self.state.successful_actions_since_consolidation >= self.state.current_consolidation_threshold
        if needs_consolidation:
            if consolidation_tool_available:
                # Check if consolidation isn't already scheduled this cycle
                if not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run):
                    # Schedule consolidation task (e.g., summarize episodic memories)
                    tasks_to_run.append((TOOL_CONSOLIDATION, {
                        "workflow_id": self.state.workflow_id,
                        "consolidation_type": "summary",
                        "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                        "max_source_memories": self.consolidation_max_sources
                    }))
                    trigger_reasons.append(f"ConsolidateThreshold({self.state.successful_actions_since_consolidation:.1f}>={self.state.current_consolidation_threshold})")
                    # Reset the success counter
                    self.state.successful_actions_since_consolidation = 0
            else:
                # Log if tool unavailable, still reset counter
                self.logger.warning(f"Skipping consolidation: Tool {TOOL_CONSOLIDATION} not available")
                self.state.successful_actions_since_consolidation = 0

        # --- Tier 3: Optimization & Focus (Based on loop interval) ---
        # Increment optimization interval counter
        self.state.loops_since_optimization += 1
        # Check if interval reached
        if self.state.loops_since_optimization >= OPTIMIZATION_LOOP_INTERVAL:
            # Schedule working memory optimization if tool available
            if optimize_wm_tool_available:
                tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.state.context_id}))
                trigger_reasons.append("OptimizeInterval")
            else:
                self.logger.warning(f"Skipping optimization: Tool {TOOL_OPTIMIZE_WM} not available")

            # Schedule automatic focus update if tool available
            if auto_focus_tool_available:
                tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.state.context_id}))
                trigger_reasons.append("FocusUpdate")
            else:
                self.logger.warning(f"Skipping auto-focus: Tool {TOOL_AUTO_FOCUS} not available")

            # Reset the interval counter
            self.state.loops_since_optimization = 0

        # --- Tier 4: Promotion Check (Based on loop interval) ---
        # Increment promotion check interval counter
        self.state.loops_since_promotion_check += 1
        # Check if interval reached
        if self.state.loops_since_promotion_check >= MEMORY_PROMOTION_LOOP_INTERVAL:
            if promote_mem_tool_available:
                # Schedule the internal check function, not the tool directly
                tasks_to_run.append(("CHECK_PROMOTIONS", {})) # Special marker task name
                trigger_reasons.append("PromotionInterval")
            else:
                self.logger.warning(f"Skipping promotion check: Tool {TOOL_PROMOTE_MEM} unavailable.")
            # Reset the interval counter
            self.state.loops_since_promotion_check = 0

        # --- Tier 5: Lowest Priority - Maintenance ---
        # Increment maintenance interval counter
        self.state.loops_since_maintenance += 1
        # Check if interval reached
        if self.state.loops_since_maintenance >= MAINTENANCE_INTERVAL:
            if maintenance_tool_available:
                # Schedule memory expiration task
                tasks_to_run.append((TOOL_DELETE_EXPIRED_MEMORIES, {})) # No args needed usually
                trigger_reasons.append("MaintenanceInterval")
                self.state.loops_since_maintenance = 0 # Reset interval counter
            else:
                # Log if tool unavailable
                self.logger.warning(f"Skipping maintenance: Tool {TOOL_DELETE_EXPIRED_MEMORIES} not available")


        # --- Execute Scheduled Tasks ---
        if tasks_to_run:
            unique_reasons_str = ', '.join(sorted(set(trigger_reasons))) # Log unique reasons
            self.logger.info(f"Running {len(tasks_to_run)} periodic tasks (Triggers: {unique_reasons_str})...", emoji_key="brain")

            # Optional: Prioritize tasks (e.g., run maintenance first)
            tasks_to_run.sort(key=lambda x: 0 if x[0] == TOOL_DELETE_EXPIRED_MEMORIES else 1 if x[0] == TOOL_COMPUTE_STATS else 2)

            # Execute tasks sequentially in this loop cycle
            for tool_name, args in tasks_to_run:
                # Check shutdown flag before each task execution
                if self._shutdown_event.is_set():
                    self.logger.info("Shutdown detected during periodic tasks, aborting remaining.")
                    break
                try:
                    # Handle the special internal promotion check task
                    if tool_name == "CHECK_PROMOTIONS":
                        await self._trigger_promotion_checks() # Call the helper method
                        continue # Move to next scheduled task

                    # Execute standard UMS tool calls
                    self.logger.debug(f"Executing periodic task: {tool_name} with args: {args}")
                    result_content = await self._execute_tool_call_internal(
                        tool_name, args, record_action=False # Don't record periodic tasks as agent actions
                    )

                    # --- Meta-Cognition Feedback Loop ---
                    # Check if the task was reflection or consolidation and if it succeeded
                    if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and result_content.get('success'):
                        feedback = ""
                        # Extract the relevant content from the result dictionary
                        if tool_name == TOOL_REFLECTION:
                            feedback = result_content.get("content", "")
                        elif tool_name == TOOL_CONSOLIDATION:
                             feedback = result_content.get("consolidated_content", "")

                        # Handle cases where result might be nested under 'data'
                        if not feedback and isinstance(result_content.get("data"), dict):
                            if tool_name == TOOL_REFLECTION:
                                feedback = result_content["data"].get("content", "")
                            elif tool_name == TOOL_CONSOLIDATION:
                                feedback = result_content["data"].get("consolidated_content", "")

                        # If feedback content exists, store a summary for the next main loop iteration
                        if feedback:
                            # Create a concise summary (e.g., first line)
                            feedback_summary = str(feedback).split('\n', 1)[0][:150]
                            self.state.last_meta_feedback = f"Feedback from {tool_name.split(':')[-1]}: {feedback_summary}..."
                            self.logger.info(f"Received meta-feedback: {self.state.last_meta_feedback}")
                            # Force replan after receiving significant feedback
                            self.state.needs_replan = True
                        else:
                            # Log if the task succeeded but returned no content for feedback
                            self.logger.debug(f"Periodic task {tool_name} succeeded but provided no feedback content.")

                except Exception as e:
                    # Log errors from periodic tasks but allow the agent loop to continue
                    self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)

                # Optional small delay between periodic tasks within a cycle
                await asyncio.sleep(0.1)


    async def _trigger_promotion_checks(self):
        """
        Queries for recently accessed memories eligible for promotion
        (Episodic -> Semantic, Semantic -> Procedural) and schedules
        background checks for each candidate using `_check_and_trigger_promotion`.
        """
        # Ensure a workflow is active
        if not self.state.workflow_id:
             self.logger.debug("Skipping promotion check: No active workflow.")
             return

        self.logger.debug("Running periodic promotion check for recent memories...")
        query_tool_name = TOOL_QUERY_MEMORIES # Tool for searching memories
        # Check tool availability
        if not self._find_tool_server(query_tool_name):
            self.logger.warning(f"Skipping promotion check: Tool {query_tool_name} unavailable.")
            return

        candidate_memory_ids = set() # Use set to store unique IDs
        try:
            # 1. Find recent Episodic memories (potential for Semantic promotion)
            episodic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.EPISODIC.value,
                "sort_by": "last_accessed", # Prioritize recently used
                "sort_order": "DESC",
                "limit": 5, # Check top N recent/relevant
                "include_content": False # Don't need content for this check
            }
            episodic_results = await self._execute_tool_call_internal(query_tool_name, episodic_args, record_action=False)
            if episodic_results.get("success"):
                mems = episodic_results.get("memories", [])
                if isinstance(mems, list):
                    # Add valid memory IDs to the candidate set
                    candidate_memory_ids.update(m.get('memory_id') for m in mems if isinstance(m, dict) and m.get('memory_id'))

            # 2. Find recent Semantic memories of PROCEDURE or SKILL type (potential for Procedural promotion)
            semantic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.SEMANTIC.value,
                # No type filter here yet, filter after retrieval
                "sort_by": "last_accessed",
                "sort_order": "DESC",
                "limit": 5,
                "include_content": False
            }
            semantic_results = await self._execute_tool_call_internal(query_tool_name, semantic_args, record_action=False)
            if semantic_results.get("success"):
                mems = semantic_results.get("memories", [])
                if isinstance(mems, list):
                     # Filter for specific types eligible for promotion to Procedural
                     candidate_memory_ids.update(
                          m.get('memory_id') for m in mems
                          if isinstance(m, dict) and m.get('memory_id') and
                          m.get('memory_type') in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value] # Check type
                     )

            # 3. Schedule background checks for each candidate
            if candidate_memory_ids:
                self.logger.debug(f"Checking {len(candidate_memory_ids)} memories for potential promotion: {[_fmt_id(item_id) for item_id in candidate_memory_ids]}")
                # Create background tasks to check each memory individually
                promo_tasks = []
                for mem_id in candidate_memory_ids:
                     # Use _start_background_task to correctly snapshot state and manage task
                     task = self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id)
                     promo_tasks.append(task)
                # Option: Wait for these checks if promotion status is needed immediately,
                # otherwise let them run truly in the background. For now, fire-and-forget.
                # await asyncio.gather(*promo_tasks, return_exceptions=True) # Uncomment to wait
            else:
                # Log if no eligible candidates were found
                self.logger.debug("No recently accessed, eligible memories found for promotion check.")
        except Exception as e:
            # Log errors occurring during the query phase
            self.logger.error(f"Error during periodic promotion check query: {e}", exc_info=False)

    # ================================================================= context gather --
    async def _gather_context(self) -> Dict[str, Any]:
        """
        Gathers comprehensive context for the agent LLM.

        Includes:
        - Core context (recent actions, important memories, key thoughts) via TOOL_GET_CONTEXT.
        - Current working memory via TOOL_GET_WORKING_MEMORY **(Prioritized)**.
        - Proactively searched memories relevant to the current plan step **(Limited Fetch)**.
        - Relevant procedural memories **(Limited Fetch)**.
        - Summary of links around a focal or important memory **(Limited Fetch)**.
        - **Freshness indicators** for components.
        - Handles potential errors during retrieval.
        - Initiates context compression if token estimates exceed thresholds.
        """
        # <<< Start Integration Block: Enhance _gather_context (Phase 1, Step 1 Completed) >>>
        self.logger.info("Gathering comprehensive context...", emoji_key="satellite")
        start_time = time.time()
        retrieval_timestamp = datetime.now(timezone.utc).isoformat() # Timestamp for freshness

        # Initialize context dictionary with placeholders and essential state
        base_context = {
            # Core agent state info
            "current_loop": self.state.current_loop,
            "workflow_id": self.state.workflow_id, # Include current WF ID
            "context_id": self.state.context_id, # Include current Context ID
            "current_plan": [p.model_dump(exclude_none=True) for p in self.state.current_plan], # Current plan state
            "last_action_summary": self.state.last_action_summary,
            "consecutive_error_count": self.state.consecutive_error_count,
            "last_error_details": copy.deepcopy(self.state.last_error_details), # Deep copy error details
            "needs_replan": self.state.needs_replan, # Include replan flag in context
            "workflow_stack": self.state.workflow_stack,
            "meta_feedback": self.state.last_meta_feedback, # Include feedback from last meta task
            "current_thought_chain_id": self.state.current_thought_chain_id,
            # Placeholders for dynamically fetched context components
            "core_context": None, # From TOOL_GET_CONTEXT
            "current_working_memory": None, # From TOOL_GET_WORKING_MEMORY (will be dict)
            "proactive_memories": None, # Dict: {"retrieved_at": ..., "memories": [...]}
            "relevant_procedures": None, # Dict: {"retrieved_at": ..., "memories": [...]}
            "contextual_links": None, # Dict: {"retrieved_at": ..., "summary": {...}}
            "compression_summary": None, # If compression is applied
            "status": "Gathering...", # Initial status
            "errors": [] # List to collect errors during gathering
        }
        # Clear feedback after adding it to context
        self.state.last_meta_feedback = None

        # Determine the current workflow and context IDs from state
        current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_context_id = self.state.context_id

        # If no workflow is active, return immediately
        if not current_workflow_id:
            base_context["status"] = "No Active Workflow"
            base_context["message"] = "Agent must create or load a workflow."
            self.logger.warning(base_context["message"])
            return base_context

        # --- Fetch Core Context (e.g., Recent Actions, Important Memories, Key Thoughts) ---
        if self._find_tool_server(TOOL_GET_CONTEXT):
            try:
                # Fetch core context with predefined limits (FETCH_LIMIT constants)
                core_ctx_result = await self._execute_tool_call_internal(
                    TOOL_GET_CONTEXT,
                    {
                        "workflow_id": current_workflow_id,
                        # Use FETCH limits here, truncation to SHOW limits happens later if needed
                        "recent_actions_limit": CONTEXT_RECENT_ACTIONS_FETCH_LIMIT,
                        "important_memories_limit": CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT,
                        "key_thoughts_limit": CONTEXT_KEY_THOUGHTS_FETCH_LIMIT,
                    },
                    record_action=False # Internal context fetch
                )
                if core_ctx_result.get("success"):
                    # Store the successful result and add freshness timestamp
                    base_context["core_context"] = core_ctx_result
                    base_context["core_context"]["retrieved_at"] = retrieval_timestamp # Add freshness
                    # Clean up redundant success/timing info from the nested result
                    base_context["core_context"].pop("success", None)
                    base_context["core_context"].pop("processing_time", None)
                    self.logger.debug(f"Successfully retrieved core context via {TOOL_GET_CONTEXT}.")
                else:
                    err_msg = f"Core context retrieval ({TOOL_GET_CONTEXT}) failed: {core_ctx_result.get('error')}"
                    base_context["errors"].append(err_msg)
                    self.logger.warning(err_msg)
            except Exception as e:
                err_msg = f"Core context retrieval exception: {e}"
                self.logger.error(err_msg, exc_info=False)
                base_context["errors"].append(err_msg)
        else:
            self.logger.warning(f"Skipping core context: Tool {TOOL_GET_CONTEXT} unavailable.")

        # --- Get Current Working Memory (Provides active memories and focal point) ---
        focal_mem_id_from_wm: Optional[str] = None # Store focal ID if found
        working_mem_list_from_wm: List[Dict] = [] # Store memory list if found
        if current_context_id and self._find_tool_server(TOOL_GET_WORKING_MEMORY):
            try:
                wm_result = await self._execute_tool_call_internal(
                    TOOL_GET_WORKING_MEMORY,
                    {
                        "context_id": current_context_id,
                        "include_content": False, # Keep context lighter
                        "include_links": False # Links fetched separately if needed
                    },
                    record_action=False
                )
                if wm_result.get("success"):
                    # Store the entire result dict and add freshness
                    base_context["current_working_memory"] = wm_result
                    base_context["current_working_memory"]["retrieved_at"] = retrieval_timestamp # Add freshness
                    # Clean up redundant fields
                    base_context["current_working_memory"].pop("success", None)
                    base_context["current_working_memory"].pop("processing_time", None)
                    # Extract focal ID and memory list for later use
                    focal_mem_id_from_wm = wm_result.get("focal_memory_id")
                    working_mem_list_from_wm = wm_result.get("working_memories", [])
                    # Log count of retrieved working memories
                    wm_count = len(working_mem_list_from_wm)
                    self.logger.info(f"Retrieved {wm_count} items from working memory (Context: {_fmt_id(current_context_id)}). Focal: {_fmt_id(focal_mem_id_from_wm)}")
                else:
                    err_msg = f"Working memory retrieval failed: {wm_result.get('error')}"
                    base_context["errors"].append(err_msg)
                    self.logger.warning(err_msg)
            except Exception as e:
                err_msg = f"Working memory retrieval exception: {e}"
                self.logger.error(err_msg, exc_info=False)
                base_context["errors"].append(err_msg)
        else:
            self.logger.warning(f"Skipping working memory retrieval: Context ID missing or tool {TOOL_GET_WORKING_MEMORY} unavailable.")


        # --- Goal-Directed Proactive Memory Retrieval (Using Hybrid Search) ---
        # Find memories relevant to the current plan step
        active_plan_step_desc = self.state.current_plan[0].description if self.state.current_plan else "Achieve main goal"
        # Formulate a query based on the current step
        proactive_query = f"Information relevant to planning or executing: {active_plan_step_desc}"
        # Prefer hybrid search, fallback to semantic
        search_tool_proactive = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
        if self._find_tool_server(search_tool_proactive):
            search_args = {
                "workflow_id": current_workflow_id,
                "query": proactive_query,
                "limit": CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT, # Use FETCH limit
                "include_content": False # Keep context light
            }
            # Adjust weights for hybrid search
            if search_tool_proactive == TOOL_HYBRID_SEARCH:
                search_args.update({"semantic_weight": 0.7, "keyword_weight": 0.3}) # Prioritize semantics a bit
            try:
                result_content = await self._execute_tool_call_internal(
                    search_tool_proactive, search_args, record_action=False
                )
                if result_content.get("success"):
                    proactive_mems = result_content.get("memories", [])
                    # Determine score key based on tool used
                    score_key = "hybrid_score" if search_tool_proactive == TOOL_HYBRID_SEARCH else "similarity"
                    # Format results for context, include freshness
                    base_context["proactive_memories"] = {
                        "retrieved_at": retrieval_timestamp,
                        "memories": [
                            {
                                "memory_id": m.get("memory_id"),
                                "description": m.get("description"),
                                "score": round(m.get(score_key, 0), 3), # Include score
                                "type": m.get("memory_type") # Include type
                             }
                            for m in proactive_mems # Iterate through results
                        ]
                    }
                    if base_context["proactive_memories"]["memories"]:
                        self.logger.info(f"Retrieved {len(base_context['proactive_memories']['memories'])} proactive memories using {search_tool_proactive.split(':')[-1]}.")
                else:
                    err_msg = f"Proactive memory search ({search_tool_proactive}) failed: {result_content.get('error')}"
                    base_context["errors"].append(err_msg)
                    self.logger.warning(err_msg)
            except Exception as e:
                err_msg = f"Proactive memory search exception: {e}"
                self.logger.warning(err_msg, exc_info=False)
                base_context["errors"].append(err_msg)
        else:
            self.logger.warning("Skipping proactive memory search: No suitable search tool available.")

        # --- Fetch Relevant Procedural Memories (Using Hybrid Search) ---
        # Find procedural memories related to the current step
        search_tool_proc = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
        if self._find_tool_server(search_tool_proc):
            # Formulate a query focused on finding procedures/how-to steps
            proc_query = f"How to accomplish step-by-step: {active_plan_step_desc}"
            search_args = {
                "workflow_id": current_workflow_id,
                "query": proc_query,
                "limit": CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT, # Use FETCH limit
                "memory_level": MemoryLevel.PROCEDURAL.value, # Explicitly filter for procedural level
                "include_content": False # Keep context light
            }
            if search_tool_proc == TOOL_HYBRID_SEARCH:
                search_args.update({"semantic_weight": 0.6, "keyword_weight": 0.4}) # Balanced weights maybe?
            try:
                proc_result = await self._execute_tool_call_internal(
                    search_tool_proc, search_args, record_action=False
                )
                if proc_result.get("success"):
                    proc_mems = proc_result.get("memories", [])
                    score_key = "hybrid_score" if search_tool_proc == TOOL_HYBRID_SEARCH else "similarity"
                    # Format results for context, include freshness
                    base_context["relevant_procedures"] = {
                         "retrieved_at": retrieval_timestamp,
                         "procedures": [
                            {
                                "memory_id": m.get("memory_id"),
                                "description": m.get("description"),
                                "score": round(m.get(score_key, 0), 3)
                            }
                            for m in proc_mems
                         ]
                    }
                    if base_context["relevant_procedures"]["procedures"]:
                        self.logger.info(f"Retrieved {len(base_context['relevant_procedures']['procedures'])} relevant procedures using {search_tool_proc.split(':')[-1]}.")
                else:
                    err_msg = f"Procedure search failed: {proc_result.get('error')}"
                    base_context["errors"].append(err_msg)
                    self.logger.warning(err_msg)
            except Exception as e:
                err_msg = f"Procedure search exception: {e}"
                self.logger.warning(err_msg, exc_info=False)
                base_context["errors"].append(err_msg)
        else:
            self.logger.warning("Skipping procedure search: No suitable search tool available.")

        # --- Contextual Link Traversal (Prioritizing Focal Memory) ---
        # Find memories linked to the current focus or most relevant items
        get_linked_memories_tool = TOOL_GET_LINKED_MEMORIES
        if self._find_tool_server(get_linked_memories_tool):
            mem_id_to_traverse = None
            # 1. PRIORITIZE focal memory from working memory result
            if focal_mem_id_from_wm:
                mem_id_to_traverse = focal_mem_id_from_wm
                self.logger.debug(f"Link traversal starting from focal memory: {_fmt_id(mem_id_to_traverse)}")

            # 2. If no focal, try the first memory *in* the working memory list
            if not mem_id_to_traverse and working_mem_list_from_wm:
                first_wm_item = working_mem_list_from_wm[0]
                if isinstance(first_wm_item, dict):
                    mem_id_to_traverse = first_wm_item.get("memory_id")
                    if mem_id_to_traverse:
                        self.logger.debug(f"Link traversal starting from first working memory item: {_fmt_id(mem_id_to_traverse)}")

            # 3. If still no ID, try the first important memory from the core context
            if not mem_id_to_traverse:
                core_ctx_data = base_context.get("core_context", {})
                if isinstance(core_ctx_data, dict):
                    important_mem_list = core_ctx_data.get("important_memories", [])
                    if isinstance(important_mem_list, list) and important_mem_list:
                        first_mem = important_mem_list[0]
                        if isinstance(first_mem, dict):
                            mem_id_to_traverse = first_mem.get("memory_id")
                            if mem_id_to_traverse:
                                self.logger.debug(f"Link traversal starting from first important memory: {_fmt_id(mem_id_to_traverse)}")

            # If we found a relevant memory ID to start traversal from
            if mem_id_to_traverse:
                self.logger.debug(f"Attempting link traversal from relevant memory: {_fmt_id(mem_id_to_traverse)}...")
                try:
                    links_result_content = await self._execute_tool_call_internal(
                        get_linked_memories_tool,
                        {
                            "memory_id": mem_id_to_traverse,
                            "direction": "both", # Get incoming and outgoing links
                            "limit": CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT, # Use FETCH limit
                            "include_memory_details": False # Just need link info for context
                         },
                        record_action=False
                    )
                    if links_result_content.get("success"):
                        links_data = links_result_content.get("links", {})
                        outgoing_links = links_data.get("outgoing", [])
                        incoming_links = links_data.get("incoming", [])
                        # Create a concise summary of the links found
                        link_summary = {
                            "source_memory_id": mem_id_to_traverse,
                            "outgoing_count": len(outgoing_links),
                            "incoming_count": len(incoming_links),
                            "top_links_summary": [] # List of concise link descriptions
                        }
                        # Add summaries for top outgoing links (up to SHOW limit)
                        for link in outgoing_links[:CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT]:
                            link_summary["top_links_summary"].append(
                                f"OUT: {link.get('link_type', 'related')} -> {_fmt_id(link.get('target_memory_id'))}"
                            )
                        # Add summaries for top incoming links (up to SHOW limit)
                        for link in incoming_links[:CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT]:
                            link_summary["top_links_summary"].append(
                                f"IN: {_fmt_id(link.get('source_memory_id'))} -> {link.get('link_type', 'related')}"
                            )
                        # Add freshness timestamp to the link summary
                        base_context["contextual_links"] = {"retrieved_at": retrieval_timestamp, "summary": link_summary}
                        self.logger.info(f"Retrieved link summary for memory {_fmt_id(mem_id_to_traverse)} ({len(outgoing_links)} out, {len(incoming_links)} in).")
                    else:
                        err_msg = f"Link retrieval ({get_linked_memories_tool}) failed: {links_result_content.get('error', 'Unknown')}"
                        base_context["errors"].append(err_msg)
                        self.logger.warning(err_msg)
                except Exception as e:
                    err_msg = f"Link retrieval exception: {e}"
                    self.logger.warning(err_msg, exc_info=False)
                    base_context["errors"].append(err_msg)
            else:
                # Log if no suitable starting point for link traversal was found
                self.logger.debug("No relevant memory found (focal/working/important) to perform link traversal from.")
        else:
            # Log if the link retrieval tool is unavailable
            self.logger.debug(f"Skipping link traversal: Tool '{get_linked_memories_tool}' unavailable.")

        # --- Context Compression Check ---
        # This should happen *after* all components are gathered
        try:
            # Estimate tokens of the fully assembled context
            estimated_tokens = await self._estimate_tokens_anthropic(base_context)
            # If estimate exceeds threshold, attempt compression
            if estimated_tokens > CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD:
                self.logger.warning(f"Context ({estimated_tokens} tokens) exceeds threshold {CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD}. Attempting summary compression.")
                # Check if summarization tool is available
                if self._find_tool_server(TOOL_SUMMARIZE_TEXT):
                    # Strategy: Summarize the potentially longest/most verbose part first
                    # Example: Summarize 'core_context' -> 'recent_actions' if it exists and is large
                    core_ctx = base_context.get("core_context")
                    actions_to_summarize = None
                    # Check structure before accessing potentially missing keys
                    if isinstance(core_ctx, dict):
                        actions_to_summarize = core_ctx.get("recent_actions")

                    # Only summarize if the actions list exists and is substantial (e.g., > 1000 chars JSON)
                    if actions_to_summarize and isinstance(actions_to_summarize, list) and len(json.dumps(actions_to_summarize, default=str))                    > 1000:
                         actions_text = json.dumps(actions_to_summarize, default=str) # Serialize for summarizer
                         summary_result = await self._execute_tool_call_internal(
                              TOOL_SUMMARIZE_TEXT,
                              {
                                  "text_to_summarize": actions_text,
                                  "target_tokens": CONTEXT_COMPRESSION_TARGET_TOKENS, # Use target constant
                                  # Use specialized summarizer prompt for actions
                                  "prompt_template": "summarize_context_block", # Assuming this maps to the right prompt in UMS
                                  "context_type": "actions", # Tell summarizer what it's summarizing
                                  "workflow_id": current_workflow_id, # Provide workflow context
                                  "record_summary": False # Don't store this ephemeral summary
                              },
                              record_action=False # Internal action
                         )
                         if summary_result.get("success"):
                              # Add a note about the compression to the context
                              base_context["compression_summary"] = f"Summary of recent actions: {summary_result.get('summary', 'Summary failed.')[:150]}..."
                              # Replace the original actions list in core_context with a reference to the summary
                              if isinstance(core_ctx, dict):
                                   core_ctx["recent_actions"] = f"[Summarized: See compression_summary field]"
                              self.logger.info(f"Compressed recent actions. New context size estimate: {await self._estimate_tokens_anthropic(base_context)} tokens")
                         else:
                              err_msg = f"Context compression tool ({TOOL_SUMMARIZE_TEXT}) failed: {summary_result.get('error')}"
                              base_context["errors"].append(err_msg); self.logger.warning(err_msg)
                    else:
                         # Log if no suitable large component found for compression
                         self.logger.info("No large action list found to compress, context remains large.")
                else:
                    # Log if summarization tool is unavailable
                    self.logger.warning(f"Cannot compress context: Tool '{TOOL_SUMMARIZE_TEXT}' unavailable.")
        except Exception as e:
            # Catch errors during the estimation/compression process
            err_msg = f"Error during context compression check: {e}"
            self.logger.error(err_msg, exc_info=False)
            base_context["errors"].append(err_msg)
        # <<< End Integration Block: Context Gathering >>>

        # Final status update
        base_context["status"] = "Ready" if not base_context["errors"] else "Ready with Errors"
        self.logger.info(f"Context gathering complete. Errors: {len(base_context['errors'])}. Time: {(time.time() - start_time):.3f}s")
        return base_context


    # ------------------------------------------------------------ main loop --
    async def run(self, goal: str, max_loops: int = 100) -> None:
        """
        The main execution loop for the agent.

        Args:
            goal: The high-level goal for the agent to achieve.
            max_loops: Maximum number of iterations before stopping automatically.
        """
        self.logger.info(f"Starting agent loop. Goal: '{goal}'. Max loops: {max_loops}")

        # Initialize or ensure workflow exists
        if not self.state.workflow_id:
            self.logger.info("No active workflow found. Creating initial workflow.")
            wf_create_args = {
                "title": f"Agent Task: {goal[:50]}...",
                "goal": goal,
                "description": f"Agent workflow initiated at {datetime.now(timezone.utc).isoformat()} to achieve: {goal}",
                "tags": ["agent_run", AGENT_NAME.lower()]
            }
            wf_create_result = await self._execute_tool_call_internal(
                TOOL_CREATE_WORKFLOW, wf_create_args, record_action=False
            )
            # Note: _handle_workflow_side_effects should set self.state.workflow_id, context_id, stack, chain_id etc.
            if not wf_create_result.get("success") or not self.state.workflow_id:
                self.logger.critical(f"Failed to create initial workflow: {wf_create_result.get('error')}. Aborting.")
                return
        elif not self.state.current_thought_chain_id:
            # If workflow loaded but no chain ID, try setting default
            await self._set_default_thought_chain_id()


        # --- Main Think-Act Loop ---
        while self.state.current_loop < max_loops and not self.state.goal_achieved_flag and not self._shutdown_event.is_set():
            self.state.current_loop += 1
            self.logger.info(f"--- Starting Loop {self.state.current_loop}/{max_loops} (WF: {_fmt_id(self.state.workflow_id)}) ---")

            # --- 1. Periodic Cognitive Tasks ---
            try:
                 await self._run_periodic_tasks()
            except Exception as e:
                 self.logger.error(f"Error during periodic tasks: {e}", exc_info=True) # Log but continue loop

            # Check shutdown flag again after periodic tasks
            if self._shutdown_event.is_set(): break

            # --- 2. Gather Context ---
            context = await self._gather_context()

            # Check if no active workflow after context gathering (e.g., root workflow finished)
            if not self.state.workflow_id:
                 self.logger.info("No active workflow. Agent loop concluding.")
                 break # Exit loop if workflow finished

            # --- 3. Pre-computation/Pre-analysis (If needed before LLM call) ---
            # (Placeholder for future logic)

            # --- 4. Call LLM for Decision ---
            # Clear error details before LLM call (unless needs_replan is set for error recovery)
            if not self.state.needs_replan:
                 self.state.last_error_details = None
            # Call the LLM
            decision = await self._call_agent_llm(goal, context)

            # --- 5. Execute Decision ---
            tool_result_content: Optional[Dict[str, Any]] = None # Store result for heuristic update
            llm_proposed_plan = decision.get("updated_plan_steps") # Check if LLM proposed plan via text

            # Handle decision based on type
            if decision.get("decision") == "call_tool":
                tool_name = decision.get("tool_name")
                arguments = decision.get("arguments", {})

                # --- Plan Validation (Pre-Execution Check) ---
                # Ensure the intended step (likely first in plan) is valid
                if not self.state.current_plan:
                    self.logger.error("Attempting tool call but plan is empty! Forcing replan.")
                    self.state.needs_replan = True
                    self.state.last_error_details = {"error": "Plan became empty before tool call.", "type": "PlanValidationError"}
                elif not self.state.current_plan[0].description:
                    self.logger.error(f"Attempting tool call for invalid plan step (missing description)! Step ID: {self.state.current_plan[0].id}. Forcing replan.")
                    self.state.needs_replan = True
                    self.state.last_error_details = {"error": "Current plan step is invalid (missing description).", "type": "PlanValidationError", "step_id": self.state.current_plan[0].id}
                # If checks pass, execute the tool call
                elif tool_name:
                    # Extract dependencies *from the current plan step* that this tool call corresponds to
                    current_step_deps = self.state.current_plan[0].depends_on if self.state.current_plan else []
                    # Execute the tool call, passing planned dependencies for checking
                    tool_result_content = await self._execute_tool_call_internal(
                        tool_name, arguments, record_action=True, planned_dependencies=current_step_deps
                    )
                else: # Should not happen if parsing worked correctly
                    self.logger.error("LLM decided to call tool but no tool name was provided.")
                    self.state.last_error_details = {"error": "LLM tool call decision missing tool name.", "type": "LLMOutputError"}
                    tool_result_content = {"success": False, "error": "Missing tool name from LLM."}

            elif decision.get("decision") == "thought_process":
                # Record the thought provided by the LLM
                thought_content = decision.get("content")
                if thought_content:
                    # Default to 'inference' type if LLM doesn't specify
                    thought_type = ThoughtType.INFERENCE.value
                    # Advanced: Could try to infer type from text content if needed
                    tool_result_content = await self._execute_tool_call_internal(
                        TOOL_RECORD_THOUGHT,
                        {"content": thought_content, "thought_type": thought_type},
                        # Allow workflow_id and thought_chain_id injection
                        record_action=False # Recording thoughts isn't a primary world action
                    )
                else: # Handle missing content
                    self.logger.warning("LLM provided 'thought_process' decision but no content.")
                    tool_result_content = {"success": False, "error": "Missing thought content from LLM."}

            elif decision.get("decision") == "complete":
                # Goal achieved signal from LLM
                self.logger.info(f"LLM signaled goal completion: {decision.get('summary')}")
                self.state.goal_achieved_flag = True # Set flag to terminate loop
                # Optionally update workflow status to completed
                if self.state.workflow_id and self._find_tool_server(TOOL_UPDATE_WORKFLOW_STATUS):
                    await self._execute_tool_call_internal(
                        TOOL_UPDATE_WORKFLOW_STATUS,
                        {
                            "workflow_id": self.state.workflow_id,
                            "status": WorkflowStatus.COMPLETED.value,
                            "completion_message": decision.get('summary', 'Goal marked achieved by agent.')
                        },
                        record_action=False # Meta-action
                    )
                # Break the loop after handling completion
                break

            elif decision.get("decision") == "error":
                 # LLM or internal error during decision making
                 self.logger.error(f"LLM decision error: {decision.get('message')}")
                 self.state.last_action_summary = f"LLM Decision Error: {decision.get('message', 'Unknown')[:100]}"
                 # Store error details if not already set by tool execution
                 if not self.state.last_error_details:
                     self.state.last_error_details = {"error": decision.get('message'), "type": "LLMError"}
                 self.state.needs_replan = True # Force replan after LLM error
                 # action_successful remains False

            # --- 6. Apply Plan Updates ---
            # Priority: Plan proposed by LLM via text > Heuristic update
            if llm_proposed_plan:
                 try:
                     # Validate the structure (already done by _call_agent_llm parsing)
                     # --- Plan Cycle Detection ---
                     if self._detect_plan_cycle(llm_proposed_plan):
                         err_msg = "LLM-proposed plan contains a dependency cycle. Applying heuristic update instead."
                         self.logger.error(err_msg)
                         self.state.last_error_details = {"error": err_msg, "type": "PlanValidationError", "proposed_plan": [p.model_dump() for p in llm_proposed_plan]}
                         self.state.needs_replan = True # Force replan again
                         # Fallback to heuristic update if LLM plan is invalid
                         await self._apply_heuristic_plan_update(decision, tool_result_content)
                     else:
                         # Apply the LLM's validated plan
                         self.state.current_plan = llm_proposed_plan
                         self.state.needs_replan = False # LLM provided the plan, assume it's intended
                         self.logger.info(f"Applied LLM-proposed plan update ({len(llm_proposed_plan)} steps).")
                         # Clear errors after successful LLM plan update
                         self.state.last_error_details = None
                         self.state.consecutive_error_count = 0
                 except Exception as plan_apply_err:
                      # Catch errors applying the plan (should be rare if validation passed)
                      self.logger.error(f"Error applying LLM proposed plan: {plan_apply_err}. Falling back to heuristic.", exc_info=True)
                      self.state.last_error_details = {"error": f"Failed to apply LLM plan: {plan_apply_err}", "type": "PlanUpdateError"}
                      self.state.needs_replan = True
                      await self._apply_heuristic_plan_update(decision, tool_result_content)

            elif decision.get("tool_name") != AGENT_TOOL_UPDATE_PLAN: # Heuristic only if LLM didn't explicitly update plan
                # Apply heuristic updates if LLM didn't use the plan update tool or provide a valid text plan
                await self._apply_heuristic_plan_update(decision, tool_result_content)
            # else: If AGENT_TOOL_UPDATE_PLAN was called, success/failure already handled by _execute_tool_call_internal

            # --- 7. Check Error Limit & Save State ---
            if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                self.logger.critical(f"Max consecutive errors ({MAX_CONSECUTIVE_ERRORS}) reached. Aborting loop.")
                # Update workflow status to failed if possible
                if self.state.workflow_id and self._find_tool_server(TOOL_UPDATE_WORKFLOW_STATUS):
                     await self._execute_tool_call_internal(
                         TOOL_UPDATE_WORKFLOW_STATUS,
                         {
                             "workflow_id": self.state.workflow_id,
                             "status": WorkflowStatus.FAILED.value,
                             "completion_message": f"Aborted after {MAX_CONSECUTIVE_ERRORS} consecutive errors."
                         },
                         record_action=False
                     )
                break # Exit loop

            # Save state at the end of each loop iteration
            await self._save_agent_state()

            # Optional: Small delay between loops
            # await asyncio.sleep(0.5)

        # --- Loop End ---
        if self._shutdown_event.is_set():
            self.logger.info("Agent loop terminated due to shutdown signal.")
        elif self.state.current_loop >= max_loops:
            self.logger.warning(f"Agent loop reached max iterations ({max_loops}). Stopping.")
        elif self.state.goal_achieved_flag:
            self.logger.info("Agent loop finished: Goal achieved.", emoji_key="tada")
        else:
            self.logger.warning("Agent loop finished for unexpected reason.")

        # Final state save and cleanup handled by shutdown() or run_agent_process() finally block


# =============================================================================
# Driver helpers & CLI entry‑point
# =============================================================================

async def run_agent_process(
    mcp_server_url: str,
    anthropic_key: str,
    goal: str,
    max_loops: int,
    state_file: str,
    config_file: Optional[str],
) -> None:
    """
    Sets up the MCPClient, Agent Master Loop, signal handling,
    and runs the main agent execution loop.
    """
    # Ensure MCPClient is available
    if not MCP_CLIENT_AVAILABLE:
        print("❌ ERROR: MCPClient dependency not met.")
        sys.exit(1)

    mcp_client_instance = None
    agent_loop_instance = None
    exit_code = 0
    # Use standard print initially, switch to client's safe_print if available
    printer = print

    try:
        printer("Instantiating MCP Client...")
        # Initialize MCP Client (pass URL and optional config path)
        mcp_client_instance = MCPClient(base_url=mcp_server_url, config_path=config_file)
        # Use safe_print for console output if provided by the client
        if hasattr(mcp_client_instance, 'safe_print') and callable(mcp_client_instance.safe_print):
            printer = mcp_client_instance.safe_print
            log.info("Using MCPClient's safe_print for output.")

        # Configure API key if not already set in config
        if not mcp_client_instance.config.api_key:
            if anthropic_key:
                printer("Using provided Anthropic API key.")
                mcp_client_instance.config.api_key = anthropic_key
                # Re-initialize the anthropic client instance within MCPClient if necessary
                mcp_client_instance.anthropic = AsyncAnthropic(api_key=anthropic_key)
            else:
                # Critical error if key is missing
                printer("❌ CRITICAL ERROR: Anthropic API key missing in config and not provided.")
                raise ValueError("Anthropic API key missing.")

        printer("Setting up MCP Client connections...")
        # Perform necessary setup (like connecting to servers/discovery)
        await mcp_client_instance.setup(interactive_mode=False) # Run in non-interactive mode

        printer("Instantiating Agent Master Loop...")
        # Create the agent instance, passing the initialized MCP client and state file path
        agent_loop_instance = AgentMasterLoop(mcp_client_instance=mcp_client_instance, agent_state_file=state_file)

        # --- Signal Handling Setup ---
        # Get the current asyncio event loop
        loop = asyncio.get_running_loop()
        # Create an event to signal shutdown across tasks
        stop_event = asyncio.Event()

        # Define the signal handler function
        def signal_handler_wrapper(signum):
            signal_name = signal.Signals(signum).name
            log.warning(f"Signal {signal_name} received. Initiating graceful shutdown.")
            # Set the event to signal other tasks
            stop_event.set()
            # Trigger the agent's internal shutdown method asynchronously
            if agent_loop_instance:
                 asyncio.create_task(agent_loop_instance.shutdown())
            # Avoid calling loop.stop() directly, let tasks finish gracefully

        # Register the handler for SIGINT (Ctrl+C) and SIGTERM
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                loop.add_signal_handler(sig, signal_handler_wrapper, sig)
                log.debug(f"Registered signal handler for {sig.name}")
            except ValueError: # Might fail if handler already registered
                log.debug(f"Signal handler for {sig.name} may already be registered.")
            except NotImplementedError: # Signal handling might not be supported (e.g., Windows sometimes)
                log.warning(f"Signal handling for {sig.name} not supported on this platform.")


        printer("Initializing agent...")
        if not await agent_loop_instance.initialize():
             printer("❌ Agent initialization failed. Exiting.")
             exit_code = 1
             return # Exit early if initialization fails

        printer(f"Running Agent Loop for goal: \"{goal}\"")
        # Create tasks for the main agent run and for waiting on the stop signal
        run_task = asyncio.create_task(agent_loop_instance.run(goal=goal, max_loops=max_loops))
        stop_task = asyncio.create_task(stop_event.wait())

        # Wait for either the agent run to complete OR the stop signal to be received
        done, pending = await asyncio.wait(
            {run_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED # Return as soon as one task finishes
        )

        # Handle shutdown signal completion
        if stop_task in done:
             printer("\n[yellow]Shutdown signal processed. Waiting for agent task to finalize...[/yellow]")
             # If the agent task is still running, attempt to cancel it
             if run_task in pending:
                 run_task.cancel()
                 try:
                     # Wait for the cancellation to be processed
                     await run_task
                 except asyncio.CancelledError:
                      log.info("Agent run task cancelled gracefully after signal.")
                 except Exception as e:
                      # Log error if cancellation failed unexpectedly
                      log.error(f"Exception during agent run task finalization after signal: {e}", exc_info=True)
             # Set standard exit code for signal interruption (like Ctrl+C)
             exit_code = 130

        # Handle normal agent completion or error completion
        elif run_task in done:
             try:
                 run_task.result() # Check for exceptions raised by the run task
                 log.info("Agent run task completed normally.")
             except Exception as e:
                 # If the run task raised an exception, log it and set error exit code
                 printer(f"\n❌ Agent loop finished with error: {e}")
                 log.error("Agent run task finished with an exception:", exc_info=True)
                 exit_code = 1


    except KeyboardInterrupt:
        # Fallback for Ctrl+C if signal handler doesn't catch it (e.g., during setup)
        printer("\n[yellow]KeyboardInterrupt caught (fallback).[/yellow]")
        exit_code = 130
    except ValueError as ve: # Catch specific config/value errors during setup
        printer(f"\n❌ Configuration Error: {ve}")
        exit_code = 2
    except Exception as main_err:
        # Catch any other critical errors during setup or main execution
        printer(f"\n❌ Critical error during setup or execution: {main_err}")
        log.critical("Top-level execution error", exc_info=True)
        exit_code = 1
    finally:
        # --- Cleanup Sequence ---
        printer("Initiating final shutdown sequence...")
        # Ensure agent shutdown method is called (might be redundant if signaled, but safe)
        if agent_loop_instance and not agent_loop_instance._shutdown_event.is_set():
             printer("Ensuring agent loop shutdown...")
             await agent_loop_instance.shutdown() # Call directly if not already shutting down
        # Ensure MCP client connections are closed
        if mcp_client_instance:
            printer("Closing MCP client connections...")
            try:
                await mcp_client_instance.close()
            except Exception as close_err:
                printer(f"[red]Error closing MCP client:[/red] {close_err}")
        printer("Agent execution finished.")
        # Exit the process only if running as the main script
        if __name__ == "__main__":
            # Short delay to allow logs/output to flush before exiting
            await asyncio.sleep(0.5)
            sys.exit(exit_code)

# Main execution block when script is run directly
if __name__ == "__main__":
    # (Keep existing __main__ block for configuration loading and running)
    # Load configuration from environment variables or defaults
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8013")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    AGENT_GOAL = os.environ.get(
        "AGENT_GOAL", # Default goal for testing/example
        "Create workflow 'Tier 3 Test': Research Quantum Computing impact on Cryptography.",
    )
    MAX_ITER = int(os.environ.get("MAX_ITERATIONS", "30")) # Default max loops
    STATE_FILE = os.environ.get("AGENT_STATE_FILE", AGENT_STATE_FILE) # Use constant default
    CONFIG_PATH = os.environ.get("MCP_CLIENT_CONFIG") # Optional MCPClient config file path

    # Validate essential configuration
    if not ANTHROPIC_API_KEY:
        print("❌ ERROR: ANTHROPIC_API_KEY missing in environment variables.")
        sys.exit(1)
    if not MCP_CLIENT_AVAILABLE:
        # This check happens earlier, but keep for robustness
        print("❌ ERROR: MCPClient dependency missing.")
        sys.exit(1)

    # Display configuration being used before starting
    print(f"--- {AGENT_NAME} ---")
    print(f"Memory System URL: {MCP_SERVER_URL}")
    print(f"Agent Goal: {AGENT_GOAL}")
    print(f"Max Iterations: {MAX_ITER}")
    print(f"State File: {STATE_FILE}")
    print(f"Client Config: {CONFIG_PATH or 'Default internal config'}")
    print(f"Log Level: {logging.getLevelName(log.level)}")
    print("Anthropic API Key: Found")
    print("-----------------------------------------")


    # Define the main async function to run the agent process
    async def _main() -> None:
        await run_agent_process(
            MCP_SERVER_URL,
            ANTHROPIC_API_KEY,
            AGENT_GOAL,
            MAX_ITER,
            STATE_FILE,
            CONFIG_PATH,
        )

    # Run the main async function using asyncio.run()
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        # Catch Ctrl+C during initial asyncio.run setup if signal handler isn't active yet
        print("\n[yellow]Initial KeyboardInterrupt detected. Exiting.[/yellow]")
        sys.exit(130)
