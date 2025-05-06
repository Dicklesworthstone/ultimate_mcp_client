"""
EideticEngine Agent Master Loop (AML) - v4.1 P1 - GOAL STACK + MOMENTUM
======================================================================

This module implements the core orchestration logic for the EideticEngine
AI agent. It manages the primary think-act cycle, interacts with the
Unified Memory System (UMS) via MCPClient, leverages an LLM (Anthropic Claude)
for decision-making and planning, and incorporates several cognitive functions
inspired by human memory and reasoning.

** V4.1 P1 implements Phase 1 improvements: refined context, adaptive
thresholds, plan validation/repair, structured error handling, robust
background task management, AND adds explicit Goal Stack management and
a "Mental Momentum" bias. **

Key Functionalities:
--------------------
*   **Workflow & Context Management:**
    - Creates, manages, and tracks progress within structured workflows (via UMS).
    - Supports sub-workflow execution via a workflow stack (agent-managed).
    - **Manages an explicit Goal Stack (agent's view of UMS-managed goals).**
    - Gathers rich, multi-faceted context for the LLM decision-making process, including:
        *   **Current Goal Stack information (agent assembled, from UMS data).**
        *   UMS-provided context package (working memory, core context, proactive/procedural memories, links).
        *   **Freshness indicators** for context components.
    - Implements structure-aware context truncation and optional LLM-based compression.

*   **Planning & Execution:**
    - Maintains an explicit, modifiable plan (agent-managed).
    - Allows the LLM to propose plan updates via a dedicated tool or text parsing.
    - Includes a heuristic fallback mechanism to update plan steps.
    - **Validates plan steps and detects dependency cycles.**
    - Checks action prerequisites (dependencies) before execution (via UMS).
    - Executes tools via the MCPClient, handling server lookup and argument injection.
    - Records detailed action history in UMS.

*   **LLM Interaction & Reasoning:**
    - Constructs detailed prompts for the LLM.
    - **Prompts explicitly guide analysis of working memory, goal stack, and provide error recovery strategies.**
    - Parses LLM responses to identify tool calls, textual reasoning, or goal completion signals.
    - Manages dedicated thought chains in UMS for recording reasoning.

*   **Cognitive & Meta-Cognitive Processes:**
    - **Memory Interaction:** Uses UMS tools for all memory operations.
    - **Working Memory Management:** Uses UMS tools for WM retrieval, optimization, and focus.
    - **Goal Management:** Uses UMS tools to create goals and mark their status. Agent maintains local view of stack.
    - **Background Cognitive Tasks:** Initiates asynchronous tasks for UMS auto-linking and memory promotion.
    - **Periodic Meta-cognition:** Runs scheduled tasks (reflection, consolidation via UMS tools).
    - **Adaptive Thresholds:** Dynamically adjusts meta-cognition frequency based on performance and UMS stats.
    - **Maintenance:** Uses UMS tool to delete expired memories.

*   **State & Error Handling:**
    - Persists the complete agent runtime state (workflow, goal stack view, plan, counters, thresholds) to JSON.
    - Implements retry logic with backoff for tool failures.
    - Tracks consecutive errors and halts execution if a limit is reached.
    - Provides detailed, categorized error information to the LLM.
    - Handles graceful shutdown via system signals.

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
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union  # Added TYPE_CHECKING

import aiofiles
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from mcp_client_multi import MCPClient
else:
    MCPClient = "MCPClient"


# --- Workflow & Action Status (from UMS) ---
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


# --- Content Types (from UMS) ---
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
    REFLECTION = "reflection"  # Distinct from ThoughtType.REFLECTION
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


if TYPE_CHECKING:
    from mcp_client_multi import MCPClient
else:
    MCPClient = "MCPClient"  # Placeholder for runtime, type hint only

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
AGENT_STATE_FILE = "agent_loop_state_v4.1_integrated.json"
AGENT_NAME = "EidenticEngine4.1-P1-GoalStackMomentum"

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
CONTEXT_GOAL_DETAILS_FETCH_LIMIT = 3  # Max depth for parent goal fetching
CONTEXT_RECENT_ACTIONS_SHOW_LIMIT = 7
CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT = 5
CONTEXT_KEY_THOUGHTS_SHOW_LIMIT = 5
CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT = 2
CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT = 3
CONTEXT_WORKING_MEMORY_SHOW_LIMIT = 10
CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT = 3
CONTEXT_GOAL_STACK_SHOW_LIMIT = 5  # Max goals in stack summary for prompt
CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD = 15_000
CONTEXT_COMPRESSION_TARGET_TOKENS = 5_000
MAX_CONSECUTIVE_ERRORS = 3

# UMS Tool constants (agent uses these to call specific UMS functions)
TOOL_GET_WORKFLOW_DETAILS = "unified_memory:get_workflow_details"
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context"  # DEPRECATED IN FAVOR OF get_rich_context_package
TOOL_GET_RICH_CONTEXT_PACKAGE = "unified_memory:get_rich_context_package"  # Main context tool
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
TOOL_QUERY_MEMORIES = "unified_memory:query_memories"
TOOL_CREATE_LINK = "unified_memory:create_memory_link"
TOOL_GET_MEMORY_BY_ID = "unified_memory:get_memory_by_id"
TOOL_GET_LINKED_MEMORIES = "unified_memory:get_linked_memories"
TOOL_LIST_WORKFLOWS = "unified_memory:list_workflows"
TOOL_SUMMARIZE_TEXT = "unified_memory:summarize_text"
TOOL_GET_GOAL_DETAILS = "unified_memory:get_goal_details"
TOOL_CREATE_GOAL = "unified_memory:create_goal"
TOOL_UPDATE_GOAL_STATUS = "unified_memory:update_goal_status"
AGENT_TOOL_UPDATE_PLAN = "agent:update_plan"  # Agent's internal tool

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
        context = json.loads(json.dumps(context, default=str))  # Force serialization
        full = json.dumps(context, indent=2, default=str, ensure_ascii=False)

    if len(full) <= max_len:
        return full

    log.debug(f"Context length {len(full)} exceeds max {max_len}. Applying structured truncation.")
    ctx_copy = copy.deepcopy(context)  # Work on a copy
    ctx_copy["_truncation_applied"] = "structure‑aware_agent_side"
    original_length = len(full)

    # Define paths and limits for truncation within the agent's view of context
    # UMS context items are already dictionaries by the time they get here.
    list_paths_to_truncate = [
        ("ums_context_package", "core_context", "recent_actions", CONTEXT_RECENT_ACTIONS_SHOW_LIMIT),
        ("ums_context_package", "core_context", "important_memories", CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT),
        ("ums_context_package", "core_context", "key_thoughts", CONTEXT_KEY_THOUGHTS_SHOW_LIMIT),
        ("ums_context_package", "proactive_memories", "memories", CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT),
        ("ums_context_package", "current_working_memory", "working_memories", CONTEXT_WORKING_MEMORY_SHOW_LIMIT),
        ("ums_context_package", "relevant_procedures", "procedures", CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT),
        ("agent_assembled_goal_context", None, "goal_stack_summary_from_agent_state", CONTEXT_GOAL_STACK_SHOW_LIMIT),
        (None, "current_plan_snapshot", None, 5),  # Truncate plan snapshot if long
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
        ("ums_context_package"),  # Entire UMS package if still too large
    ]

    # Apply list truncations
    for path_parts in list_paths_to_truncate:
        container = ctx_copy
        key_to_truncate = path_parts[-2]  # The key holding the list
        limit_count = path_parts[-1]
        # Navigate to the container
        valid_path = True
        for part_idx in range(len(path_parts) - 2):
            part = path_parts[part_idx]
            if part is None:
                continue  # Skip None parts (e.g. for current_plan_snapshot)
            if part in container and isinstance(container[part], dict):
                container = container[part]
            else:
                valid_path = False
                break
        if not valid_path:
            continue

        # Perform truncation
        if key_to_truncate in container and isinstance(container[key_to_truncate], list) and len(container[key_to_truncate]) > limit_count:
            original_count_val = len(container[key_to_truncate])
            note = {"truncated_note": f"{original_count_val - limit_count} items omitted from '{'/'.join(str(p) for p in path_parts if p)}'"}
            container[key_to_truncate] = container[key_to_truncate][:limit_count]
            if limit_count > 0:
                container[key_to_truncate].append(note)
            log.debug(f"Truncated list at '{'/'.join(str(p) for p in path_parts if p)}' to {limit_count} items.")
            serial_val = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial_val) <= max_len:
                log.info(f"Context truncated (List reduction: {len(serial_val)} bytes).")
                return serial_val

    # Apply key removals
    for key_info_tuple in keys_to_remove_low_priority:
        container = ctx_copy
        key_to_remove_final = key_info_tuple[-1]
        # Navigate to parent container
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
    assigned_tool: Optional[str] = None
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
    goal_stack: List[Dict[str, Any]] = field(default_factory=list)
    current_goal_id: Optional[str] = None
    current_plan: List[PlanStep] = field(default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)])
    current_thought_chain_id: Optional[str] = None
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False
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
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=_default_tool_stats)
    background_tasks: Set[asyncio.Task] = field(default_factory=set, init=False, repr=False)


# =====================================================================
# Agent Master Loop
# =====================================================================
class AgentMasterLoop:
    _INTERNAL_OR_META_TOOLS: Set[str] = {
        TOOL_RECORD_ACTION_START,
        TOOL_RECORD_ACTION_COMPLETION,
        TOOL_GET_CONTEXT,
        TOOL_GET_RICH_CONTEXT_PACKAGE,
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
        TOOL_GET_WORKFLOW_DETAILS,
        TOOL_GET_GOAL_DETAILS,
        TOOL_ADD_ACTION_DEPENDENCY,
        TOOL_CREATE_LINK,
        TOOL_CREATE_GOAL,
        TOOL_UPDATE_GOAL_STATUS,  # UMS Goal tools
        TOOL_LIST_WORKFLOWS,
        TOOL_COMPUTE_STATS,
        TOOL_SUMMARIZE_TEXT,
        TOOL_OPTIMIZE_WM,
        TOOL_AUTO_FOCUS,
        TOOL_PROMOTE_MEM,
        TOOL_REFLECTION,
        TOOL_CONSOLIDATION,
        TOOL_DELETE_EXPIRED_MEMORIES,
        AGENT_TOOL_UPDATE_PLAN,
    }

    def __init__(
        self,
        mcp_client_instance: MCPClient,  # Type hint fixed
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
        self.anthropic_client: AsyncAnthropic = self.mcp_client.anthropic  # type: ignore

        self.consolidation_memory_level = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources = 10
        self.auto_linking_threshold = 0.7
        self.auto_linking_max_links = 3
        self.reflection_type_sequence = ["summary", "progress", "gaps", "strengths", "plan"]
        self.state = AgentState()
        self._shutdown_event = asyncio.Event()
        self._bg_tasks_lock = asyncio.Lock()
        self._bg_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BG_TASKS)
        self.tool_schemas: List[Dict[str, Any]] = []
        self.logger.info(f"AgentMasterLoop initialized. LLM: {self.agent_llm_model}")

    async def shutdown(self) -> None:
        self.logger.info("Shutdown requested.")
        self._shutdown_event.set()
        await self._cleanup_background_tasks()
        await self._save_agent_state()
        self.logger.info("Agent loop shutdown complete.")

    def _construct_agent_prompt(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        system_blocks: List[str] = [
            f"You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System.",
            "",
            f"Overall Goal: {goal}",
        ]

        # Extract current goal details from context (agent_assembled_goal_context)
        current_goal_block = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
        if current_goal_block and isinstance(current_goal_block, dict):
            desc = current_goal_block.get("description", "N/A")
            gid = _fmt_id(current_goal_block.get("goal_id"))
            status = current_goal_block.get("status", "N/A")
            system_blocks.append(f"Current Goal: {desc} (ID: {gid}, Status: {status})")
        else:
            system_blocks.append("Current Goal: None specified (Focus on Overall Goal or plan step)")

        system_blocks.append("")
        system_blocks.append("Available Unified Memory & Agent Tools (Use ONLY these):")

        if not self.tool_schemas:
            system_blocks.append("- CRITICAL WARNING: No tools loaded. Cannot function.")
        else:
            essential_cognitive_tools = {
                TOOL_ADD_ACTION_DEPENDENCY,
                TOOL_RECORD_ARTIFACT,
                TOOL_HYBRID_SEARCH,
                TOOL_STORE_MEMORY,
                TOOL_UPDATE_MEMORY,
                TOOL_CREATE_LINK,
                TOOL_CREATE_THOUGHT_CHAIN,
                TOOL_GET_THOUGHT_CHAIN,
                TOOL_RECORD_THOUGHT,
                TOOL_REFLECTION,
                TOOL_CONSOLIDATION,
                TOOL_PROMOTE_MEM,
                TOOL_OPTIMIZE_WM,
                TOOL_AUTO_FOCUS,
                TOOL_GET_WORKING_MEMORY,
                TOOL_QUERY_MEMORIES,
                TOOL_SEMANTIC_SEARCH,
                AGENT_TOOL_UPDATE_PLAN,
                TOOL_CREATE_GOAL,
                TOOL_UPDATE_GOAL_STATUS,
                TOOL_GET_GOAL_DETAILS,
            }
            for schema in self.tool_schemas:
                sanitized = schema["name"]
                original = self.mcp_client.server_manager.sanitized_to_original.get(sanitized, sanitized)
                desc = schema.get("description", "No description.")
                is_essential = original in essential_cognitive_tools
                prefix = "**" if is_essential else ""
                input_schema_str = json.dumps(schema.get("input_schema", schema.get("parameters", {})))
                system_blocks.append(
                    f"\n- {prefix}Name: `{sanitized}` (Represents: `{original}`){prefix}\n  Desc: {desc}\n  Schema: {input_schema_str}"
                )
        system_blocks.append("")
        system_blocks.extend(
            [
                "Your Process at each step:",
                "1.  Context Analysis: Deeply analyze 'Current Context'. Note workflow status, errors (`last_error_details`), **goal stack (`agent_assembled_goal_context` -> `goal_stack_summary_from_agent_state`) and the `current_goal_details_from_ums`**, UMS package (`ums_context_package` containing core_context, working_memory, proactive/procedural memories, links), `current_plan`, `current_thought_chain_id`, and `meta_feedback`. Pay attention to `retrieved_at` timestamps for freshness.",
                "2.  Error Handling: If `last_error_details` exists, **FIRST** reason about the error `type` and `message`. Propose a recovery strategy. Refer to 'Recovery Strategies'.",
                "3.  Reasoning & Planning:",
                "    a. State step-by-step reasoning towards the Current Goal. Record key thoughts using `record_thought`.",
                "    b. Evaluate `current_plan`. Is it aligned with the Current Goal? Valid? Addresses errors? Dependencies met?",
                "    c. **Goal Management:** If Current Goal is too complex, use `unified_memory:create_goal` (providing `parent_goal_id` as `current_goal_id`). When a goal is met/fails, use `unified_memory:update_goal_status` with the `goal_id` and status.",
                '    d. Action Dependencies: If Step B needs output from Step A (ID \'a123\'), include `"depends_on": ["a123"]` in B\'s plan object. Then use `unified_memory:add_action_dependency` UMS tool.',
                "    e. Artifacts: Plan `unified_memory:record_artifact` for creations. Use `unified_memory:get_artifacts` or `unified_memory:get_artifact_by_id` for existing.",
                "    f. Memory: Use `unified_memory:store_memory` for new facts/insights. Use `unified_memory:update_memory` for corrections.",
                "    g. Thought Chains: Use `unified_memory:create_thought_chain` for distinct sub-problems.",
                "    h. Linking: Use `unified_memory:create_memory_link` for relationships.",
                "    i. Search: Prefer `unified_memory:hybrid_search_memories`. Use `unified_memory:search_semantic_memories` for pure conceptual similarity.",
                "    j. Plan Update Tool: Use `agent:update_plan` ONLY for significant changes, error recovery, or fixing validation issues. Do NOT use for simple step completion.",
                "4.  Action Decision: Choose ONE action based on the *first planned step*:",
                "    *   Call UMS Tool: (e.g., `unified_memory:create_goal`, `unified_memory:store_memory`). Provide args per schema. **Mandatory:** Call `unified_memory:create_workflow` if context shows 'No Active Workflow'.",
                "    *   Record Thought: Use `unified_memory:record_thought`.",
                "    *   Update Plan Tool: Call `agent:update_plan` with the **complete, repaired** plan.",
                "    *   Signal Completion: If Current Goal is MET (use `unified_memory:update_goal_status`) OR Overall Goal is MET (respond ONLY with 'Goal Achieved: ...summary...').",
                "5.  Output Format: Respond ONLY with the valid JSON for the chosen tool call OR 'Goal Achieved: ...summary...' text.",
            ]
        )
        system_blocks.extend(
            [
                "\nKey Considerations:",
                "*   Goal Focus: Always work towards the Current Goal. Use UMS goal tools.",
                "*   Mental Momentum: Prioritize current plan steps if progress is steady.",
                "*   Dependencies & Cycles: Ensure `depends_on` actions are complete. Avoid cycles.",
                "*   UMS Context: Leverage the `ums_context_package` provided by the UMS.",
                "*   Errors: Prioritize error analysis based on `last_error_details.type`.",
                "*   User Guidance: Pay close attention to thoughts of type 'user_guidance' or memories of type 'user_input'. These are direct inputs from the operator and will likely require plan adjustments.*"
            ]
        )
        system_blocks.extend(
            [
                "\nRecovery Strategies based on `last_error_details.type`:",
                "*   `InvalidInputError`: Review tool schema, args, context. Correct args and retry OR choose different tool/step.",
                "*   `DependencyNotMetError`: Use `unified_memory:get_action_details` on dependency IDs. Adjust plan order (`agent:update_plan`) or wait.",
                "*   `ServerUnavailable` / `NetworkError`: Tool's server might be down. Try different tool, wait, or adjust plan.",
                "*   `APILimitError` / `RateLimitError`: External API busy. Plan to wait (record thought) before retry.",
                "*   `ToolExecutionError` / `ToolInternalError`: Tool failed. Analyze message. Try different args, alternative tool, or adjust plan.",
                "*   `PlanUpdateError`: Proposed plan structure was invalid. Re-examine plan and dependencies, try `agent:update_plan` again.",
                "*   `PlanValidationError`: Proposed plan has logical issues (e.g., cycles). Debug dependencies, propose corrected plan using `agent:update_plan`.",
                "*   `CancelledError`: Previous action cancelled. Re-evaluate current step.",
                "*   `GoalManagementError`: Error managing UMS goal stack (e.g., marking non-existent goal). Review `agent_assembled_goal_context` and goal logic.",
                "*   `UnknownError` / `UnexpectedExecutionError`: Analyze error message carefully. Simplify step, use different approach, or record_thought if stuck.",
            ]
        )
        system_prompt = "\n".join(system_blocks)

        context_json = _truncate_context(context)
        user_blocks = [
            "Current Context:",
            "```json",
            context_json,
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
            user_blocks += [
                "**CRITICAL: Address Last Error Details**:",
                "```json",
                json.dumps(self.state.last_error_details, indent=2, default=str),
                "```",
                "",
            ]
        if self.state.last_meta_feedback:
            user_blocks += ["**Meta-Cognitive Feedback**:", self.state.last_meta_feedback, ""]

        # Reiterate current goal from agent's assembled context for emphasis
        current_goal_desc_for_prompt = "Overall Goal"
        if context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums"):
            cg_details = context["agent_assembled_goal_context"]["current_goal_details_from_ums"]
            if isinstance(cg_details, dict) and cg_details.get("description"):
                current_goal_desc_for_prompt = cg_details["description"]

        user_blocks += [
            f"Current Goal Reminder: {current_goal_desc_for_prompt}",
            "",
            "Instruction: Analyze context & errors (use recovery strategies if needed). Reason step-by-step towards the Current Goal. Evaluate and **REPAIR** the plan if `needs_replan` is true or errors indicate plan issues (use `agent:update_plan`). Manage goals using UMS tools (`unified_memory:create_goal` or `unified_memory:update_goal_status`) if needed. Otherwise, decide ONE action based on the *first planned step*: call a UMS tool (output tool_use JSON), record a thought (`unified_memory:record_thought`), or signal completion (use `unified_memory:update_goal_status` for sub-goals or output 'Goal Achieved: ...' for overall goal).",
        ]
        user_prompt = "\n".join(user_blocks)
        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]

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
        snapshot_wf_id = self.state.workflow_id
        snapshot_ctx_id = self.state.context_id

        async def _wrapper():
            log.debug(f"Waiting for semaphore... Task: {asyncio.current_task().get_name()}. Current count: {self._bg_task_semaphore._value}")
            await self._bg_task_semaphore.acquire()
            log.debug(f"Acquired semaphore. Task: {asyncio.current_task().get_name()}. New count: {self._bg_task_semaphore._value}")
            try:
                await asyncio.wait_for(
                    coro_fn(self, *args, workflow_id=snapshot_wf_id, context_id=snapshot_ctx_id, **kwargs), timeout=BACKGROUND_TASK_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Background task {asyncio.current_task().get_name()} timed out.")
            except Exception:
                self.logger.debug(f"Exception in wrapper {asyncio.current_task().get_name()}. Done callback logs.")

        task_name = f"bg_{coro_fn.__name__}_{_fmt_id(snapshot_wf_id)}_{random.randint(100, 999)}"
        task = asyncio.create_task(_wrapper(), name=task_name)
        asyncio.create_task(self._add_bg_task(task))
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
                self.logger.error(f"Task {task_name} error during cleanup: {res}")
            else:
                self.logger.debug(f"Task {task_name} finalized during cleanup.")
        async with self._bg_tasks_lock:
            self.state.background_tasks.clear()
        final_sem_count = self._bg_task_semaphore._value
        if final_sem_count != MAX_CONCURRENT_BG_TASKS:
            self.logger.warning(f"Semaphore count {final_sem_count} after cleanup, expected {MAX_CONCURRENT_BG_TASKS}.")
        self.logger.info("Background tasks cleanup finished.")

    async def _estimate_tokens_anthropic(self, data: Any) -> int:
        if data is None:
            return 0
        try:
            if not self.anthropic_client:
                raise RuntimeError("Anthropic client unavailable")
            text_to_count = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
            token_count = await self.anthropic_client.count_tokens(text_to_count)
            return int(token_count)
        except Exception as e:
            self.logger.warning(f"Token estimation via Anthropic API failed: {e}. Using fallback.")
            text_representation = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
            return len(text_representation) // 4

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
        ),
        retry_backoff: float = 2.0,
        jitter: Tuple[float, float] = (0.1, 0.5),
        **kwargs,
    ):
        attempt = 0
        last_exception = None
        while True:
            try:
                return await coro_fun(*args, **kwargs)
            except retry_exceptions as e:
                attempt += 1
                last_exception = e
                if attempt >= max_retries:
                    self.logger.error(f"{coro_fun.__name__} failed after {max_retries} attempts: {e}")
                    raise
                delay = (retry_backoff ** (attempt - 1)) + random.uniform(*jitter)
                self.logger.warning(
                    f"{coro_fun.__name__} failed ({type(e).__name__}: {str(e)[:100]}...); retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                if self._shutdown_event.is_set():
                    self.logger.warning(f"Shutdown during retry for {coro_fun.__name__}.")
                    raise asyncio.CancelledError(f"Shutdown during retry") from last_exception
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.logger.info(f"{coro_fun.__name__} cancelled during retry.")
                raise

    async def _save_agent_state(self) -> None:
        state_dict = dataclasses.asdict(self.state)
        state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        state_dict.pop("background_tasks", None)
        state_dict["tool_usage_stats"] = {k: dict(v) for k, v in self.state.tool_usage_stats.items()}
        state_dict["current_plan"] = [step.model_dump(exclude_none=True) for step in self.state.current_plan]
        state_dict["goal_stack"] = self.state.goal_stack
        try:
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.agent_state_file.with_suffix(f".tmp_{os.getpid()}")
            async with aiofiles.open(tmp_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(state_dict, indent=2, ensure_ascii=False))
                await f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    self.logger.warning(f"os.fsync failed: {e}")
            os.replace(tmp_file, self.agent_state_file)
            self.logger.debug(f"State saved atomically → {self.agent_state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)
            if "tmp_file" in locals() and tmp_file.exists():
                try:
                    os.remove(tmp_file)
                except OSError as rm_err:
                    self.logger.error(f"Failed remove tmp state file {tmp_file}: {rm_err}")

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
                    continue
                name = fld.name
                processed_keys.add(name)
                if name in data:
                    value = data[name]
                    if name == "current_plan":
                        try:
                            kwargs["current_plan"] = (
                                [PlanStep(**d) for d in value] if isinstance(value, list) else [PlanStep(description=DEFAULT_PLAN_STEP)]
                            )
                        except (ValidationError, TypeError) as e:
                            self.logger.warning(f"Plan reload failed: {e}. Resetting.")
                            kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    elif name == "tool_usage_stats":
                        dd = _default_tool_stats()
                        if isinstance(value, dict):
                            for k, v_dict in value.items():
                                if isinstance(v_dict, dict):
                                    dd[k]["success"] = int(v_dict.get("success", 0))
                                    dd[k]["failure"] = int(v_dict.get("failure", 0))
                                    dd[k]["latency_ms_total"] = float(v_dict.get("latency_ms_total", 0.0))
                        kwargs["tool_usage_stats"] = dd
                    elif name == "goal_stack":
                        kwargs[name] = value if isinstance(value, list) and all(isinstance(item, dict) for item in value) else []
                    else:
                        kwargs[name] = value
                else:  # Field missing, use default
                    if fld.default_factory is not dataclasses.MISSING:
                        kwargs[name] = fld.default_factory()
                    elif fld.default is not dataclasses.MISSING:
                        kwargs[name] = fld.default
                    elif name == "current_reflection_threshold":
                        kwargs[name] = BASE_REFLECTION_THRESHOLD
                    elif name == "current_consolidation_threshold":
                        kwargs[name] = BASE_CONSOLIDATION_THRESHOLD
                    elif name == "goal_stack":
                        kwargs[name] = []
                    elif name == "current_goal_id":
                        kwargs[name] = None
            extra_keys = set(data.keys()) - processed_keys - {"timestamp"}
            if extra_keys:
                self.logger.warning(f"Ignoring unknown keys in state file: {extra_keys}")
            temp_state = AgentState(**kwargs)
            # Validate/Correct loaded state
            if not isinstance(temp_state.current_reflection_threshold, int):
                temp_state.current_reflection_threshold = BASE_REFLECTION_THRESHOLD
            else:
                temp_state.current_reflection_threshold = max(
                    MIN_REFLECTION_THRESHOLD, min(MAX_REFLECTION_THRESHOLD, temp_state.current_reflection_threshold)
                )
            if not isinstance(temp_state.current_consolidation_threshold, int):
                temp_state.current_consolidation_threshold = BASE_CONSOLIDATION_THRESHOLD
            else:
                temp_state.current_consolidation_threshold = max(
                    MIN_CONSOLIDATION_THRESHOLD, min(MAX_CONSOLIDATION_THRESHOLD, temp_state.current_consolidation_threshold)
                )
            if not isinstance(temp_state.goal_stack, list):
                temp_state.goal_stack = []
            if temp_state.current_goal_id and not any(g.get("goal_id") == temp_state.current_goal_id for g in temp_state.goal_stack):
                self.logger.warning(f"Loaded current_goal_id {_fmt_id(temp_state.current_goal_id)} not in stack. Resetting.")
                temp_state.current_goal_id = temp_state.goal_stack[-1].get("goal_id") if temp_state.goal_stack else None
            self.state = temp_state
            self.logger.info(f"Loaded state from {self.agent_state_file}; loop {self.state.current_loop}")
        except (json.JSONDecodeError, TypeError, FileNotFoundError) as e:
            self.logger.error(f"State load failed: {e}. Resetting.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
        except Exception as e:
            self.logger.critical(f"Unexpected error loading state: {e}. Resetting.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )

    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        if not self.mcp_client or not self.mcp_client.server_manager:
            return None
        sm = self.mcp_client.server_manager
        if tool_name in sm.tools:
            server_name = sm.tools[tool_name].server_name
            return server_name if server_name in sm.active_sessions else None
        if tool_name.startswith("core:") and "CORE" in sm.active_sessions:
            return "CORE"
        if tool_name == AGENT_TOOL_UPDATE_PLAN:
            return "AGENT_INTERNAL"
        return None

    async def inject_manual_thought(self, content: str, thought_type: str = ThoughtType.USER_GUIDANCE.value) -> bool: # Default to new enum
        """
        Allows external injection of a thought into the agent's current reasoning process.
        This thought will be recorded in UMS and optionally stored as a distinct memory.
        """
        if not self.state.workflow_id or not self.state.current_thought_chain_id:
            self.logger.warning("Cannot inject thought: No active workflow or thought chain.")
            return False

        # Validate the provided thought_type against the enum
        try:
            validated_thought_type = ThoughtType(thought_type.lower()).value
        except ValueError:
            self.logger.error(f"AML: Invalid thought_type '{thought_type}' for injected thought. Defaulting to USER_GUIDANCE.")
            validated_thought_type = ThoughtType.USER_GUIDANCE.value

        self.logger.info(f"AML: Injecting manual thought (type: {validated_thought_type}): '{content[:100]}...'")
        try:
            thought_payload = {
                "workflow_id": self.state.workflow_id,
                "content": content,
                "thought_type": validated_thought_type,
                "thought_chain_id": self.state.current_thought_chain_id,
            }
            result = await self._execute_tool_call_internal(
                TOOL_RECORD_THOUGHT,
                thought_payload,
                record_action=False # This is an input, not a self-decided agent action
            )

            if result.get("success") and result.get("thought_id"):
                thought_id = result.get("thought_id")
                self.logger.info(f"AML: Successfully injected and recorded thought ID: {_fmt_id(thought_id)}")

                # Store this injected thought also as a distinct, high-priority memory
                # This makes it more discoverable by general memory searches.
                memory_store_payload = {
                    "workflow_id": self.state.workflow_id,
                    "content": f"User Guidance Received: {content}",
                    "memory_type": MemoryType.USER_INPUT.value, # Use the new memory type
                    "memory_level": MemoryLevel.SEMANTIC.value, # Or EPISODIC if preferred, SEMANTIC makes sense for guidance
                    "importance": 8.5, # Give user input high importance
                    "confidence": 0.95, # Assume user input is fairly reliable
                    "description": "Guidance or input directly provided by the user/operator.",
                    "tags": ["user_input", "guidance", "manual_override_data"],
                    "thought_id": thought_id, # Link memory back to the thought
                    "generate_embedding": True, # Good to embed user inputs
                }
                mem_result = await self._execute_tool_call_internal(
                    TOOL_STORE_MEMORY,
                    memory_store_payload,
                    record_action=False # Logging this as a separate agent action might be too noisy
                )
                if mem_result.get("success") and mem_result.get("memory_id"):
                    self.logger.info(f"AML: Stored injected guidance as UMS memory ID: {_fmt_id(mem_result.get('memory_id'))}")
                else:
                    self.logger.warning(f"AML: Failed to store injected guidance as a separate memory: {mem_result.get('error')}")

                self.state.last_action_summary = f"Received guidance: {content[:70]}..." # Update for visibility in agent status
                self.state.needs_replan = True
                self.logger.info("AML: Marked 'needs_replan = True' due to new user guidance.")
                return True
            else:
                self.logger.error(f"AML: Failed to record injected thought via UMS: {result.get('error')}")
                return False
        except Exception as e:
            self.logger.error(f"AML: Exception while injecting manual thought: {e}", exc_info=True)
            return False
        

    async def initialize(self) -> bool:
        self.logger.info("Initializing Agent loop …")
        await self._load_agent_state()
        if self.state.workflow_id and not self.state.context_id:
            self.state.context_id = self.state.workflow_id
            self.logger.info(f"Initialized context_id from workflow_id: {_fmt_id(self.state.workflow_id)}")
        try:
            if not self.mcp_client.server_manager:
                self.logger.error("MCP Client server manager not init.")
                return False

            # Determine the provider for the agent's LLM model
            agent_provider = self.mcp_client.get_provider_from_model(self.agent_llm_model)
            if not agent_provider:
                self.logger.error(f"Could not determine provider for agent's model '{self.agent_llm_model}'. Cannot format tools.")
                return False

            # Call MCPClient's method to get tools formatted for the agent's provider
            # _format_tools_for_provider is a method of MCPClient, not ServerManager
            all_formatted_tools = self.mcp_client._format_tools_for_provider(agent_provider) # Pass the determined provider

            if all_formatted_tools is None: # Can be None if no tools or formatting fails
                all_formatted_tools = [] 
                self.logger.warning(f"No tools formatted by MCPClient for provider '{agent_provider}'. Agent might have limited capabilities.")

            self.tool_schemas = [] 
            loaded_tool_names = set() # This will store ORIGINAL MCP tool names that are kept

            if all_formatted_tools: # Ensure it's not None
                for llm_tool_schema in all_formatted_tools: # llm_tool_schema is what the LLM will see
                    sanitized_name_for_llm = ""
                    if isinstance(llm_tool_schema, dict):
                        if "name" in llm_tool_schema: # Anthropic format
                            sanitized_name_for_llm = llm_tool_schema["name"]
                        elif "function" in llm_tool_schema and \
                            isinstance(llm_tool_schema["function"], dict) and \
                            "name" in llm_tool_schema["function"]: # OpenAI format
                            sanitized_name_for_llm = llm_tool_schema["function"]["name"]
                    
                    if not sanitized_name_for_llm:
                        self.logger.warning(f"AML Initialize: Skipping tool schema with unexpected structure or missing name: {str(llm_tool_schema)[:200]}")
                        continue

                    # Look up the original MCP name using the sanitized name from the LLM schema
                    original_mcp_tool_name = self.mcp_client.server_manager.sanitized_to_original.get(sanitized_name_for_llm)

                    if original_mcp_tool_name:
                        # Check the prefix of the ORIGINAL MCP name
                        if original_mcp_tool_name.startswith("unified_memory:"):
                            self.tool_schemas.append(llm_tool_schema) # Add the LLM-formatted schema
                            loaded_tool_names.add(original_mcp_tool_name) # Track by original name
                    # We will add AGENT_TOOL_UPDATE_PLAN separately as it's not from MCP server_manager
                    # else: # If not found in map, it might be a tool not originating from MCP server (like agent:update_plan later)
                    #    self.logger.debug(f"AML Initialize: Sanitized name '{sanitized_name_for_llm}' not in MCPClient's original name map. Likely an agent-internal tool or new.")

            # Now, add AGENT_TOOL_UPDATE_PLAN if it wasn't already processed (it shouldn't be in all_formatted_tools from MCPC)
            # Construct its schema as the LLM needs to see it (this depends on the agent_provider)
            plan_tool_llm_schema = { # Generic structure, adapt if provider needs specific 'type: function' wrapper
                "name": AGENT_TOOL_UPDATE_PLAN,
                "description": "Replace agent's current plan. Use for significant replanning, error recovery, or fixing validation issues.",
                "input_schema": { # For Anthropic style
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": PlanStep.model_json_schema(), # Get Pydantic schema
                            "description": "The new complete list of plan steps."
                        }
                    },
                    "required": ["plan"],
                }
            }
            # If OpenAI provider, wrap in "function" and "parameters"
            if agent_provider == "openai": # Assuming Provider.OPENAI.value is 'openai'
                plan_tool_llm_schema = {
                    "type": "function",
                    "function": {
                        "name": AGENT_TOOL_UPDATE_PLAN, # OpenAI sanitization rules apply
                        "description": plan_tool_llm_schema["description"],
                        "parameters": plan_tool_llm_schema["input_schema"]
                    }
                }
            # Add other provider specific formatting for AGENT_TOOL_UPDATE_PLAN if necessary

            self.tool_schemas.append(plan_tool_llm_schema)
            loaded_tool_names.add(AGENT_TOOL_UPDATE_PLAN) # Track it as loaded

            self.logger.info(f"AML: Loaded {len(self.tool_schemas)} relevant tool schemas for agent's LLM ({self.agent_llm_model} / {agent_provider}): {sorted(loaded_tool_names)}")

            essential = [ # These are ORIGINAL MCP tool names
                TOOL_CREATE_WORKFLOW, TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION,
                TOOL_RECORD_THOUGHT, TOOL_STORE_MEMORY, TOOL_GET_WORKING_MEMORY,
                TOOL_HYBRID_SEARCH, TOOL_GET_RICH_CONTEXT_PACKAGE, TOOL_REFLECTION,
                TOOL_CONSOLIDATION, TOOL_GET_WORKFLOW_DETAILS, TOOL_CREATE_GOAL,
                TOOL_UPDATE_GOAL_STATUS, TOOL_GET_GOAL_DETAILS, AGENT_TOOL_UPDATE_PLAN,
            ]
            missing = [t for t in essential if t not in loaded_tool_names] # Check against original names

            if missing:
                self.logger.error(f"AML: Missing essential tools from available schemas: {missing}. Functionality WILL BE impaired.")

            # Rest of the initialization (workflow check, goal stack validation, default thought chain)
            top_wf = (self.state.workflow_stack[-1] if self.state.workflow_stack else None) or self.state.workflow_id
            if top_wf and not await self._check_workflow_exists(top_wf):
                self.logger.warning(f"AML: Stored workflow '{_fmt_id(top_wf)}' not found; resetting workflow state.")
                preserved_stats = self.state.tool_usage_stats
                pres_ref_thresh = self.state.current_reflection_threshold
                pres_con_thresh = self.state.current_consolidation_threshold
                self.state = AgentState(
                    tool_usage_stats=preserved_stats, current_reflection_threshold=pres_ref_thresh, current_consolidation_threshold=pres_con_thresh
                )
                await self._save_agent_state()
            
            await self._validate_goal_stack_on_load()
            
            if self.state.workflow_id and not self.state.current_thought_chain_id:
                await self._set_default_thought_chain_id()
            
            self.logger.info("Agent loop initialization complete.")
            return True
        except Exception as e:
            self.logger.critical(f"Agent loop init failed: {e}", exc_info=True)
            return False


    async def _set_default_thought_chain_id(self):
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.debug("No active workflow for default thought chain.")
            return
        get_details_tool = TOOL_GET_WORKFLOW_DETAILS
        if self._find_tool_server(get_details_tool):
            try:
                details = await self._execute_tool_call_internal(
                    get_details_tool,
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
                    self.logger.error(f"Tool '{get_details_tool}' failed for default chain: {details.get('error')}")
            except Exception as e:
                self.logger.error(f"Error fetching WF details for default chain: {e}", exc_info=False)
        else:
            self.logger.warning(f"Tool '{get_details_tool}' unavailable for default chain.")
        self.logger.info(f"Could not set primary thought chain ID for WF {_fmt_id(current_wf_id)}.")

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        self.logger.debug(f"Checking existence of workflow {_fmt_id(workflow_id)} using {TOOL_GET_WORKFLOW_DETAILS}.")
        tool_name = TOOL_GET_WORKFLOW_DETAILS
        if not self._find_tool_server(tool_name):
            self.logger.error(f"Tool {tool_name} unavailable.")
            return False
        try:
            result = await self._execute_tool_call_internal(
                tool_name,
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
        except ToolInputError as e:
            self.logger.debug(f"Workflow {_fmt_id(workflow_id)} not found (ToolInputError: {e}).")
            return False
        except Exception as e:
            self.logger.error(f"Error checking WF {_fmt_id(workflow_id)}: {e}", exc_info=False)
            return False

    async def _validate_goal_stack_on_load(self):
        """
        Validates the loaded agent's goal stack against the UMS.
        If `self.state.current_goal_id` is set, it rebuilds the stack from UMS.
        If not, it clears the local stack.
        """
        if not self.state.workflow_id:
            self.logger.warning("Cannot validate goal stack on load: No active workflow ID in state.")
            self.state.goal_stack = []
            self.state.current_goal_id = None
            return

        if self.state.current_goal_id:
            self.logger.info(f"Validating loaded goal stack, current_goal_id: {_fmt_id(self.state.current_goal_id)}")
            # Rebuild the stack from UMS based on the current_goal_id
            ums_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
            if ums_stack:
                # Check if the UMS-fetched stack's leaf goal matches our current_goal_id
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
            else:  # UMS couldn't reconstruct the stack for the current_goal_id
                self.logger.warning(
                    f"Could not reconstruct goal stack from UMS for loaded current_goal_id {_fmt_id(self.state.current_goal_id)}. "
                    f"Resetting goal stack and current_goal_id."
                )
                self.state.goal_stack = []
                self.state.current_goal_id = None
        else:  # No current_goal_id was loaded, so clear the stack
            if self.state.goal_stack:  # If stack had items but no current_goal_id
                self.logger.info("Loaded goal stack present but no current_goal_id. Clearing local stack.")
                self.state.goal_stack = []
            else:  # Both current_goal_id and stack are empty, which is fine
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
        tool_name = TOOL_GET_ACTION_DETAILS
        if not self._find_tool_server(tool_name):
            self.logger.error(f"Tool {tool_name} unavailable.")
            return False, f"Tool {tool_name} unavailable."
        self.logger.debug(f"Checking prerequisites: {[_fmt_id(i) for i in ids]}")
        try:
            res = await self._execute_tool_call_internal(tool_name, {"action_ids": ids, "include_dependencies": False}, record_action=False)
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
        self, tool_name: str, tool_args: Dict[str, Any], planned_dependencies: Optional[List[str]] = None
    ) -> Optional[str]:
        start_tool = TOOL_RECORD_ACTION_START
        if not self._find_tool_server(start_tool):
            self.logger.error(f"Tool '{start_tool}' unavailable.")
            return None
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning("No active WF ID for action start.")
            return None
        payload = {
            "workflow_id": current_wf_id,
            "title": f"Execute: {tool_name.split(':')[-1]}",
            "action_type": ActionType.TOOL_USE.value,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reasoning": f"Agent initiated: {tool_name}",
        }
        action_id: Optional[str] = None
        try:
            res = await self._execute_tool_call_internal(start_tool, payload, record_action=False)
            if res.get("success"):
                action_id = res.get("action_id")
                if action_id:
                    self.logger.debug(f"Action started: {_fmt_id(action_id)} for {tool_name}")
                    if planned_dependencies:
                        await self._record_action_dependencies_internal(action_id, planned_dependencies)
                else:
                    self.logger.warning(f"{start_tool} success but no action_id.")
            else:
                self.logger.error(f"Failed record action start for {tool_name}: {res.get('error')}")
        except Exception as e:
            self.logger.error(f"Exception recording action start for {tool_name}: {e}", exc_info=True)
        return action_id

    async def _record_action_dependencies_internal(self, source_id: str, target_ids: List[str]) -> None:
        if not source_id or not target_ids:
            return
        valid_targets = {tid for tid in target_ids if tid and tid != source_id}
        if not valid_targets:
            return
        dep_tool = TOOL_ADD_ACTION_DEPENDENCY
        if not self._find_tool_server(dep_tool):
            self.logger.error(f"Tool '{dep_tool}' unavailable.")
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
                    dep_tool, {"source_action_id": source_id, "target_action_id": target_id, "dependency_type": "requires"}, record_action=False
                )
            )
            for target_id in valid_targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            target_id = list(valid_targets)[i]
            if isinstance(res, Exception):
                self.logger.error(f"Error recording dep {_fmt_id(source_id)} -> {_fmt_id(target_id)}: {res}", exc_info=False)
            elif isinstance(res, dict) and not res.get("success"):
                self.logger.warning(f"Failed recording dep {_fmt_id(source_id)} -> {_fmt_id(target_id)}: {res.get('error')}")

    async def _record_action_completion_internal(self, action_id: str, result: Dict[str, Any]) -> None:
        completion_tool = TOOL_RECORD_ACTION_COMPLETION
        if not self._find_tool_server(completion_tool):
            self.logger.error(f"Tool '{completion_tool}' unavailable.")
            return
        status = ActionStatus.COMPLETED.value if isinstance(result, dict) and result.get("success") else ActionStatus.FAILED.value
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"No active WF ID for action completion {_fmt_id(action_id)}.")
            return
        payload = {"action_id": action_id, "status": status, "tool_result": result}
        try:
            completion_res = await self._execute_tool_call_internal(completion_tool, payload, record_action=False)
            if completion_res.get("success"):
                self.logger.debug(f"Action completion recorded for {_fmt_id(action_id)} (Status: {status})")
            else:
                self.logger.error(f"Failed record action completion for {_fmt_id(action_id)}: {completion_res.get('error')}")
        except Exception as e:
            self.logger.error(f"Exception recording action completion for {_fmt_id(action_id)}: {e}", exc_info=True)

    async def _run_auto_linking(self, memory_id: str, *, workflow_id: Optional[str], context_id: Optional[str]) -> None:
        if workflow_id != self.state.workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping auto-link for {_fmt_id(memory_id)}: WF changed or shutdown.")
            return
        try:
            if not memory_id or not workflow_id:
                self.logger.debug(f"Skipping auto-link: Missing ID/WFID.")
                return
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            if self._shutdown_event.is_set():
                return
            self.logger.debug(f"Attempting auto-link for memory {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}...")
            source_res = await self._execute_tool_call_internal(
                TOOL_GET_MEMORY_BY_ID, {"memory_id": memory_id, "include_links": False}, record_action=False
            )
            if not source_res.get("success") or source_res.get("workflow_id") != workflow_id:
                self.logger.warning(f"Auto-link failed for {_fmt_id(memory_id)}: Source mem error.")
                return
            source_mem = source_res
            query_text = source_mem.get("description", "") or source_mem.get("content", "")[:200]
            if not query_text:
                self.logger.debug(f"Skipping auto-link for {_fmt_id(memory_id)}: No query text.")
                return
            search_tool = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
            if not self._find_tool_server(search_tool):
                self.logger.warning(f"Skipping auto-link: Tool {search_tool} unavailable.")
                return
            search_args = {
                "workflow_id": workflow_id,
                "query": query_text,
                "limit": self.auto_linking_max_links + 1,
                "threshold": self.auto_linking_threshold,
                "include_content": False,
            }
            if search_tool == TOOL_HYBRID_SEARCH:
                search_args.update({"semantic_weight": 0.8, "keyword_weight": 0.2})
            similar_res = await self._execute_tool_call_internal(search_tool, search_args, record_action=False)
            if not similar_res.get("success"):
                self.logger.warning(f"Auto-link search failed for {_fmt_id(memory_id)}: {similar_res.get('error')}")
                return
            link_count = 0
            score_key = "hybrid_score" if search_tool == TOOL_HYBRID_SEARCH else "similarity"
            for sim_mem_summary in similar_res.get("memories", []):
                if self._shutdown_event.is_set():
                    break
                target_id = sim_mem_summary.get("memory_id")
                sim_score = sim_mem_summary.get(score_key, 0.0)
                if not target_id or target_id == memory_id:
                    continue
                target_res = await self._execute_tool_call_internal(
                    TOOL_GET_MEMORY_BY_ID, {"memory_id": target_id, "include_links": False}, record_action=False
                )
                if not target_res.get("success") or target_res.get("workflow_id") != workflow_id:
                    continue
                target_mem = target_res
                link_type = LinkType.RELATED.value
                source_type = source_mem.get("memory_type")
                target_type = target_mem.get("memory_type")
                if source_type == MemoryType.INSIGHT.value and target_type == MemoryType.FACT.value:
                    link_type = LinkType.SUPPORTS.value
                # ... (more link type rules)
                link_tool = TOOL_CREATE_LINK
                if not self._find_tool_server(link_tool):
                    self.logger.warning(f"Tool {link_tool} unavailable.")
                    break
                link_args = {
                    "source_memory_id": memory_id,
                    "target_memory_id": target_id,
                    "link_type": link_type,
                    "strength": round(sim_score, 3),
                    "description": f"Auto-link ({link_type})",
                }
                link_result = await self._execute_tool_call_internal(link_tool, link_args, record_action=False)
                if link_result.get("success"):
                    link_count += 1
                    self.logger.debug(f"Auto-linked {_fmt_id(memory_id)} to {_fmt_id(target_id)} ({link_type}, {sim_score:.2f})")
                else:
                    self.logger.warning(f"Failed auto-create link {_fmt_id(memory_id)}->{_fmt_id(target_id)}: {link_result.get('error')}")
                if link_count >= self.auto_linking_max_links:
                    break
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.warning(f"Error in auto-linking for {_fmt_id(memory_id)}: {e}", exc_info=False)

    async def _check_and_trigger_promotion(self, memory_id: str, *, workflow_id: Optional[str], context_id: Optional[str]):
        if workflow_id != self.state.workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping promo check for {_fmt_id(memory_id)}: WF changed/shutdown.")
            return
        promo_tool = TOOL_PROMOTE_MEM
        if not memory_id or not self._find_tool_server(promo_tool):
            self.logger.debug(f"Skipping promo check for {_fmt_id(memory_id)}: Invalid ID or tool unavailable.")
            return
        try:
            await asyncio.sleep(random.uniform(0.1, 0.4))
            if self._shutdown_event.is_set():
                return
            self.logger.debug(f"Checking promo potential for memory {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}...")
            promo_res = await self._execute_tool_call_internal(promo_tool, {"memory_id": memory_id}, record_action=False)
            if promo_res.get("success"):
                if promo_res.get("promoted"):
                    self.logger.info(f"⬆️ Memory {_fmt_id(memory_id)} promoted from {promo_res.get('previous_level')} to {promo_res.get('new_level')}.")
                else:
                    self.logger.debug(f"Memory {_fmt_id(memory_id)} not promoted: {promo_res.get('reason')}")
            else:
                self.logger.warning(f"Promo check tool failed for {_fmt_id(memory_id)}: {promo_res.get('error')}")
        except Exception as e:
            self.logger.warning(f"Error in promo check task for {_fmt_id(memory_id)}: {e}", exc_info=False)

    async def _execute_tool_call_internal(
        self, tool_name: str, arguments: Dict[str, Any], record_action: bool = True, planned_dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        target_server = self._find_tool_server(tool_name)
        if not target_server and tool_name != AGENT_TOOL_UPDATE_PLAN:
            err = f"Tool server unavailable for {tool_name}"
            self.logger.error(err)
            self.state.last_error_details = {"tool": tool_name, "error": err, "type": "ServerUnavailable", "status_code": 503}
            return {"success": False, "error": err, "status_code": 503}
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_ctx_id = self.state.context_id
        current_goal_id_for_tool = self.state.current_goal_id
        final_arguments = arguments.copy()
        if (
            final_arguments.get("workflow_id") is None
            and current_wf_id
            and tool_name
            not in {
                TOOL_CREATE_WORKFLOW,
                TOOL_LIST_WORKFLOWS,
                "core:list_servers",
                "core:get_tool_schema",
                AGENT_TOOL_UPDATE_PLAN,
                TOOL_CREATE_GOAL,
                TOOL_UPDATE_GOAL_STATUS,
                TOOL_GET_GOAL_DETAILS,
            }
        ):
            final_arguments["workflow_id"] = current_wf_id
        if final_arguments.get("context_id") is None and current_ctx_id and tool_name in {TOOL_GET_WORKING_MEMORY, TOOL_OPTIMIZE_WM, TOOL_AUTO_FOCUS}:
            final_arguments["context_id"] = current_ctx_id
        if final_arguments.get("thought_chain_id") is None and self.state.current_thought_chain_id and tool_name == TOOL_RECORD_THOUGHT:
            final_arguments["thought_chain_id"] = self.state.current_thought_chain_id
        # Updated: Use TOOL_CREATE_GOAL for UMS interaction
        if final_arguments.get("parent_goal_id") is None and current_goal_id_for_tool and tool_name == TOOL_CREATE_GOAL:
            final_arguments["parent_goal_id"] = current_goal_id_for_tool
        if planned_dependencies:
            ok, reason = await self._check_prerequisites(planned_dependencies)
            if not ok:
                err_msg = f"Prerequisites not met for {tool_name}: {reason}"
                self.logger.warning(err_msg)
                self.state.last_error_details = {
                    "tool": tool_name,
                    "error": err_msg,
                    "type": "DependencyNotMetError",
                    "dependencies": planned_dependencies,
                    "status_code": 412,
                }
                self.state.needs_replan = True
                return {"success": False, "error": err_msg, "status_code": 412}
            self.logger.info(f"Prerequisites {[_fmt_id(d) for d in planned_dependencies]} met for {tool_name}.")
        if tool_name == AGENT_TOOL_UPDATE_PLAN:
            try:
                new_plan_data = final_arguments.get("plan", [])
                if not isinstance(new_plan_data, list):
                    raise ValueError("`plan` must be a list.")
                validated_plan = [PlanStep(**p) for p in new_plan_data]
                if self._detect_plan_cycle(validated_plan):
                    err_msg = "Proposed plan contains a dependency cycle."
                    self.logger.error(err_msg)
                    self.state.last_error_details = {
                        "tool": tool_name,
                        "error": err_msg,
                        "type": "PlanValidationError",
                        "proposed_plan": new_plan_data,
                    }
                    self.state.needs_replan = True
                    return {"success": False, "error": err_msg}
                self.state.current_plan = validated_plan
                self.state.needs_replan = False
                self.logger.info(f"Internal plan update successful ({len(validated_plan)} steps).")
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                return {"success": True, "message": f"Plan updated with {len(validated_plan)} steps."}
            except (ValidationError, TypeError, ValueError) as e:
                err_msg = f"Failed to validate/apply new plan: {e}"
                self.logger.error(err_msg)
                self.state.last_error_details = {
                    "tool": tool_name,
                    "error": err_msg,
                    "type": "PlanUpdateError",
                    "proposed_plan": final_arguments.get("plan"),
                }
                self.state.consecutive_error_count += 1
                self.state.needs_replan = True
                return {"success": False, "error": err_msg}
        action_id: Optional[str] = None
        should_record = record_action and tool_name not in self._INTERNAL_OR_META_TOOLS
        if should_record:
            action_id = await self._record_action_start_internal(tool_name, final_arguments, planned_dependencies)
        record_stats = self.state.tool_usage_stats[tool_name]
        idempotent = tool_name in {
            TOOL_GET_CONTEXT,
            TOOL_GET_RICH_CONTEXT_PACKAGE,
            TOOL_GET_MEMORY_BY_ID,
            TOOL_SEMANTIC_SEARCH,
            TOOL_HYBRID_SEARCH,
            TOOL_GET_ACTION_DETAILS,
            TOOL_LIST_WORKFLOWS,
            TOOL_COMPUTE_STATS,
            TOOL_GET_WORKING_MEMORY,
            TOOL_GET_LINKED_MEMORIES,
            TOOL_GET_ARTIFACTS,
            TOOL_GET_ARTIFACT_BY_ID,
            TOOL_GET_ACTION_DEPENDENCIES,
            TOOL_GET_THOUGHT_CHAIN,
            TOOL_GET_WORKFLOW_DETAILS,
            TOOL_GET_GOAL_DETAILS,
            TOOL_SUMMARIZE_TEXT,
        }
        start_ts = time.time()
        res = {}
        try:

            async def _do_call():
                call_args = {k: v for k, v in final_arguments.items() if v is not None}
                return await self.mcp_client.execute_tool(target_server, tool_name, call_args)

            raw = await self._with_retries(
                _do_call,
                max_retries=3 if idempotent else 1,
                retry_exceptions=(
                    ToolError,
                    ToolInputError,
                    asyncio.TimeoutError,
                    ConnectionError,
                    APIConnectionError,
                    RateLimitError,
                    APIStatusError,
                ),
            )
            latency_ms = (time.time() - start_ts) * 1000
            record_stats["latency_ms_total"] += latency_ms
            if isinstance(raw, dict) and ("success" in raw or "isError" in raw):
                is_error = raw.get("isError", not raw.get("success", True))
                content = raw.get("content", raw.get("error", raw.get("data")))
                if is_error:
                    res = {"success": False, "error": str(content), "status_code": raw.get("status_code")}
                else:
                    res = content if isinstance(content, dict) and "success" in content else {"success": True, "data": content}
            elif isinstance(raw, dict):
                res = {"success": True, "data": raw}
            else:
                res = {"success": True, "data": raw}
            if res.get("success"):
                record_stats["success"] += 1
                self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
                if tool_name in [TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY] and res.get("memory_id"):
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=res["memory_id"])
                if tool_name == TOOL_RECORD_ARTIFACT and res.get("linked_memory_id"):
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=res["linked_memory_id"])
                if tool_name in [TOOL_GET_MEMORY_BY_ID, TOOL_QUERY_MEMORIES, TOOL_HYBRID_SEARCH, TOOL_SEMANTIC_SEARCH, TOOL_GET_WORKING_MEMORY]:
                    mem_ids_to_check = set()
                    potential_mems = []
                    if tool_name == TOOL_GET_MEMORY_BY_ID:
                        mem_data = res if "memory_id" in res else res.get("data", {})
                        potential_mems = [mem_data] if isinstance(mem_data, dict) else []
                    elif tool_name == TOOL_GET_WORKING_MEMORY:
                        potential_mems = res.get("working_memories", [])
                        focus_id = res.get("focal_memory_id")
                    if focus_id:
                        mem_ids_to_check.add(focus_id)  # type: ignore
                    else:
                        potential_mems = res.get("memories", [])
                    if isinstance(potential_mems, list):
                        mem_ids_to_check.update(m.get("memory_id") for m in potential_mems[:3] if isinstance(m, dict) and m.get("memory_id"))
                    for mem_id_chk in filter(None, mem_ids_to_check):
                        self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id_chk)
                if tool_name == TOOL_CREATE_THOUGHT_CHAIN and res.get("success"):
                    chain_data = res if "thought_chain_id" in res else res.get("data", {})
                    if isinstance(chain_data, dict):
                        new_chain_id = chain_data.get("thought_chain_id")
                    if new_chain_id:
                        self.state.current_thought_chain_id = new_chain_id
                        self.logger.info(f"Switched current thought chain: {_fmt_id(new_chain_id)}")  # type: ignore
            else:
                record_stats["failure"] += 1
                error_type = "ToolExecutionError"
                status_code = res.get("status_code")
                error_message = res.get("error", "Unknown failure")
                if status_code == 412:
                    error_type = "DependencyNotMetError"
                elif status_code == 503:
                    error_type = "ServerUnavailable"
                elif "input" in str(error_message).lower() or "validation" in str(error_message).lower():
                    error_type = "InvalidInputError"
                elif "timeout" in str(error_message).lower():
                    error_type = "NetworkError"
                elif tool_name in [TOOL_CREATE_GOAL, TOOL_UPDATE_GOAL_STATUS] and (
                    "not found" in str(error_message).lower() or "invalid" in str(error_message).lower()
                ):
                    error_type = "GoalManagementError"
                self.state.last_error_details = {
                    "tool": tool_name,
                    "args": arguments,
                    "error": error_message,
                    "status_code": status_code,
                    "type": error_type,
                }
                self.logger.warning(f"Tool {tool_name} failed. Type: {error_type}, Error: {error_message}")
            summary = ""
            if res.get("success"):
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
                ]
                data_payload = res.get("data", res)
                if isinstance(data_payload, dict):
                    for k in summary_keys:
                        if k in data_payload and data_payload[k]:
                            summary = f"{k}: {_fmt_id(data_payload[k]) if 'id' in k else str(data_payload[k])}"
                            break
                    else:
                        data_str = str(data_payload)[:70]
                        summary = f"Success. Data: {data_str}{'...' if len(str(data_payload)) > 70 else ''}"
                else:
                    data_str = str(data_payload)[:70]
                    summary = f"Success. Data: {data_str}{'...' if len(str(data_payload)) > 70 else ''}"
            else:
                err_type = self.state.last_error_details.get("type", "Unknown") if self.state.last_error_details else "Unknown"
                err_msg = str(res.get("error", "Unknown Error"))[:100]
                summary = f"Failed ({err_type}): {err_msg}"
            if res.get("status_code"):
                summary += f" (Code: {res['status_code']})"
            self.state.last_action_summary = f"{tool_name} -> {summary}"
            self.logger.info(f"{"🏁" if res.get("success") else "⚠️"} {self.state.last_action_summary}")
        except (ToolError, ToolInputError) as e:
            err_str = str(e)
            status_code = getattr(e, "status_code", None)
            error_type = "InvalidInputError" if isinstance(e, ToolInputError) else "ToolInternalError"
            if status_code == 412:
                error_type = "DependencyNotMetError"
            self.logger.error(f"Tool Error exec {tool_name}: {err_str}", exc_info=False)
            res = {"success": False, "error": err_str, "status_code": status_code}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": error_type, "status_code": status_code}
            self.state.last_action_summary = f"{tool_name} -> Failed ({error_type}): {err_str[:100]}"
        except APIConnectionError as e:
            err_str = f"LLM API Conn Error: {e}"
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
        except asyncio.TimeoutError as e:
            err_str = f"Op timed out: {e}"
            self.logger.error(f"Timeout exec {tool_name}: {err_str}", exc_info=False)
            res = {"success": False, "error": err_str}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "TimeoutError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: Timeout"
        except asyncio.CancelledError:
            err_str = "Tool exec cancelled."
            self.logger.warning(f"{tool_name} exec cancelled.")
            res = {"success": False, "error": err_str, "status_code": 499}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "CancelledError"}
            self.state.last_action_summary = f"{tool_name} -> Cancelled"
            raise
        except Exception as e:
            err_str = str(e)
            self.logger.error(f"Unexpected Error exec {tool_name}: {err_str}", exc_info=True)
            res = {"success": False, "error": f"Unexpected error: {err_str}"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "UnexpectedExecutionError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: Unexpected error."
        if action_id:
            await self._record_action_completion_internal(action_id, res)
        await self._handle_workflow_and_goal_side_effects(tool_name, final_arguments, res)  # Pass original arguments
        return res

    async def _handle_workflow_and_goal_side_effects(self, tool_name: str, arguments: Dict, result_content: Dict):
        """
        Handles agent state changes triggered by specific tool outcomes,
        especially workflow creation/termination and UMS-driven goal stack updates.
        """
        # --- Side effects for Workflow Creation ---
        if tool_name == TOOL_CREATE_WORKFLOW and result_content.get("success"):
            new_wf_id = result_content.get("workflow_id")
            primary_chain_id = result_content.get("primary_thought_chain_id")
            parent_wf_id_arg = arguments.get("parent_workflow_id")
            wf_title = result_content.get("title", "Untitled Workflow")
            wf_goal_desc = result_content.get("goal", "Achieve objectives for this workflow")  # Use UMS goal desc

            if new_wf_id:
                self.state.workflow_id = new_wf_id
                self.state.context_id = new_wf_id
                is_sub_workflow = parent_wf_id_arg and parent_wf_id_arg in self.state.workflow_stack
                log_prefix = "sub-" if is_sub_workflow else "new "
                if is_sub_workflow:
                    self.state.workflow_stack.append(new_wf_id)
                else:
                    self.state.workflow_stack = [new_wf_id]
                self.state.current_thought_chain_id = primary_chain_id

                # Create the root goal for this new workflow in UMS
                self.state.goal_stack = []  # Reset local stack for new/root workflow
                self.state.current_goal_id = None
                if self._find_tool_server(TOOL_CREATE_GOAL):
                    try:
                        goal_creation_args = {
                            "workflow_id": new_wf_id,
                            "description": wf_goal_desc,
                            # parent_goal_id is None for a root goal of a new workflow
                        }
                        self.logger.info(f"Attempting to create UMS root goal for {log_prefix}workflow {_fmt_id(new_wf_id)}: '{wf_goal_desc}'")
                        goal_res = await self._execute_tool_call_internal(TOOL_CREATE_GOAL, goal_creation_args, record_action=False)
                        # UMS create_goal should return the full goal object under a "goal" key
                        created_ums_goal = goal_res.get("goal") if goal_res.get("success") else None

                        if isinstance(created_ums_goal, dict) and created_ums_goal.get("goal_id"):
                            self.state.goal_stack.append(created_ums_goal)  # Add full UMS goal object
                            self.state.current_goal_id = created_ums_goal.get("goal_id")
                            self.logger.info(
                                f"Created UMS root goal {_fmt_id(self.state.current_goal_id)} for {log_prefix}workflow {_fmt_id(new_wf_id)}."
                            )
                        else:
                            self.logger.warning(
                                f"Failed to create UMS root goal for new workflow {_fmt_id(new_wf_id)}: {goal_res.get('error', 'UMS tool did not return valid goal data')}"
                            )
                    except Exception as goal_err:
                        self.logger.error(f"Error creating UMS root goal for new workflow {_fmt_id(new_wf_id)}: {goal_err}", exc_info=True)
                else:
                    self.logger.warning(f"Cannot create root goal for new workflow: UMS Tool {TOOL_CREATE_GOAL} unavailable.")

                self.logger.info(
                    f"🏷️ Switched to {log_prefix}workflow: {_fmt_id(new_wf_id)}. Chain: {_fmt_id(primary_chain_id)}. Current Goal: {_fmt_id(self.state.current_goal_id)}"
                )
                self.state.current_plan = [PlanStep(description=f"Start {log_prefix}workflow: '{wf_title}'. Goal: {wf_goal_desc}.")]
                self.state.consecutive_error_count = 0
                self.state.needs_replan = False
                self.state.last_error_details = None

        # --- Side effects for Creating a New Goal (typically a sub-goal) ---
        elif tool_name == TOOL_CREATE_GOAL and result_content.get("success"):
            # This case handles when the LLM decides to call create_goal, usually for a sub-goal
            created_ums_goal = result_content.get("goal")  # UMS tool returns the created goal object
            if isinstance(created_ums_goal, dict) and created_ums_goal.get("goal_id"):
                # Add the new UMS goal to the agent's local stack representation
                self.state.goal_stack.append(created_ums_goal)
                # Set the new goal as the current focus
                self.state.current_goal_id = created_ums_goal["goal_id"]
                self.logger.info(
                    f"📌 Pushed new UMS goal {_fmt_id(self.state.current_goal_id)} to local stack: '{created_ums_goal.get('description', '')[:50]}...'. Stack depth: {len(self.state.goal_stack)}"
                )
                self.state.needs_replan = True
                self.state.current_plan = [PlanStep(description=f"Start new goal: '{created_ums_goal.get('description', '')[:50]}...'")]
                self.state.last_error_details = None
            else:
                self.logger.warning(f"UMS Tool {TOOL_CREATE_GOAL} succeeded but did not return valid goal data: {result_content}")

        # --- Side effects for Updating Goal Status ---
        elif tool_name == TOOL_UPDATE_GOAL_STATUS and result_content.get("success"):
            goal_id_marked_in_ums = arguments.get("goal_id")  # The goal ID passed to the UMS tool
            new_status_in_ums = arguments.get("status")  # The new status passed to the UMS tool

            # Get data returned by the UMS tool
            updated_goal_details_from_ums = result_content.get("updated_goal_details")
            parent_goal_id_from_ums = result_content.get("parent_goal_id")  # Parent of the *marked* goal
            is_root_finished_from_ums = result_content.get("is_root_finished", False)

            # Validate the UMS response
            if not isinstance(updated_goal_details_from_ums, dict) or updated_goal_details_from_ums.get("goal_id") != goal_id_marked_in_ums:
                self.logger.error(
                    f"UMS {TOOL_UPDATE_GOAL_STATUS} returned inconsistent 'updated_goal_details' for {goal_id_marked_in_ums}. Aborting side-effects for this update."
                )
                return

            # 1. Update the specific goal in the agent's local stack with fresh UMS data
            goal_found_in_local_stack_and_updated = False
            for i, local_goal_dict in enumerate(self.state.goal_stack):
                if local_goal_dict.get("goal_id") == goal_id_marked_in_ums:
                    self.state.goal_stack[i] = updated_goal_details_from_ums  # Replace with fresh full object
                    goal_found_in_local_stack_and_updated = True
                    self.logger.info(f"Updated goal {_fmt_id(goal_id_marked_in_ums)} in local stack with UMS data (new status: {new_status_in_ums}).")
                    break
            if not goal_found_in_local_stack_and_updated:
                self.logger.warning(
                    f"Goal {_fmt_id(goal_id_marked_in_ums)} (marked {new_status_in_ums} in UMS) not found in current agent goal stack for update. Local stack: {[_fmt_id(g.get('goal_id')) for g in self.state.goal_stack]}"
                )

            # 2. If the goal that was just marked in UMS was our *current* active goal, and it's now terminal...
            if goal_id_marked_in_ums == self.state.current_goal_id and new_status_in_ums in ["completed", "failed", "abandoned"]:
                self.logger.info(f"Current active goal {_fmt_id(self.state.current_goal_id)} reached terminal state '{new_status_in_ums}'.")

                # Set the new current_goal_id based on what UMS returned for the parent of the marked goal.
                self.state.current_goal_id = parent_goal_id_from_ums  # This could be None if root was marked
                self.logger.info(f"Agent's current_goal_id updated to parent: {_fmt_id(self.state.current_goal_id)}")

                # Rebuild the local goal stack view from UMS, anchored by the new current_goal_id (or empty if it's None)
                if self.state.current_goal_id:
                    self.state.goal_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
                else:  # Current goal became None (meaning root goal was likely finished)
                    self.state.goal_stack = []  # Clear local stack

                self.logger.info(
                    f"🎯 Focus shifted. New current goal: {_fmt_id(self.state.current_goal_id) if self.state.current_goal_id else 'Overall Goal (stack empty)'}. Local stack depth: {len(self.state.goal_stack)}"
                )

                # 3. Check if the overall workflow/root goal is finished based on UMS
                if is_root_finished_from_ums:  # UMS explicitly said a root goal is now terminal
                    self.logger.info("UMS indicated a root goal is finished. Overall goal/workflow presumed finished.")
                    self.state.goal_achieved_flag = new_status_in_ums == "completed"  # Overall success depends on status of this root goal
                    # Update UMS workflow status
                    if self.state.workflow_id and self._find_tool_server(TOOL_UPDATE_WORKFLOW_STATUS):
                        final_wf_status = WorkflowStatus.COMPLETED.value if self.state.goal_achieved_flag else WorkflowStatus.FAILED.value
                        await self._execute_tool_call_internal(
                            TOOL_UPDATE_WORKFLOW_STATUS,
                            {
                                "workflow_id": self.state.workflow_id,
                                "status": final_wf_status,
                                "completion_message": f"Overall root goal marked {new_status_in_ums} via UMS.",
                            },
                            record_action=False,
                        )
                    self.state.current_plan = []  # Clear plan as workflow is done
                elif self.state.current_goal_id:  # If not root finished, but shifted to a new current goal
                    self.state.needs_replan = True
                    current_goal_desc_for_plan = "Unknown Goal"
                    # Find description for the new current goal from the (rebuilt) stack
                    for g_dict in self.state.goal_stack:
                        if g_dict.get("goal_id") == self.state.current_goal_id:
                            current_goal_desc_for_plan = g_dict.get("description", "Unknown Goal")[:50]
                            break
                    self.state.current_plan = [
                        PlanStep(
                            description=f"Returned from sub-goal {_fmt_id(goal_id_marked_in_ums)} (status: {new_status_in_ums}). Re-assess current goal: '{current_goal_desc_for_plan}...' ({_fmt_id(self.state.current_goal_id)})."
                        )
                    ]
                self.state.last_error_details = None  # Clear error details when goal status changes

        # --- Side effects for Workflow Status Update (Completion/Failure/Abandonment) ---
        elif tool_name == TOOL_UPDATE_WORKFLOW_STATUS and result_content.get("success"):
            status = arguments.get("status")  # Status requested in the tool call
            wf_id_updated = arguments.get("workflow_id")  # Workflow that was updated

            if wf_id_updated and self.state.workflow_stack and wf_id_updated == self.state.workflow_stack[-1]:
                is_terminal = status in [WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value, WorkflowStatus.ABANDONED.value]
                if is_terminal:
                    finished_wf = self.state.workflow_stack.pop()
                    parent_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else None
                    if parent_wf_id:
                        self.state.workflow_id = parent_wf_id
                        self.state.context_id = self.state.workflow_id
                        await self._set_default_thought_chain_id()
                        # If returning to a parent workflow, its goal stack should be re-established
                        # The current_goal_id for the parent workflow should be the goal that *spawned* the sub-workflow.
                        # This requires the agent to have stored this initiating_goal_id when it created the sub-workflow.
                        # For now, we'll assume the agent needs to re-evaluate its goals in the parent context.
                        # A robust way is to re-fetch the parent's current/leaf goal from UMS if its ID is known.
                        # As a simpler heuristic for now, if goal_stack has items and top one is for parent_wf_id, use it.
                        if self.state.goal_stack and self.state.goal_stack[-1].get("workflow_id") == parent_wf_id:
                            self.state.current_goal_id = self.state.goal_stack[-1].get("goal_id")
                        else:  # Fallback: clear stack and let agent figure it out, or try to find parent's leaf goal
                            self.state.goal_stack = []  # Clear local stack, agent needs to re-evaluate in parent WF
                            self.state.current_goal_id = None  # Or try to find active goal for parent_wf_id in UMS

                        self.logger.info(
                            f"⬅️ Sub-workflow {_fmt_id(finished_wf)} finished ({status}). Returning to parent {_fmt_id(self.state.workflow_id)}. Chain: {_fmt_id(self.state.current_thought_chain_id)}. Goal: {_fmt_id(self.state.current_goal_id)}"
                        )
                        self.state.needs_replan = True
                        self.state.current_plan = [
                            PlanStep(description=f"Sub-workflow {_fmt_id(finished_wf)} ({status}) finished. Re-assess current context.")
                        ]
                        self.state.last_error_details = None
                    else:  # Root workflow finished
                        self.logger.info(f"Root workflow {_fmt_id(finished_wf)} finished ({status}).")
                        self.state.workflow_id = None
                        self.state.context_id = None
                        self.state.current_thought_chain_id = None
                        self.state.current_plan = []
                        self.state.goal_stack = []
                        self.state.current_goal_id = None
                        self.state.goal_achieved_flag = status == WorkflowStatus.COMPLETED.value

    async def _fetch_goal_stack_from_ums(self, leaf_goal_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        Helper to reconstruct the current goal stack by querying UMS.
        Traverses from the given leaf_goal_id up to its root parent.

        Args:
            leaf_goal_id: The ID of the current/leaf goal to start from.
                          If None, returns an empty stack.

        Returns:
            A list of goal dictionaries (full UMS goal objects),
            ordered from root to leaf.
        """
        if not leaf_goal_id:
            self.logger.debug("_fetch_goal_stack_from_ums: No leaf_goal_id provided, returning empty stack.")
            return []

        tool_name = TOOL_GET_GOAL_DETAILS
        if not self._find_tool_server(tool_name):
            self.logger.warning(f"Tool {tool_name} unavailable. Cannot reconstruct goal stack from UMS.")
            return []  # Cannot fetch if tool is missing

        reconstructed_stack: List[Dict[str, Any]] = []
        current_id_to_fetch: Optional[str] = leaf_goal_id
        fetch_depth = 0
        max_fetch_depth = CONTEXT_GOAL_DETAILS_FETCH_LIMIT  # Use defined constant

        self.logger.debug(f"Reconstructing goal stack from UMS, starting with leaf: {_fmt_id(leaf_goal_id)}")

        while current_id_to_fetch and fetch_depth < max_fetch_depth:
            try:
                # Call UMS tool to get details of the current goal ID
                res = await self._execute_tool_call_internal(
                    tool_name,
                    {"goal_id": current_id_to_fetch},
                    record_action=False,  # Internal fetch, don't log as agent action
                )
                # The UMS get_goal_details tool returns the goal object under the "goal" key
                goal_data = res.get("goal") if isinstance(res, dict) and res.get("success") else None

                if isinstance(goal_data, dict):
                    reconstructed_stack.append(goal_data)  # Add the full goal object
                    parent_id = goal_data.get("parent_goal_id")
                    self.logger.debug(f"Fetched goal {_fmt_id(current_id_to_fetch)}. Parent: {_fmt_id(parent_id)}. Depth: {fetch_depth}")
                    current_id_to_fetch = parent_id  # Move to the parent for next iteration
                    fetch_depth += 1
                else:
                    self.logger.warning(
                        f"Failed to fetch details for goal {_fmt_id(current_id_to_fetch)} "
                        f"or invalid data received from UMS. Stopping stack reconstruction. Response: {res}"
                    )
                    break  # Stop if a goal in the chain is not found or data is invalid
            except Exception as e:
                self.logger.error(f"Exception while fetching goal {_fmt_id(current_id_to_fetch)} for stack reconstruction: {e}", exc_info=True)
                break  # Stop on any exception

        reconstructed_stack.reverse()  # Order from root (oldest parent) to leaf (current)
        self.logger.info(
            f"Reconstructed goal stack from UMS: {[_fmt_id(g.get('goal_id')) for g in reconstructed_stack]} (Depth: {len(reconstructed_stack)})"
        )
        return reconstructed_stack

    async def _apply_heuristic_plan_update(self, last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        self.logger.info("📋 Applying heuristic plan update (fallback)...")
        if not self.state.current_plan:
            self.state.current_plan = [PlanStep(description="Fallback: Re-evaluate.")]
            self.state.needs_replan = True
            return
        current_step = self.state.current_plan[0]
        decision_type = last_decision.get("decision")
        action_successful = False
        tool_name_executed = last_decision.get("tool_name")
        if decision_type == "call_tool" and tool_name_executed != AGENT_TOOL_UPDATE_PLAN:
            tool_success = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            action_successful = tool_success
            if tool_success:
                current_step.status = ActionStatus.COMPLETED.value
                summary = "Success."
                if isinstance(last_tool_result_content, dict):
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
                    ]
                    data_payload = last_tool_result_content.get("data", last_tool_result_content)
                    if isinstance(data_payload, dict):
                        for k in summary_keys:
                            if k in data_payload and data_payload[k]:
                                summary = f"{k}: {_fmt_id(data_payload[k]) if 'id' in k else str(data_payload[k])}"
                                break
                        else:
                            data_str = str(data_payload)[:70]
                            summary = f"Success. Data: {data_str}{'...' if len(str(data_payload)) > 70 else ''}"
                    else:
                        data_str = str(data_payload)[:70]
                        summary = f"Success. Data: {data_str}{'...' if len(str(data_payload)) > 70 else ''}"
                current_step.result_summary = summary[:150]
                self.state.current_plan.pop(0)
                if not self.state.current_plan:
                    self.state.current_plan.append(PlanStep(description="Plan finished. Analyze overall result and decide if goal is met."))
                self.state.needs_replan = False
            else:
                current_step.status = ActionStatus.FAILED.value
                error_msg = "Unknown failure"
                if isinstance(last_tool_result_content, dict):
                    error_details = self.state.last_error_details
                    if error_details:
                        error_msg = f"Type: {error_details.get('type', 'Unknown')}, Msg: {error_details.get('error', 'Unknown')}"
                    else:
                        error_msg = str(last_tool_result_content.get("error", "Unknown failure"))
                current_step.result_summary = f"Failure: {error_msg[:150]}"
                if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Analyze failure of step"):
                    self.state.current_plan.insert(
                        1, PlanStep(description=f"Analyze failure of step '{current_step.description[:30]}...' and replan.")
                    )
                self.state.needs_replan = True
        elif decision_type == "thought_process":
            action_successful = True
            current_step.status = ActionStatus.COMPLETED.value
            current_step.result_summary = f"Thought Recorded: {last_decision.get('content', '')[:50]}..."
            self.state.current_plan.pop(0)
            if not self.state.current_plan:
                self.state.current_plan.append(PlanStep(description="Decide next action based on recorded thought and overall goal."))
            self.state.needs_replan = False
        elif decision_type == "complete":
            action_successful = True
            self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing.", status="completed")]
            self.state.needs_replan = False
        else:
            action_successful = False
            if tool_name_executed != AGENT_TOOL_UPDATE_PLAN:
                current_step.status = ActionStatus.FAILED.value
                err_summary = self.state.last_action_summary or "Unknown agent error"
                current_step.result_summary = f"Agent/Tool Error: {err_summary[:100]}..."
                if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Re-evaluate due to agent error"):
                    self.state.current_plan.insert(1, PlanStep(description="Re-evaluate due to agent error or unclear decision."))
            self.state.needs_replan = True
        if action_successful:
            self.state.consecutive_error_count = 0
            if tool_name_executed and tool_name_executed not in self._INTERNAL_OR_META_TOOLS:
                self.state.successful_actions_since_reflection += 1.0
                self.state.successful_actions_since_consolidation += 1.0
                self.logger.debug(
                    f"Incr success R:{self.state.successful_actions_since_reflection:.1f}, C:{self.state.successful_actions_since_consolidation:.1f} after: {tool_name_executed}"
                )
            elif decision_type == "thought_process":
                self.state.successful_actions_since_reflection += 0.5
                self.state.successful_actions_since_consolidation += 0.5
                self.logger.debug(
                    f"Incr success R:{self.state.successful_actions_since_reflection:.1f}, C:{self.state.successful_actions_since_consolidation:.1f} after thought."
                )
        else:
            self.state.consecutive_error_count += 1
            self.logger.warning(f"Consecutive error count: {self.state.consecutive_error_count}")
            if self.state.successful_actions_since_reflection > 0:
                self.logger.info(f"Reset reflection counter (was {self.state.successful_actions_since_reflection:.1f}).")
                self.state.successful_actions_since_reflection = 0
        log_plan_msg = f"Plan updated heuristically. Steps: {len(self.state.current_plan)}. "
        if self.state.current_plan:
            next_step = self.state.current_plan[0]
            depends_str = f"Depends: {[_fmt_id(d) for d in next_step.depends_on]}" if next_step.depends_on else "Depends: None"
            log_plan_msg += f"Next: '{next_step.description[:60]}...' (Status: {next_step.status}, {depends_str})"
        else:
            log_plan_msg += "Plan empty."
        self.logger.info(f"📋 {log_plan_msg}")

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
        momentum_bias = 0  # Initialize momentum_bias
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
        if not self.state.workflow_id or not self.state.context_id or self._shutdown_event.is_set():
            return
        tasks_to_run: List[Tuple[str, Dict]] = []
        trigger_reasons: List[str] = []
        reflection_tool_available = self._find_tool_server(TOOL_REFLECTION) is not None
        consolidation_tool_available = self._find_tool_server(TOOL_CONSOLIDATION) is not None
        optimize_wm_tool_available = self._find_tool_server(TOOL_OPTIMIZE_WM) is not None
        auto_focus_tool_available = self._find_tool_server(TOOL_AUTO_FOCUS) is not None
        promotion_query_tool_available = self._find_tool_server(TOOL_QUERY_MEMORIES) is not None
        stats_tool_available = self._find_tool_server(TOOL_COMPUTE_STATS) is not None
        maintenance_tool_available = self._find_tool_server(TOOL_DELETE_EXPIRED_MEMORIES) is not None
        self.state.loops_since_stats_adaptation += 1
        if self.state.loops_since_stats_adaptation >= STATS_ADAPTATION_INTERVAL:
            if stats_tool_available:
                trigger_reasons.append("StatsInterval")
                try:
                    stats_result = await self._execute_tool_call_internal(
                        TOOL_COMPUTE_STATS, {"workflow_id": self.state.workflow_id}, record_action=False
                    )
                    if stats_result.get("success"):
                        self._adapt_thresholds(stats_result)
                    episodic_count = stats_result.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
                    if (
                        episodic_count > (self.state.current_consolidation_threshold * 2.0)
                        and consolidation_tool_available
                        and not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run)
                    ):
                        self.logger.info(f"High episodic ({episodic_count}), scheduling consolidation.")
                        tasks_to_run.append(
                            (
                                TOOL_CONSOLIDATION,
                                {
                                    "workflow_id": self.state.workflow_id,
                                    "consolidation_type": "summary",
                                    "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                                    "max_source_memories": self.consolidation_max_sources,
                                },
                            )
                        )
                        trigger_reasons.append(f"HighEpisodic({episodic_count})")
                        self.state.successful_actions_since_consolidation = 0
                    else:
                        self.logger.warning(f"Failed compute stats for adaptation: {stats_result.get('error')}")
                except Exception as e:
                    self.logger.error(f"Error during stats/adaptation: {e}", exc_info=False)
                finally:
                    self.state.loops_since_stats_adaptation = 0
            else:
                self.logger.warning(f"Skipping stats/adaptation: Tool {TOOL_COMPUTE_STATS} not available")
        needs_reflection = self.state.needs_replan or self.state.successful_actions_since_reflection >= self.state.current_reflection_threshold
        if needs_reflection:
            if reflection_tool_available and not any(task[0] == TOOL_REFLECTION for task in tasks_to_run):
                reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
                tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
                reason_str = (
                    f"Replan({self.state.needs_replan})"
                    if self.state.needs_replan
                    else f"SuccessCount({self.state.successful_actions_since_reflection:.1f}>={self.state.current_reflection_threshold})"
                )
                trigger_reasons.append(f"Reflect({reason_str})")
                self.state.successful_actions_since_reflection = 0
                self.state.reflection_cycle_index += 1
            else:
                self.logger.warning(f"Skipping reflection: Tool {TOOL_REFLECTION} unavailable")
                self.state.successful_actions_since_reflection = 0
        needs_consolidation = self.state.successful_actions_since_consolidation >= self.state.current_consolidation_threshold
        if needs_consolidation:
            if consolidation_tool_available and not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run):
                tasks_to_run.append(
                    (
                        TOOL_CONSOLIDATION,
                        {
                            "workflow_id": self.state.workflow_id,
                            "consolidation_type": "summary",
                            "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                            "max_source_memories": self.consolidation_max_sources,
                        },
                    )
                )
                trigger_reasons.append(
                    f"ConsolidateThreshold({self.state.successful_actions_since_consolidation:.1f}>={self.state.current_consolidation_threshold})"
                )
                self.state.successful_actions_since_consolidation = 0
            else:
                self.logger.warning(f"Skipping consolidation: Tool {TOOL_CONSOLIDATION} unavailable")
                self.state.successful_actions_since_consolidation = 0
        self.state.loops_since_optimization += 1
        if self.state.loops_since_optimization >= OPTIMIZATION_LOOP_INTERVAL:
            if optimize_wm_tool_available:
                tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.state.context_id}))
                trigger_reasons.append("OptimizeInterval")
            else:
                self.logger.warning(f"Skipping WM opt: Tool {TOOL_OPTIMIZE_WM} unavailable")
            if auto_focus_tool_available:
                tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.state.context_id}))
                trigger_reasons.append("FocusUpdateInterval")
            else:
                self.logger.warning(f"Skipping auto-focus: Tool {TOOL_AUTO_FOCUS} unavailable")
            self.state.loops_since_optimization = 0
        self.state.loops_since_promotion_check += 1
        if self.state.loops_since_promotion_check >= MEMORY_PROMOTION_LOOP_INTERVAL:
            if promotion_query_tool_available:
                tasks_to_run.append(("CHECK_PROMOTIONS", {}))
                trigger_reasons.append("PromotionInterval")
            else:
                self.logger.warning(f"Skipping promo check: Tool {TOOL_QUERY_MEMORIES} unavailable.")
            self.state.loops_since_promotion_check = 0
        self.state.loops_since_maintenance += 1
        if self.state.loops_since_maintenance >= MAINTENANCE_INTERVAL:
            if maintenance_tool_available:
                tasks_to_run.append((TOOL_DELETE_EXPIRED_MEMORIES, {}))
                trigger_reasons.append("MaintenanceInterval")
                self.state.loops_since_maintenance = 0
            else:
                self.logger.warning(f"Skipping maintenance: Tool {TOOL_DELETE_EXPIRED_MEMORIES} unavailable")
        if tasks_to_run:
            unique_reasons_str = ", ".join(sorted(set(trigger_reasons)))
            self.logger.info(f"🧠 Running {len(tasks_to_run)} periodic tasks (Triggers: {unique_reasons_str})...")
            tasks_to_run.sort(key=lambda x: 0 if x[0] == TOOL_DELETE_EXPIRED_MEMORIES else 1 if x[0] == TOOL_COMPUTE_STATS else 2)
            for tool_name, args in tasks_to_run:
                if self._shutdown_event.is_set():
                    self.logger.info("Shutdown during periodic tasks.")
                    break
                try:
                    if tool_name == "CHECK_PROMOTIONS":
                        await self._trigger_promotion_checks()
                        continue
                    self.logger.debug(f"Executing periodic UMS Tool: {tool_name} with args: {args}")
                    result_content = await self._execute_tool_call_internal(tool_name, args, record_action=False)
                    if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and result_content.get("success"):
                        feedback = ""
                        if tool_name == TOOL_REFLECTION:
                            feedback = result_content.get("content", "")
                        elif tool_name == TOOL_CONSOLIDATION:
                            feedback = result_content.get("consolidated_content", "")
                        if not feedback and isinstance(result_content.get("data"), dict):
                            nested_data = result_content["data"]
                            feedback = nested_data.get("content" if tool_name == TOOL_REFLECTION else "consolidated_content", "")
                        if feedback:
                            feedback_summary = str(feedback).split("\n", 1)[0][:150]
                            self.state.last_meta_feedback = f"Feedback from {tool_name.split(':')[-1]}: {feedback_summary}..."
                            self.logger.info(f"Meta-feedback from UMS {tool_name}: {self.state.last_meta_feedback}")
                            self.state.needs_replan = True
                        else:
                            self.logger.debug(f"Periodic UMS task {tool_name} no feedback content. Result: {result_content}")
                except Exception as e:
                    self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)
                await asyncio.sleep(0.1)

    async def _trigger_promotion_checks(self):
        if not self.state.workflow_id:
            self.logger.debug("Skipping promo check: No active WF.")
            return
        self.logger.debug("Running periodic promotion check...")
        query_tool = TOOL_QUERY_MEMORIES
        candidate_ids = set()
        try:
            episodic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.EPISODIC.value,
                "sort_by": "last_accessed",
                "sort_order": "DESC",
                "limit": 5,
                "include_content": False,
            }
            episodic_res = await self._execute_tool_call_internal(query_tool, episodic_args, record_action=False)
            if episodic_res.get("success"):
                mems = episodic_res.get("memories", [])
                candidate_ids.update(m.get("memory_id") for m in mems if isinstance(m, dict) and m.get("memory_id"))
            semantic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.SEMANTIC.value,
                "sort_by": "last_accessed",
                "sort_order": "DESC",
                "limit": 5,
                "include_content": False,
            }
            semantic_res = await self._execute_tool_call_internal(query_tool, semantic_args, record_action=False)
            if semantic_res.get("success"):
                mems = semantic_res.get("memories", [])
                candidate_ids.update(
                    m.get("memory_id")
                    for m in mems
                    if isinstance(m, dict) and m.get("memory_id") and m.get("memory_type") in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]
                )
            if candidate_ids:
                self.logger.debug(f"Checking {len(candidate_ids)} memories for promo: {[_fmt_id(i) for i in candidate_ids]}")
            for mem_id in candidate_ids:
                self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id)
            else:
                self.logger.debug("No eligible memories for promo check.")
        except Exception as e:
            self.logger.error(f"Error during promo check query: {e}", exc_info=False)

    async def _gather_context(self) -> Dict[str, Any]:
        self.logger.info("🛰️ Gathering comprehensive context for LLM...")
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
        self.state.last_meta_feedback = None

        current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_cognitive_context_id = self.state.context_id  # Agent's current UMS cognitive_state.state_id
        current_plan_step_desc = self.state.current_plan[0].description if self.state.current_plan else DEFAULT_PLAN_STEP

        if not current_workflow_id:
            context_payload["status_message_from_agent"] = "No Active Workflow."
            self.logger.warning(context_payload["status_message_from_agent"])
            context_payload["ums_package_retrieval_status"] = "skipped_no_workflow"
            context_payload["processing_time_sec"] = time.time() - start_time
            return context_payload

        context_payload["workflow_id"] = current_workflow_id
        context_payload["cognitive_context_id_agent"] = current_cognitive_context_id

        # 1. Agent Assembles Its Goal Context using _fetch_goal_stack_from_ums
        agent_goal_context_block: Dict[str, Any] = {
            "retrieved_at": agent_retrieval_timestamp,
            "current_goal_details_from_ums": None,  # This will be the leaf goal object
            "goal_stack_summary_from_agent_state": [],  # This will be list of parent summaries
            "data_source_comment": "Goal stack and current goal details fetched from UMS by agent.",
        }
        if self.state.current_goal_id:
            full_ums_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
            if full_ums_stack:
                agent_goal_context_block["current_goal_details_from_ums"] = full_ums_stack[-1]  # Leaf is current
                # Create summary for prompt, applying show limit
                agent_goal_context_block["goal_stack_summary_from_agent_state"] = [
                    {"goal_id": _fmt_id(g.get("goal_id")), "description": (g.get("description") or "")[:150] + "...", "status": g.get("status")}
                    for g in full_ums_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]  # Show N items from root towards leaf
                ]
            else:
                err_msg = f"Agent: Failed to fetch UMS goal stack for current goal {_fmt_id(self.state.current_goal_id)}."
                context_payload["errors_in_context_gathering"].append(err_msg)
                agent_goal_context_block["current_goal_details_from_ums"] = {
                    "error_fetching_details": err_msg,
                    "goal_id_attempted": self.state.current_goal_id,
                }
                self.logger.warning(err_msg)
        context_payload["agent_assembled_goal_context"] = agent_goal_context_block

        # 2. Call UMS Tool for Rich Context Package (excluding goals)
        ums_package_tool_name = TOOL_GET_RICH_CONTEXT_PACKAGE
        ums_package_data: Optional[Dict[str, Any]] = {
            "error_ums_package_tool_unavailable": f"Tool {ums_package_tool_name} unavailable."
        }  # Default error

        if self._find_tool_server(ums_package_tool_name):
            # Determine focal_memory_id_hint from the agent's current working memory state if available
            # The UMS `get_rich_context_package` might use its own `get_working_memory` call.
            # If agent maintains its own working memory list (e.g., `self.state.working_memory_ids`), use focal from there.
            # For now, assume UMS `get_rich_context_package` handles WM details based on `context_id`.
            # We can pass a hint if the agent has a strong idea from a previous cycle.
            focal_id_hint_for_ums = None  # Placeholder
            if context_payload.get("ums_context_package", {}).get("current_working_memory", {}).get("focal_memory_id"):  # If UMS already provided WM
                focal_id_hint_for_ums = context_payload["ums_context_package"]["current_working_memory"]["focal_memory_id"]
            elif isinstance(self.state.last_error_details, dict) and self.state.last_error_details.get(
                "focal_memory_id_from_last_wm"
            ):  # Fallback from previous error context
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
                "show_limits": {  # For UMS to use if it does truncation/summarization
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
                self.logger.debug(f"Agent: Calling UMS tool '{ums_package_tool_name}'...")
                raw_ums_response = await self._execute_tool_call_internal(ums_package_tool_name, ums_package_params, record_action=False)
                if raw_ums_response.get("success"):
                    ums_package_data = raw_ums_response.get("context_package", {})  # UMS tool returns under "context_package"
                    if not isinstance(ums_package_data, dict):
                        err_msg = f"Agent: UMS tool {ums_package_tool_name} returned invalid 'context_package'."
                        self.logger.error(err_msg)
                        context_payload["errors_in_context_gathering"].append(err_msg)
                        ums_package_data = {"error_ums_pkg_invalid": err_msg}
                    else:
                        self.logger.info("Agent: Successfully retrieved rich context package from UMS.")
                        ums_internal_errors = ums_package_data.get("errors")  # UMS tool should use "errors" key for its own errors
                        if ums_internal_errors and isinstance(ums_internal_errors, list):
                            context_payload["errors_in_context_gathering"].extend([f"UMS_PKG_ERR: {e}" for e in ums_internal_errors])
                        ums_package_data.pop("errors", None)  # Clean from context
                else:
                    err_msg = f"Agent: UMS rich context pkg retrieval failed: {raw_ums_response.get('error')}"
                    context_payload["errors_in_context_gathering"].append(err_msg)
                    self.logger.warning(err_msg)
                    ums_package_data = {"error_ums_pkg_retrieval": err_msg}
            except Exception as e:
                err_msg = f"Agent: Exception calling UMS for rich context pkg: {e}"
                self.logger.error(err_msg, exc_info=True)
                context_payload["errors_in_context_gathering"].append(err_msg)
                ums_package_data = {"error_ums_pkg_exception": err_msg}
        else:  # UMS tool not available
            err_msg = f"Agent: UMS tool '{ums_package_tool_name}' unavailable."
            self.logger.error(err_msg)
            context_payload["errors_in_context_gathering"].append(err_msg)
            # ums_package_data remains the default error set above

        context_payload["ums_context_package"] = ums_package_data

        # 3. Agent-Side Final Compression (if needed)
        # This logic remains similar, but acts on the combined context_payload
        # It uses self._estimate_tokens_anthropic and UMS TOOL_SUMMARIZE_TEXT
        ums_compression_info = context_payload.get("ums_context_package", {}).get("compression_summary")  # UMS tool uses "compression_summary"
        needs_agent_compression = True
        if ums_compression_info:
            self.logger.info("Agent: UMS provided compression. Checking if agent-side pass needed.")
            current_tokens = await self._estimate_tokens_anthropic(context_payload)
            if current_tokens <= CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD:
                needs_agent_compression = False
            else:
                self.logger.warning(f"Agent: Context still large ({current_tokens}) after UMS compression. Agent pass.")
        if needs_agent_compression:
            try:
                est_tokens = await self._estimate_tokens_anthropic(context_payload)
                if est_tokens > CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD:
                    self.logger.warning(f"Agent: Context ({est_tokens} tokens) > threshold. Agent-side summary compression.")
                    if self._find_tool_server(TOOL_SUMMARIZE_TEXT):
                        # Summarize the *entire* context_payload. This can be lossy.
                        # A more nuanced approach would pick specific large sub-sections from context_payload.
                        context_str_to_summarize = json.dumps(context_payload, default=str, ensure_ascii=False)
                        MAX_INPUT_SUMMARY = 45000  # Limit for summarizer input
                        if len(context_str_to_summarize) > MAX_INPUT_SUMMARY:
                            self.logger.warning("Agent: Full context for agent summary too long. Truncating input.")
                            context_str_to_summarize = context_str_to_summarize[-MAX_INPUT_SUMMARY:]

                        summary_res = await self._execute_tool_call_internal(
                            TOOL_SUMMARIZE_TEXT,
                            {
                                "text_to_summarize": context_str_to_summarize,
                                "target_tokens": CONTEXT_COMPRESSION_TARGET_TOKENS,
                                "context_type": "full_agent_context_payload_agent_pass",
                                "workflow_id": current_workflow_id,
                                "record_summary": False,
                            },
                            record_action=False,
                        )
                        if summary_res.get("success") and summary_res.get("summary"):
                            context_payload["agent_final_compression_summary"] = {
                                "summary_content": summary_res["summary"],
                                "original_tokens_before_agent_pass": est_tokens,
                                "retrieved_at": agent_retrieval_timestamp,
                            }
                            self.logger.info("Agent: Agent-side full context compression applied.")
                            # The _construct_agent_prompt method needs to be aware of 'agent_final_compression_summary'
                            # and potentially use it as the primary context if present, or merge it smartly.
                        else:
                            err_msg = f"Agent: Agent-side full compression failed: {summary_res.get('error')}"
                            context_payload["errors_in_context_gathering"].append(err_msg)
                            self.logger.warning(err_msg)
                    else:
                        self.logger.warning("Agent: Agent-side compression needed but UMS summarize_text tool unavailable.")
            except Exception as e:
                err_msg = f"Agent: Error during agent-side context compression: {e}"
                context_payload["errors_in_context_gathering"].append(err_msg)
                self.logger.error(err_msg, exc_info=True)

        final_errors = len(context_payload.get("errors_in_context_gathering", []))
        context_payload["status_message_from_agent"] = "Context ready" if not final_errors else f"Context ready with {final_errors} errors"
        self.logger.info(f"Agent: Context gathering complete. Errors: {final_errors}. Time: {(time.time() - start_time):.3f}s")
        if final_errors > 0:
            self.logger.debug(f"Agent: Errors during context gathering: {context_payload.get('errors_in_context_gathering')}")
        return context_payload


    async def prepare_next_turn_data(self, overall_goal: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepares all data needed by MCPClient to make the next LLM call.
        This includes gathering context, constructing prompt messages, and providing tool schemas.

        Args:
            overall_goal: The overarching goal for the agent.

        Returns:
            A tuple containing:
            - prompt_messages_for_llm (List[Dict]): The messages to send to the LLM.
            - tool_schemas_for_llm (List[Dict]): The tool schemas for the LLM.
            - agent_context_snapshot (Dict): The full context dictionary that was used to build the prompt.
                                            (MCPClient might use this for logging or its own UI).
        """
        self.logger.info(
            f"AML: Preparing next turn data for Loop {self.state.current_loop} (WF: {_fmt_id(self.state.workflow_id)}, Goal: {_fmt_id(self.state.current_goal_id)})"
        )

        # 1. Periodic Cognitive Tasks (Run before gathering context for the turn)
        #    This ensures meta-cognitive outputs can influence the current turn.
        try:
            await self._run_periodic_tasks()
        except Exception as e:
            self.logger.error(f"AML: Error during periodic tasks: {e}", exc_info=True)
        if self._shutdown_event.is_set():
            # If shutdown, return empty/signal to stop
            raise asyncio.CancelledError("AML shutdown during periodic tasks in prepare_next_turn_data")


        # 2. Gather Context
        agent_context_snapshot = await self._gather_context()
        if not self.state.workflow_id: # Check if workflow ended
            self.logger.info("AML: No active workflow after context gathering. Signaling to stop.")
            raise asyncio.CancelledError("AML: No active workflow, cannot prepare turn.") # Or a custom exception

        # 3. Construct Prompt
        if not self.state.needs_replan: # Clear previous error if not actively replanning for it
            self.state.last_error_details = None
        prompt_messages_for_llm = self._construct_agent_prompt(overall_goal, agent_context_snapshot)

        # Tool schemas are already loaded in self.tool_schemas during agent.initialize()
        return prompt_messages_for_llm, self.tool_schemas, agent_context_snapshot


    async def execute_llm_decision(
        self,
        llm_decision: Dict[str, Any],
        # overall_goal: str # May not be needed here if decisions are self-contained
    ) -> bool:
        """
        Executes the decision received from the LLM (via MCPClient).
        This includes calling tools, recording thoughts, updating plans, etc.
        Handles error limits and state saving.

        Args:
            llm_decision: The decision dictionary from the LLM.
                          Expected format:
                          {"decision": "call_tool", "tool_name": "...", "arguments": {...}}
                          OR {"decision": "thought_process", "content": "..."}
                          OR {"decision": "complete", "summary": "..."}
                          OR {"decision": "plan_update", "updated_plan_steps": List[PlanStep]}
                          OR {"decision": "error", "message": "..."}
            overall_goal: The overarching goal (might be useful for some heuristic fallbacks).

        Returns:
            bool: True if the agent should continue to the next loop,
                  False if the agent should stop (e.g., goal achieved, max errors).
        """
        self.logger.info(
            f"AML: Executing LLM Decision for Loop {self.state.current_loop} (WF: {_fmt_id(self.state.workflow_id)}, Goal: {_fmt_id(self.state.current_goal_id)})"
        )
        self.logger.debug(f"AML: Received LLM Decision: {str(llm_decision)[:300]}")

        tool_result_content: Optional[Dict[str, Any]] = None
        llm_proposed_plan_steps: Optional[List[PlanStep]] = llm_decision.get("updated_plan_steps")
        decision_type = llm_decision.get("decision")

        if decision_type == "call_tool":
            tool_name = llm_decision.get("tool_name")
            arguments = llm_decision.get("arguments", {})
            if not self.state.current_plan:
                self.logger.error("AML: Plan empty before tool call! Forcing replan.")
                self.state.needs_replan = True; self.state.last_error_details = {"error": "Plan empty.", "type": "PlanValidationError"}
            elif not self.state.current_plan[0].description:
                self.logger.error(f"AML: Invalid plan step (no desc)! Step ID: {self.state.current_plan[0].id}. Forcing replan.")
                self.state.needs_replan = True; self.state.last_error_details = {"error": "Current plan step invalid.", "type": "PlanValidationError", "step_id": self.state.current_plan[0].id}
            elif tool_name:
                current_step_deps = self.state.current_plan[0].depends_on if self.state.current_plan else []
                tool_result_content = await self._execute_tool_call_internal(
                    tool_name, arguments, record_action=True, planned_dependencies=current_step_deps
                )
            else:
                self.logger.error("AML: LLM tool call decision missing tool name."); self.state.last_error_details = {"error": "LLM missing tool name.", "type": "LLMOutputError"}; tool_result_content = {"success": False, "error": "Missing tool name from LLM."}
        elif decision_type == "thought_process":
            thought_content = llm_decision.get("content")
            if thought_content:
                tool_result_content = await self._execute_tool_call_internal(
                    TOOL_RECORD_THOUGHT,
                    {"content": thought_content, "thought_type": ThoughtType.INFERENCE.value},
                    record_action=False
                )
            else:
                self.logger.warning("AML: LLM 'thought_process' decision but no content."); tool_result_content = {"success": False, "error": "Missing thought content from LLM."}
        elif decision_type == "complete":
            self.logger.info(f"AML: LLM signaled OVERALL goal completion: {llm_decision.get('summary')}")
            root_goal_to_mark = None
            if self.state.goal_stack:
                for g_dict in self.state.goal_stack:
                    if not g_dict.get("parent_goal_id"): root_goal_to_mark = g_dict.get("goal_id"); break
            if root_goal_to_mark and self._find_tool_server(TOOL_UPDATE_GOAL_STATUS):
                self.logger.info(f"AML: Marking UMS root goal {_fmt_id(root_goal_to_mark)} as completed based on LLM.")
                await self._execute_tool_call_internal(
                    TOOL_UPDATE_GOAL_STATUS,
                    {"goal_id": root_goal_to_mark, "status": GoalStatus.COMPLETED.value, "reason": "LLM signaled overall goal completion."},
                    record_action=False
                ) # Side-effect handler will set goal_achieved_flag
            else:
                self.logger.warning(f"AML: LLM signaled overall completion, but couldn't mark UMS root goal automatically. Setting flag manually.")
                self.state.goal_achieved_flag = True # Set flag manually if UMS tool unavailable for root
            # No further processing if overall goal achieved
            if self.state.goal_achieved_flag:
                await self._save_agent_state()
                return False # Signal to stop the loop
        elif decision_type == "plan_update":
            self.logger.info("AML: LLM proposed structured plan update.")
            # llm_proposed_plan_steps is already set from the decision
        elif decision_type == "error": # Error from MCPClient's LLM processing
            self.logger.error(f"AML: LLM decision processing error from MCPClient: {llm_decision.get('message')}")
            self.state.last_action_summary = f"LLM Decision Error: {llm_decision.get('message', 'Unknown')[:100]}"
            if not self.state.last_error_details:
                self.state.last_error_details = {"error": llm_decision.get('message'), "type": "LLMError"}
            self.state.needs_replan = True
        else: # Unexpected decision type
            self.logger.error(f"AML: Unexpected decision type from MCPClient: {decision_type}")
            self.state.last_action_summary = f"Agent Error: Unexpected decision '{decision_type}'"
            self.state.last_error_details = {"error": f"Unexpected decision type '{decision_type}'", "type": "AgentError"}
            self.state.needs_replan = True

        # --- Apply Plan Updates (if any) ---
        if llm_proposed_plan_steps:
            try:
                if self._detect_plan_cycle(llm_proposed_plan_steps):
                    err_msg = "AML: LLM-proposed plan has cycle. Applying heuristic update instead."
                    self.logger.error(err_msg)
                    self.state.last_error_details = {"error": err_msg, "type": "PlanValidationError", "proposed_plan": [p.model_dump() for p in llm_proposed_plan_steps]}
                    self.state.needs_replan = True
                    await self._apply_heuristic_plan_update(llm_decision, tool_result_content)
                else:
                    self.state.current_plan = llm_proposed_plan_steps
                    self.state.needs_replan = False
                    self.logger.info(f"AML: Applied LLM-proposed plan update ({len(llm_proposed_plan_steps)} steps).")
                    self.state.last_error_details = None; self.state.consecutive_error_count = 0
            except Exception as plan_apply_err:
                self.logger.error(f"AML: Error applying LLM proposed plan: {plan_apply_err}. Fallback.", exc_info=True)
                self.state.last_error_details = {"error": f"Failed apply LLM plan: {plan_apply_err}", "type": "PlanUpdateError"}
                self.state.needs_replan = True
                await self._apply_heuristic_plan_update(llm_decision, tool_result_content)
        elif decision_type != "call_tool" or llm_decision.get("tool_name") != AGENT_TOOL_UPDATE_PLAN:
            # Apply heuristic if LLM didn't use AGENT_TOOL_UPDATE_PLAN or provide a structured plan
            await self._apply_heuristic_plan_update(llm_decision, tool_result_content)

        # --- Check Error Limit & Save State ---
        if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
            self.logger.critical(f"AML: Max consecutive errors ({MAX_CONSECUTIVE_ERRORS}) reached. Signaling stop.")
            if self.state.workflow_id and self._find_tool_server(TOOL_UPDATE_WORKFLOW_STATUS):
                await self._execute_tool_call_internal(
                    TOOL_UPDATE_WORKFLOW_STATUS,
                    {"workflow_id": self.state.workflow_id, "status": WorkflowStatus.FAILED.value, "completion_message": f"Aborted after {MAX_CONSECUTIVE_ERRORS} errors."},
                    record_action=False
                )
            await self._save_agent_state()
            return False # Signal to stop

        await self._save_agent_state()

        # Check if the loop should continue
        if self.state.goal_achieved_flag or self._shutdown_event.is_set() or not self.state.workflow_id:
            return False # Signal to stop

        return True # Signal to continue


    async def run_main_loop(self, initial_goal: str, max_loops: int = 100):
        """
        The main execution loop that MCPClient will call repeatedly
        if it's running the agent in a self-driving mode.
        """
        self.logger.info(f"AgentMasterLoop.run_main_loop called. Goal: '{initial_goal}'. Max loops: {max_loops}")

        # --- Initialization of Workflow and Root Goal (if first run for this agent instance) ---
        if not self.state.workflow_id:
            self.logger.info("AML: No active workflow. Creating initial workflow via UMS.")
            wf_create_args = {
                "title": f"Agent Task: {initial_goal[:50]}...",
                "goal": initial_goal,
                "description": f"Agent workflow initiated at {datetime.now(timezone.utc).isoformat()} to achieve: {initial_goal}",
                "tags": ["agent_run", AGENT_NAME.lower()]
            }
            wf_create_result = await self._execute_tool_call_internal(
                TOOL_CREATE_WORKFLOW, wf_create_args, record_action=False
            )
            if not wf_create_result.get("success") or not self.state.workflow_id:
                self.logger.critical(f"AML: Failed to create initial UMS workflow: {wf_create_result.get('error')}. Aborting.")
                return # Cannot proceed without a workflow
            if not self.state.current_goal_id:
                self.logger.critical("AML: UMS Workflow created, but UMS root goal ID was not set. Aborting.")
                return
            self.logger.info(f"AML: Initial workflow {_fmt_id(self.state.workflow_id)} and root goal {_fmt_id(self.state.current_goal_id)} established.")
        elif not self.state.current_thought_chain_id:
            await self._set_default_thought_chain_id()
        elif self.state.goal_stack and not self.state.current_goal_id:
            self.state.current_goal_id = self.state.goal_stack[-1].get("goal_id")
            self.logger.info(f"AML: Set current_goal_id from loaded stack: {_fmt_id(self.state.current_goal_id)}")

        # --- Main Agent Logic Loop (driven by MCPClient now) ---
        # This loop represents one "turn" or cycle of the agent's independent operation.
        # MCPClient will call this method repeatedly.

        if self.state.current_loop >= max_loops:
            self.logger.warning(f"AML: Agent loop reached max iterations ({max_loops}). Stopping.")
            return # MCPClient should stop calling if this returns (or if it tracks loops)

        if self.state.goal_achieved_flag:
            self.logger.info("AML: Goal previously achieved. Loop will not run further.")
            return

        if self._shutdown_event.is_set():
            self.logger.info("AML: Shutdown signaled. Loop will not run.")
            return

        self.state.current_loop += 1 # Increment loop counter *managed by agent*

        # 1. Prepare data for LLM
        try:
            prompt_messages, tool_schemas, agent_context = await self.prepare_next_turn_data(initial_goal)
        except asyncio.CancelledError:
            self.logger.info("AML: prepare_next_turn_data was cancelled (likely shutdown or no active workflow).")
            return # Stop this turn
        except Exception as e:
            self.logger.error(f"AML: Error in prepare_next_turn_data: {e}", exc_info=True)
            # Potentially set error state and allow MCPClient to decide if it retries or stops agent
            self.state.last_error_details = {"error": f"Context/Prompt prep error: {e}", "type": "AgentError"}
            return # Stop this turn

        # 2. MCPClient makes the LLM call and gets a decision
        # This step is now OUTSIDE AgentMasterLoop.run_main_loop.
        # MCPClient will call:
        #   llm_decision = await self.mcp_client.process_agent_llm_turn(prompt_messages, tool_schemas, self.agent_llm_model)

        # 3. AgentMasterLoop executes the decision (this part is called by MCPClient after it gets LLM decision)
        # For now, this method `run_main_loop` will effectively *return* the necessary data
        # to `MCPClient` so that `MCPClient` can make the LLM call.
        # The result of the LLM call would then be passed back to a different method like `execute_llm_decision`.

        # So, `run_main_loop` should now probably be split or its responsibility changed.
        # Let's assume MCPClient calls:
        #   `prompt_messages, tool_schemas, agent_context = await agent_loop.prepare_next_turn_data(initial_goal)`
        # Then MCPClient does its LLM call.
        # Then MCPClient calls:
        #   `should_continue = await agent_loop.execute_llm_decision(llm_decision)`
        # And MCPClient uses `should_continue` to decide whether to call `prepare_next_turn_data` again.

        # For the purpose of this refactoring, `run_main_loop` will just prepare and return.
        self.logger.info(f"AML: Data prepared for MCPClient to make LLM call for loop {self.state.current_loop}.")
        # Return the necessary components for MCPClient
        # MCPClient should also know if the agent thinks it should stop (e.g. goal achieved)
        if self.state.goal_achieved_flag or self._shutdown_event.is_set() or not self.state.workflow_id:
            return None # Signal to MCPClient that agent loop should stop.

        return {"prompt_messages": prompt_messages, "tool_schemas": tool_schemas, "agent_context": agent_context}
