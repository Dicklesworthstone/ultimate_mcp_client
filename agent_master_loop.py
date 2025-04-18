"""
Supercharged Agent Master Loop – v4.0 (Phase 1 - Step 1 Integration)
======================================================================
This revision integrates the enhanced context gathering logic (proactive,
procedural, links) back into the Agent Master Loop v3.3.5 structure.

Phase 1, Step 1 Changes:
  - Restored proactive goal-relevant memory search in `_gather_context`.
  - Restored procedural memory search in `_gather_context`.
  - Restored contextual link traversal in `_gather_context`.
  - Added detailed logging for context gathering components.
  - Ensured context truncation happens *after* all components are gathered.
  - Preserved all other v3.3.5 code, docstrings, and comments.

────────────────────────────────────────────────────────────────────────────
"""

# --------------------------------------------------------------------------
# Python std‑lib & third‑party imports
# --------------------------------------------------------------------------
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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
from anthropic import AsyncAnthropic  # ← correct import path (Fix #1)
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------------------------------
# MCP‑client import & logger bootstrap
# --------------------------------------------------------------------------
try:
    # Note: ArtifactType and WorkflowStatus might not be explicitly used
    # in this file but were imported in the original v3.3 - keeping for reference
    # if other parts of the system rely on them being imported here.
    from mcp_client import (  # type: ignore
        ActionStatus,
        ActionType,
        ArtifactType,  # noqa: F401 Keep import if original had it
        LinkType,
        MCPClient,
        MemoryLevel,
        MemoryType,  # noqa: F401 Keep import if original had it
        MemoryUtils,
        ThoughtType,
        ToolError,
        ToolInputError,
        WorkflowStatus,  # noqa: F401 Keep import if original had it
    )

    MCP_CLIENT_AVAILABLE = True
    log = logging.getLogger("AgentMasterLoop")
    if not logging.root.handlers and not log.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        log = logging.getLogger("AgentMasterLoop")
        log.warning("MCPClient did not configure logger – falling back.")
    log.info("Successfully imported MCPClient and required components.")
except ImportError as import_err:
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
AGENT_STATE_FILE = "agent_loop_state_v4.0_p1s1.json" # Versioned state file
AGENT_NAME = "EidenticEngine4.0-P1S1" # Versioned agent name
MASTER_LEVEL_AGENT_LLM_MODEL_STRING = "claude-3-7-sonnet-20250219" # Use appropriate model

# ---------------- meta‑cognition thresholds ----------------
BASE_REFLECTION_THRESHOLD = int(os.environ.get("BASE_REFLECTION_THRESHOLD", "7"))
BASE_CONSOLIDATION_THRESHOLD = int(os.environ.get("BASE_CONSOLIDATION_THRESHOLD", "12"))
MIN_REFLECTION_THRESHOLD = 3
MAX_REFLECTION_THRESHOLD = 15
MIN_CONSOLIDATION_THRESHOLD = 5
MAX_CONSOLIDATION_THRESHOLD = 25

# ---------------- interval constants ----------------
OPTIMIZATION_LOOP_INTERVAL = int(os.environ.get("OPTIMIZATION_INTERVAL", "8"))
MEMORY_PROMOTION_LOOP_INTERVAL = int(os.environ.get("PROMOTION_INTERVAL", "15"))
STATS_ADAPTATION_INTERVAL = int(os.environ.get("STATS_ADAPTATION_INTERVAL", "10"))
MAINTENANCE_INTERVAL = int(os.environ.get("MAINTENANCE_INTERVAL", "50"))

# ---------------- context / token sizing ----------------
AUTO_LINKING_DELAY_SECS: Tuple[float, float] = (1.5, 3.0)
DEFAULT_PLAN_STEP = "Assess goal, gather context, formulate initial plan."

CONTEXT_RECENT_ACTIONS = 7
CONTEXT_IMPORTANT_MEMORIES = 5
CONTEXT_KEY_THOUGHTS = 5
CONTEXT_PROCEDURAL_MEMORIES = 2 # Limit for context size
CONTEXT_PROACTIVE_MEMORIES = 3 # Limit for context size
CONTEXT_WORKING_MEMORY_LIMIT = 10 # Limit for context size
CONTEXT_LINK_TRAVERSAL_LIMIT = 3 # Limit links shown in context

CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD = 15_000
CONTEXT_COMPRESSION_TARGET_TOKENS = 5_000

MAX_CONSECUTIVE_ERRORS = 3

# ---------------- unified‑memory tool constants ----------------
TOOL_GET_WORKFLOW_DETAILS = "unified_memory:get_workflow_details"
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context"
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
TOOL_GENERATE_REPORT = "unified_memory:generate_workflow_report"
TOOL_SUMMARIZE_TEXT = "unified_memory:summarize_text"
# --- New tool name constant from v3.3.5 ---
AGENT_TOOL_UPDATE_PLAN = "agent:update_plan"

# ==========================================================================
# Utility helpers
# ==========================================================================
# (Keep existing utility functions: _fmt_id, _utf8_safe_slice, _truncate_context)
def _fmt_id(val: Any, length: int = 8) -> str:
    """Return a short id string safe for logs."""
    s = str(val) if val is not None else "?"
    return s[:length] if len(s) >= length else s


def _utf8_safe_slice(s: str, max_len: int) -> str:
    """Return a UTF‑8 boundary‑safe slice ≤ max_len bytes."""
    return s.encode("utf‑8")[:max_len].decode("utf‑8", "ignore")


def _truncate_context(context: Dict[str, Any], max_len: int = 25_000) -> str:
    """
    Structure‑aware context truncation with UTF‑8 safe fallback.

    1. Serialise full context – if within limit, return.
    2. Iteratively truncate known large lists to a few items + note.
    3. Remove lowest‑priority top‑level keys until size fits.
    4. Final fallback: utf‑8 safe byte slice of JSON.
    """
    # --- Preserve existing v3.3.5 implementation ---
    try:
        full = json.dumps(context, indent=2, default=str, ensure_ascii=False)
    except TypeError:
        # Handle potential non-serializable types gracefully before dumping
        context = json.loads(json.dumps(context, default=str))
        full = json.dumps(context, indent=2, default=str, ensure_ascii=False)

    if len(full) <= max_len:
        return full

    log.debug(f"Context length {len(full)} exceeds max {max_len}. Applying structured truncation.")
    ctx_copy = copy.deepcopy(context)
    ctx_copy["_truncation_applied"] = "structure‑aware"
    original_length = len(full)

    # Priority lists for truncation/removal
    # Order matters: Truncate lists first, then remove less critical keys
    list_paths_to_truncate = [ # (parent_key or None, key_of_list, items_to_keep)
        ("core_context", "recent_actions", 3),
        ("core_context", "important_memories", 3),
        ("core_context", "key_thoughts", 3),
        (None, "proactive_memories", 2),
        (None, "current_working_memory", 5), # Keep more working memory items
        (None, "relevant_procedures", 1),
    ]
    keys_to_remove_low_priority = [
        "relevant_procedures",
        "proactive_memories",
        "contextual_links", # Remove link summary if still too large
        "key_thoughts", # Remove thoughts from core context if needed
        "important_memories", # Remove memories from core context
        "recent_actions", # Remove actions from core context last before removing core_context itself
        "core_context", # Remove entire core context section as last resort
    ]

    # 1. Truncate lists
    for parent, key, keep_count in list_paths_to_truncate:
        try:
            container = ctx_copy
            if parent:
                if parent not in container or not isinstance(container[parent], dict):
                    continue
                container = container[parent]

            if key in container and isinstance(container[key], list) and len(container[key]) > keep_count:
                original_count = len(container[key])
                note = {"truncated_note": f"{original_count - keep_count} items omitted from '{key}'"}
                container[key] = container[key][:keep_count] + [note]
                log.debug(f"Truncated list '{key}' to {keep_count} items.")

            serial = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial) <= max_len:
                log.info(f"Context truncated successfully after list reduction (Length: {len(serial)}).")
                return serial
        except (KeyError, TypeError, IndexError) as e:
            log.warning(f"Error during list truncation for key '{key}': {e}")
            continue

    # 2. Remove low-priority keys
    for key_to_remove in keys_to_remove_low_priority:
        removed = False
        if key_to_remove in ctx_copy:
            ctx_copy.pop(key_to_remove)
            removed = True
            log.debug(f"Removed low-priority key '{key_to_remove}' for truncation.")
        elif "core_context" in ctx_copy and isinstance(ctx_copy["core_context"], dict) and key_to_remove in ctx_copy["core_context"]:
            # Handle keys potentially inside core_context
            ctx_copy["core_context"].pop(key_to_remove)
            removed = True
            log.debug(f"Removed low-priority key '{key_to_remove}' from core_context for truncation.")

        if removed:
            serial = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial) <= max_len:
                log.info(f"Context truncated successfully after key removal (Length: {len(serial)}).")
                return serial

    # 3. Ultimate fallback: UTF-8 safe byte slice
    log.warning(f"Structured truncation insufficient (Length still {len(serial)}). Applying final byte-slice.")
    clipped_json_str = _utf8_safe_slice(full, max_len - 50) # Leave room for closing chars/note
    # Attempt to make it valid JSON-like structure again (might fail)
    try:
        # Find the last complete JSON structure element (object or array)
        last_brace = clipped_json_str.rfind('}')
        last_bracket = clipped_json_str.rfind(']')
        cutoff = max(last_brace, last_bracket)
        if cutoff > 0:
            final_str = clipped_json_str[:cutoff+1] + '\n// ... (CONTEXT TRUNCATED BY BYTE LIMIT) ...\n}' # Add truncation note and try to close
        else:
            final_str = clipped_json_str + '... (CONTEXT TRUNCATED)'
    except Exception:
        final_str = clipped_json_str + '... (CONTEXT TRUNCATED)'

    log.error(f"Context severely truncated from {original_length} to {len(final_str)} bytes.")
    return final_str


# ==========================================================================
# Dataclass & pydantic models
# ==========================================================================
# (Keep existing PlanStep and AgentState dataclasses from v3.3.5)
class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: f"step-{MemoryUtils.generate_id()[:8]}")
    description: str
    status: str = Field(
        default="planned",
        description="Status: planned, in_progress, completed, failed, skipped",
    )
    depends_on: List[str] = Field(default_factory=list)
    assigned_tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    is_parallel_group: Optional[str] = None  # optional parallel‑exec tag


def _default_tool_stats() -> Dict[str, Dict[str, Union[int, float]]]:
    return defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0})


@dataclass
class AgentState:
    """
    All persisted runtime state (see Phase‑4 commentary).

    `context_id` mirrors `workflow_id` by default but *may* diverge
    in future fine‑grained focus contexts.
    """

    # --- workflow stack ---
    workflow_id: Optional[str] = None
    context_id: Optional[str] = None
    workflow_stack: List[str] = field(default_factory=list)

    # --- planning & reasoning ---
    current_plan: List[PlanStep] = field(
        default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)]
    )
    current_sub_goal_id: Optional[str] = None
    current_thought_chain_id: Optional[str] = None
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False

    # --- error/replan ---
    consecutive_error_count: int = 0
    needs_replan: bool = False
    last_error_details: Optional[Dict[str, Any]] = None

    # --- meta‑cognition metrics ---
    successful_actions_since_reflection: float = 0.0
    successful_actions_since_consolidation: float = 0.0
    loops_since_optimization: int = 0
    loops_since_promotion_check: int = 0
    loops_since_stats_adaptation: int = 0
    loops_since_maintenance: int = 0
    reflection_cycle_index: int = 0
    last_meta_feedback: Optional[str] = None

    # adaptive thresholds (dynamic)
    current_reflection_threshold: int = BASE_REFLECTION_THRESHOLD
    current_consolidation_threshold: int = BASE_CONSOLIDATION_THRESHOLD

    # tool statistics
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(
        default_factory=_default_tool_stats
    )

    # background tasks (transient)
    background_tasks: Set[asyncio.Task] = field(
        default_factory=set, init=False, repr=False
    )
# =====================================================================
# Agent Master Loop
# =====================================================================
class AgentMasterLoop:
    """Agent Master Loop Orchestrator - Integrating Rich Context"""

    # internal/meta tool registry (updated for Fix #14)
    _INTERNAL_OR_META_TOOLS: Set[str] = {
        # meta/logging about actions
        TOOL_RECORD_ACTION_START,
        TOOL_RECORD_ACTION_COMPLETION,
        # retrieval / info only
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
        # dependency mgmt + link creation
        TOOL_ADD_ACTION_DEPENDENCY,
        TOOL_CREATE_LINK,
        # admin / utility
        TOOL_LIST_WORKFLOWS,
        TOOL_COMPUTE_STATS,
        TOOL_SUMMARIZE_TEXT,
        TOOL_GET_WORKFLOW_DETAILS, # Added to internal list
        # periodic cognition
        TOOL_OPTIMIZE_WM,
        TOOL_AUTO_FOCUS,
        TOOL_PROMOTE_MEM,
        TOOL_REFLECTION,
        TOOL_CONSOLIDATION,
        TOOL_DELETE_EXPIRED_MEMORIES,
        # agent internal
        AGENT_TOOL_UPDATE_PLAN,
    }

    # --------------------------------------------------------------- ctor --
    def __init__(
        self, mcp_client_instance: MCPClient, agent_state_file: str = AGENT_STATE_FILE
    ):
        # (Keep existing v3.3.5 __init__ implementation)
        if not MCP_CLIENT_AVAILABLE:
            raise RuntimeError("MCPClient unavailable.")

        self.mcp_client = mcp_client_instance
        self.anthropic_client: AsyncAnthropic = self.mcp_client.anthropic
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        # consolidation configuration
        self.consolidation_memory_level = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources = 10

        # auto‑link params
        self.auto_linking_threshold = 0.7
        self.auto_linking_max_links = 3

        self.reflection_type_sequence = [
            "summary",
            "progress",
            "gaps",
            "strengths",
            "plan",
        ]

        if not self.anthropic_client:
            self.logger.critical("Anthropic client not provided.")
            raise ValueError("Anthropic client required.")

        self.state = AgentState()
        self._shutdown_event = asyncio.Event()
        self._bg_tasks_lock = asyncio.Lock()
        self.tool_schemas: List[Dict[str, Any]] = []

    # ----------------------------------------------------------- shutdown --
    async def shutdown(self) -> None:
        """Graceful termination entry‑point (Fix #4)."""
        # (Keep existing v3.3.5 shutdown implementation)
        self.logger.info("Shutdown requested.")
        self._shutdown_event.set()
        await self._cleanup_background_tasks()
        await self._save_agent_state()
        self.logger.info("Agent loop shutdown complete.")

    # ----------------------------------------------------------- prompt --
    def _construct_agent_prompt(
        self, goal: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build system+user prompt (Phase‑1 + Fix‑#7 truncation)."""
        # (Keep existing v3.3.5 _construct_agent_prompt implementation -
        # this will be updated in Phase 1, Step 2)
        # ---------- system ----------
        system_blocks: List[str] = [
            f"You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System.",
            "",
            f"Overall Goal: {goal}",
            "",
            "Available Unified‑Memory Tools (callable via tool schema):",
        ]
        if not self.tool_schemas:
            system_blocks.append("- CRITICAL: No tools available.")
        else:
            for schema in self.tool_schemas:
                sanitized = schema["name"]
                original = self.mcp_client.server_manager.sanitized_to_original.get(
                    sanitized, sanitized
                )
                desc = schema.get("description", "No description.")
                system_blocks.append(
                    f"- `{sanitized}` (maps to `{original}`) – {desc}"
                )
        system_blocks.append("")
        system_blocks.append("Your process at each step:")
        system_blocks.append("1. Analyse current context & feedback.")
        system_blocks.append("2. Update or repair the plan.")
        system_blocks.append("3. Decide **one** action:")
        system_blocks.append("   • tool call")
        system_blocks.append("   • record thought")
        system_blocks.append("   • report 'Goal Achieved:'")
        system_blocks.append(
            f"4. Use `{AGENT_TOOL_UPDATE_PLAN}` *exclusively* when you need to update the plan."
        )
        system_prompt = "\n".join(system_blocks)

        # ---------- user ----------
        context_json = _truncate_context(context) # Use robust truncation
        user_blocks = [
            "Current Context:",
            "```json",
            context_json,
            "```",
            "",
            "Current Plan:",
            "```json",
            json.dumps(
                [step.model_dump(exclude_none=True) for step in self.state.current_plan], # Exclude None
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            f"Last Action Summary:\n{self.state.last_action_summary}\n",
        ]
        if self.state.last_error_details:
            user_blocks += [
                "**Last Error Details**:",
                "```json",
                json.dumps(self.state.last_error_details, indent=2, default=str), # Use default=str
                "```",
                "",
            ]
        if self.state.last_meta_feedback:
            user_blocks += [
                "**Meta‑Cognitive Feedback**:",
                self.state.last_meta_feedback,
                "",
            ]
        user_blocks += [
            f"Overall Goal: {goal}",
            "",
            "Instruction: analyse, reason step‑by‑step, then output **one** of:",
            "• a tool call (`tool_use` block);",
            f"• `{AGENT_TOOL_UPDATE_PLAN}` tool call to replace the plan;",
            "• a line starting with `Goal Achieved:`.",
        ]
        user_prompt = "\n".join(user_blocks)

        # Return structure for Anthropic API (system prompt + user prompt)
        # Note: Anthropic recommends placing system prompt outside the messages list
        # Adjust if your client library handles it differently.
        # Assuming messages list format here:
        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]

    # ---------------------------------------------------------- bg‑task utils --
    # (Keep existing v3.3.5 background task utility functions:
    # _background_task_done, _background_task_done_safe, _start_background_task,
    # _add_bg_task, _cleanup_background_tasks)
    def _background_task_done(self, task: asyncio.Task) -> None:
        """Callback for completed background tasks."""
        # Use create_task to avoid blocking if lock is held
        asyncio.create_task(self._background_task_done_safe(task))

    async def _background_task_done_safe(self, task: asyncio.Task) -> None:
        """Safely remove task and log exceptions."""
        async with self._bg_tasks_lock:
            self.state.background_tasks.discard(task)
        if task.cancelled():
            self.logger.debug(f"Background task {task.get_name()} was cancelled.")
            return
        exc = task.exception()
        if exc:
            self.logger.error(
                f"Background task {task.get_name()} failed:",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _start_background_task(self, coro_fn, *args, **kwargs) -> asyncio.Task:
        """
        Fire‑and‑forget helper. Captures current workflow snapshot.
        Ensures the task is added to the tracking set safely.
        """
        # Snapshot critical state needed by the background task
        snapshot_wf_id = self.state.workflow_id
        snapshot_ctx_id = self.state.context_id
        # Add other state vars if needed by specific background tasks

        async def _wrapper():
            # Pass snapshotted state to the coroutine
            await coro_fn(
                self,
                *args,
                workflow_id=snapshot_wf_id,
                context_id=snapshot_ctx_id,
                **kwargs,
            )

        # Create the task
        task = asyncio.create_task(_wrapper())

        # Safely add the task to the tracking set using the lock
        asyncio.create_task(self._add_bg_task(task))

        # Add the completion callback
        task.add_done_callback(self._background_task_done)
        self.logger.debug(f"Started background task: {task.get_name()} for WF {_fmt_id(snapshot_wf_id)}")
        return task

    async def _add_bg_task(self, task: asyncio.Task) -> None:
        """Safely add a task to the background task set."""
        async with self._bg_tasks_lock:
            self.state.background_tasks.add(task)

    async def _cleanup_background_tasks(self) -> None:
        """Cancels and awaits completion of tracked background tasks."""
        async with self._bg_tasks_lock:
            # Create a copy to iterate over while potentially modifying the set
            tasks_to_cleanup = list(self.state.background_tasks)

        if not tasks_to_cleanup:
            self.logger.debug("No background tasks to clean up.")
            return

        self.logger.info(f"Cleaning up {len(tasks_to_cleanup)} background tasks…")
        cancelled_tasks = []
        for t in tasks_to_cleanup:
            if not t.done():
                # Check if it's marked critical - skip cancellation if so?
                # Add a 'critical' attribute or check task name if needed.
                # For now, cancel all non-done tasks on cleanup.
                t.cancel()
                cancelled_tasks.append(t)

        # Wait for all tasks (including cancelled ones) to complete
        results = await asyncio.gather(*tasks_to_cleanup, return_exceptions=True)

        # Log results of cleanup
        for i, res in enumerate(results):
            task_name = tasks_to_cleanup[i].get_name()
            if isinstance(res, asyncio.CancelledError):
                self.logger.debug(f"Task {task_name} successfully cancelled.")
            elif isinstance(res, Exception):
                self.logger.error(f"Task {task_name} raised an exception during cleanup: {res}")
            # else: Task completed normally before/during cleanup

        # Clear the set under lock after gathering
        async with self._bg_tasks_lock:
            self.state.background_tasks.clear()
        self.logger.info("Background tasks cleanup finished.")


    # ------------------------------------------------------- token estimator --
    async def _estimate_tokens_anthropic(self, data: Any) -> int:
        """Robust token estimation (Fix #2)."""
        # (Keep existing v3.3.5 _estimate_tokens_anthropic implementation)
        if data is None:
            return 0
        try:
            if not self.anthropic_client:
                # Fallback if client is missing (shouldn't happen if init passed)
                raise RuntimeError("Anthropic client unavailable for token estimation")

            # Serialize complex data to JSON string for counting
            text_to_count = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)

            # Use the Anthropic client's count_tokens method
            # Note: The actual API might vary slightly depending on exact client version
            # Assuming a simple text counting method here. If it needs messages format:
            # token_count = await self.anthropic_client.count_tokens(messages=[{"role": "user", "content": text_to_count}])
            # Adjust based on your specific Anthropic SDK usage.
            # Using a hypothetical simple count method for this example:
            token_count = await self.anthropic_client.count_tokens(text_to_count)
            return int(token_count) # Ensure integer return

        except Exception as e:
            self.logger.warning(f"Token estimation via Anthropic API failed: {e}. Using fallback.")
            # Fallback rough heuristic: characters / 4
            try:
                text_representation = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
                return len(text_representation) // 4
            except Exception as fallback_e:
                self.logger.error(f"Token estimation fallback failed: {fallback_e}")
                return 0 # Return 0 if all estimation fails

    # --------------------------------------------------------------- retry util --
    async def _with_retries(
        self,
        coro_fun,
        *args,
        max_retries: int = 3,
        retry_exceptions: Tuple[type[BaseException], ...] = (Exception,), # Catch broader base exceptions
        retry_backoff: float = 2.0,
        jitter: Tuple[float, float] = (0.1, 0.5), # Add slight minimum jitter
        **kwargs,
    ):
        """
        Generic retry wrapper .

        • Retries `max_retries-1` times on listed exceptions.
        • Exponential backoff with optional jitter.
        """
        attempt = 0
        while True:
            try:
                return await coro_fun(*args, **kwargs)
            except retry_exceptions as e:
                attempt += 1
                if attempt >= max_retries:
                    self.logger.error(f"{coro_fun.__name__} failed after {max_retries} attempts. Last error: {e}")
                    raise # Re-raise the last exception after exhausting retries
                delay = (retry_backoff ** (attempt - 1)) + random.uniform(*jitter)
                self.logger.warning(
                    f"{coro_fun.__name__} failed ({type(e).__name__}: {str(e)[:100]}...); retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            # Separately catch CancelledError to prevent retrying on deliberate cancellation
            except asyncio.CancelledError:
                 self.logger.info(f"Coroutine {coro_fun.__name__} was cancelled during retry loop.")
                 raise # Propagate cancellation immediately

    # ---------------------------------------------------------------- state I/O --
    async def _save_agent_state(self) -> None:
        """Cross‑platform atomic JSON save with fsync (Fix #8)."""
        # (Keep existing v3.3.5 _save_agent_state implementation)
        state_dict = dataclasses.asdict(self.state)
        state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Ensure background_tasks (non-serializable) is removed
        state_dict.pop("background_tasks", None)
        # Convert defaultdict to regular dict for saving
        state_dict["tool_usage_stats"] = {
            k: dict(v) for k, v in self.state.tool_usage_stats.items()
        }
        # Convert PlanStep objects to dicts
        state_dict["current_plan"] = [
            step.model_dump(exclude_none=True) for step in self.state.current_plan # Use model_dump and exclude None
        ]

        try:
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            # Atomic save using temporary file
            tmp_file = self.agent_state_file.with_suffix(f".tmp_{os.getpid()}") # Process-specific tmp file
            async with aiofiles.open(tmp_file, "w", encoding='utf-8') as f: # Specify encoding
                await f.write(json.dumps(state_dict, indent=2, ensure_ascii=False)) # Ensure unicode is handled
                await f.flush()
                # Ensure data is written to disk (important for robustness)
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    # fsync might fail on some systems/filesystems, log warning but continue
                    self.logger.warning(f"os.fsync failed during state save: {e}")

            # Replace the old state file with the new one atomically
            os.replace(tmp_file, self.agent_state_file)
            self.logger.debug(f"State saved atomically → {self.agent_state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)
            # Attempt to clean up temp file if it exists
            if tmp_file.exists():
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass


    async def _load_agent_state(self) -> None:
        """Loads agent state from file, handling potential errors and missing keys."""
        # (Keep existing v3.3.5 _load_agent_state implementation, ensure thresholds init)
        if not self.agent_state_file.exists():
            # Initialize with defaults if no state file exists
            self.state = AgentState(
                 current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                 current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
            self.logger.info("No prior state file found. Starting fresh with default state.")
            return
        try:
            async with aiofiles.open(self.agent_state_file, "r", encoding='utf-8') as f: # Specify encoding
                data = json.loads(await f.read())

            kwargs: Dict[str, Any] = {}
            processed_keys = set()

            # Iterate through fields defined in the AgentState dataclass
            for fld in dataclasses.fields(AgentState):
                # Skip non-init fields like background_tasks
                if not fld.init:
                    continue

                name = fld.name
                processed_keys.add(name)

                if name in data:
                    value = data[name]
                    # Handle specific fields needing type conversion or validation
                    if name == "current_plan":
                        try:
                            # Validate and convert plan steps
                            if isinstance(value, list):
                                kwargs["current_plan"] = [PlanStep(**d) for d in value]
                            else:
                                raise TypeError("Saved plan is not a list")
                        except (ValidationError, TypeError) as e:
                            self.logger.warning(f"Plan reload failed: {e}. Resetting plan.")
                            kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    elif name == "tool_usage_stats":
                        # Reconstruct defaultdict structure
                        dd = _default_tool_stats()
                        if isinstance(value, dict):
                            for k, v_dict in value.items():
                                if isinstance(v_dict, dict):
                                    # Ensure required keys exist with correct types
                                    dd[k]["success"] = int(v_dict.get("success", 0))
                                    dd[k]["failure"] = int(v_dict.get("failure", 0))
                                    dd[k]["latency_ms_total"] = float(v_dict.get("latency_ms_total", 0.0))
                        kwargs["tool_usage_stats"] = dd
                    # Add handling for other complex types if necessary
                    # elif name == "some_other_complex_field":
                    #    kwargs[name] = process_complex_field(value)
                    else:
                        # Directly assign value if no special handling needed
                        kwargs[name] = value
                else:
                    # Field defined in AgentState but missing in saved data
                    self.logger.debug(f"Field '{name}' not found in saved state. Using default.")
                    # Use default factory or default value if defined
                    if fld.default_factory is not dataclasses.MISSING:
                        kwargs[name] = fld.default_factory()
                    elif fld.default is not dataclasses.MISSING:
                        kwargs[name] = fld.default
                    # Handle specific cases like thresholds if not using defaults directly
                    elif name == "current_reflection_threshold":
                        kwargs[name] = BASE_REFLECTION_THRESHOLD
                    elif name == "current_consolidation_threshold":
                        kwargs[name] = BASE_CONSOLIDATION_THRESHOLD
                    # else: field will be missing if no default and not in saved data

            # Warn about extra keys found in the file but not in the dataclass
            extra_keys = set(data.keys()) - processed_keys - {"timestamp"} # Exclude timestamp meta-key
            if extra_keys:
                self.logger.warning(f"Ignoring unknown keys found in state file: {extra_keys}")

            # Ensure mandatory fields have values (use defaults if somehow missed)
            # This mainly ensures thresholds are set if loading a very old state file
            if "current_reflection_threshold" not in kwargs:
                kwargs["current_reflection_threshold"] = BASE_REFLECTION_THRESHOLD
            if "current_consolidation_threshold" not in kwargs:
                kwargs["current_consolidation_threshold"] = BASE_CONSOLIDATION_THRESHOLD

            # Create the AgentState instance
            self.state = AgentState(**kwargs)
            self.logger.info(f"Loaded state from {self.agent_state_file}; current loop {self.state.current_loop}")

        except (json.JSONDecodeError, TypeError, FileNotFoundError) as e:
            self.logger.error(f"State load failed: {e}. Resetting to default state.", exc_info=True)
            # Reset to default state on load failure
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
        except Exception as e: # Catch any other unexpected errors during load
            self.logger.critical(f"Unexpected error loading state: {e}. Resetting to default state.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )


    # --------------------------------------------------- tool‑lookup helper --
    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Finds an active server providing the specified tool."""
        # (Keep existing v3.3.5 _find_tool_server implementation)
        if not self.mcp_client or not self.mcp_client.server_manager:
            self.logger.error("MCP Client or Server Manager not available for tool lookup.")
            return None

        sm = self.mcp_client.server_manager
        # Check registered tools first
        if tool_name in sm.tools:
            server_name = sm.tools[tool_name].server_name
            # Verify the server is currently connected and active
            if server_name in sm.active_sessions:
                self.logger.debug(f"Found tool '{tool_name}' on active server '{server_name}'.")
                return server_name
            else:
                self.logger.debug(f"Server '{server_name}' for tool '{tool_name}' is registered but not active.")
                return None # Found but not active

        # Handle core tools (assuming a server named "CORE" handles them)
        if tool_name.startswith("core:") and "CORE" in sm.active_sessions:
            self.logger.debug(f"Found core tool '{tool_name}' on active CORE server.")
            return "CORE"

        # Handle internal agent tool
        if tool_name == AGENT_TOOL_UPDATE_PLAN:
            self.logger.debug(f"Internal tool '{tool_name}' does not require a server.")
            return "AGENT_INTERNAL" # Return a special marker

        # Tool not found
        self.logger.debug(f"Tool '{tool_name}' not found on any active server.")
        return None


    # ------------------------------------------------------------ initialization --
    async def initialize(self) -> bool:
        """Initializes the agent loop, loads state, fetches tool schemas, and verifies workflow."""
        # (Keep existing v3.3.5 initialize implementation, but add TOOL_GET_WORKFLOW_DETAILS to essential tools list)
        self.logger.info("Initializing Agent loop …")
        await self._load_agent_state()

        # Ensure context_id matches workflow_id if context_id is missing after load
        if self.state.workflow_id and not self.state.context_id:
            self.state.context_id = self.state.workflow_id
            self.logger.info(f"Initialized context_id from loaded workflow_id: {_fmt_id(self.state.workflow_id)}")

        try:
            if not self.mcp_client.server_manager:
                self.logger.error("MCP Client server manager not initialized.")
                return False

            # Fetch all available tool schemas from the server manager
            all_tools = self.mcp_client.server_manager.format_tools_for_anthropic()

            # Inject schema for internal agent plan‑update tool (Fix #14)
            # Use Pydantic model to generate schema reliably
            plan_step_schema = PlanStep.model_json_schema()
            # Remove title if Pydantic adds it automatically
            plan_step_schema.pop('title', None)
            all_tools.append(
                {
                    "name": AGENT_TOOL_UPDATE_PLAN,
                    "description": "Replace the agent's current plan with a new list of plan steps. Use this for significant replanning or error recovery.",
                    "input_schema": { # Use 'input_schema' to match Anthropic format
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "array",
                                "description": "Complete new plan as a list of PlanStep objects.",
                                "items": plan_step_schema, # Reference the generated schema
                            }
                        },
                        "required": ["plan"],
                    },
                }
            )

            # Filter for unified_memory tools + agent internal tool
            self.tool_schemas = []
            loaded_tool_names = set()
            for sc in all_tools:
                # Correctly handle potential missing mapping for sanitized names
                original_name = self.mcp_client.server_manager.sanitized_to_original.get(sc["name"], sc["name"])
                if original_name.startswith("unified_memory:") or sc["name"] == AGENT_TOOL_UPDATE_PLAN:
                    self.tool_schemas.append(sc)
                    loaded_tool_names.add(original_name)

            self.logger.info(f"Loaded {len(self.tool_schemas)} relevant tool schemas: {loaded_tool_names}")

            # Verify essential tools are available
            essential = [
                TOOL_CREATE_WORKFLOW,
                TOOL_RECORD_ACTION_START,
                TOOL_RECORD_ACTION_COMPLETION,
                TOOL_RECORD_THOUGHT,
                TOOL_STORE_MEMORY,
                TOOL_GET_WORKING_MEMORY,
                TOOL_HYBRID_SEARCH, # Essential for rich context
                TOOL_GET_CONTEXT,   # Essential for context
                TOOL_REFLECTION,    # Essential for meta-cognition
                TOOL_CONSOLIDATION, # Essential for knowledge synthesis
                TOOL_GET_WORKFLOW_DETAILS, # Needed for setting default chain ID
            ]
            missing = [t for t in essential if not self._find_tool_server(t)]
            if missing:
                # Log as error because functionality will be impaired
                self.logger.error(f"Missing essential tools: {missing}. Agent functionality WILL BE impaired.")
                # Depending on strictness, could return False here

            # Check validity of loaded workflow state
            top_wf = (
                self.state.workflow_stack[-1] if self.state.workflow_stack else None
            ) or self.state.workflow_id
            if top_wf and not await self._check_workflow_exists(top_wf):
                self.logger.warning(
                    f"Stored workflow '{_fmt_id(top_wf)}' not found; resetting workflow-specific state."
                )
                # Preserve non-workflow specific state
                preserved_stats = self.state.tool_usage_stats
                pres_ref_thresh = self.state.current_reflection_threshold
                pres_con_thresh = self.state.current_consolidation_threshold
                # Reset state but keep stats and thresholds
                self.state = AgentState(
                    tool_usage_stats=preserved_stats,
                    current_reflection_threshold=pres_ref_thresh,
                    current_consolidation_threshold=pres_con_thresh
                )
                await self._save_agent_state() # Save the reset state

            # Initialize current thought chain ID if workflow exists but chain ID is missing
            # This relies on TOOL_GET_WORKFLOW_DETAILS being available
            if self.state.workflow_id and not self.state.current_thought_chain_id:
                await self._set_default_thought_chain_id()

            self.logger.info("Agent loop initialization complete.")
            return True
        except Exception as e:
            self.logger.critical(f"Agent loop initialization failed: {e}", exc_info=True)
            return False

    async def _set_default_thought_chain_id(self):
        """Sets the current_thought_chain_id to the primary chain of the current workflow."""
        # (Keep existing v3.3.5 _set_default_thought_chain_id implementation)
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.debug("Cannot set default thought chain ID: No active workflow.")
            return # No workflow active

        get_details_tool = TOOL_GET_WORKFLOW_DETAILS # Use constant

        # Check if the tool is available
        if self._find_tool_server(get_details_tool):
            try:
                # Use internal call to avoid recording this as a primary agent action
                details = await self._execute_tool_call_internal(
                    get_details_tool,
                    {
                        "workflow_id": current_wf_id,
                        "include_thoughts": True, # Need thoughts to get chain ID
                        "include_actions": False, # Don't need other details
                        "include_artifacts": False,
                        "include_memories": False
                    },
                    record_action=False # This is an internal setup action
                )
                # Check successful execution AND if thought_chains exist in the result
                if details.get("success"):
                    thought_chains = details.get("thought_chains")
                    if isinstance(thought_chains, list) and thought_chains:
                        # Assume the first chain listed is the primary one (usually ordered by creation)
                        first_chain = thought_chains[0]
                        chain_id = first_chain.get("thought_chain_id")
                        if chain_id:
                            self.state.current_thought_chain_id = chain_id
                            self.logger.info(f"Set current_thought_chain_id to primary chain: {_fmt_id(self.state.current_thought_chain_id)} for workflow {_fmt_id(current_wf_id)}")
                            return # Success
                        else:
                             self.logger.warning(f"Primary thought chain found for workflow {current_wf_id}, but it lacks an ID in the details.")
                    else:
                         self.logger.warning(f"Could not find any thought chains in details for workflow {current_wf_id}.")
                else:
                    # Log the error from the tool call if it failed
                    self.logger.error(f"Tool '{get_details_tool}' failed while trying to get default thought chain: {details.get('error')}")

            except Exception as e:
                # Log exceptions during the tool call itself
                self.logger.error(f"Error fetching workflow details to set default thought chain ID: {e}", exc_info=False)
        else:
            self.logger.warning(f"Cannot set default thought chain ID: Tool '{get_details_tool}' unavailable.")

        # Fallback message if chain couldn't be set
        self.logger.info(f"Could not determine primary thought chain ID for WF {_fmt_id(current_wf_id)} on init/load. Will use default on first thought recording.")

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """Checks if a workflow ID exists using get_workflow_details."""
        # (Keep existing efficient v3.3.5 _check_workflow_exists implementation)
        self.logger.debug(f"Checking existence of workflow {_fmt_id(workflow_id)} using {TOOL_GET_WORKFLOW_DETAILS}.")
        tool_name = TOOL_GET_WORKFLOW_DETAILS
        if not self._find_tool_server(tool_name):
            self.logger.error(f"Cannot check workflow existence: Tool {tool_name} unavailable.")
            # Should we assume it exists or not? Assume not for safety.
            return False
        try:
            # Call with minimal includes for efficiency
            result = await self._execute_tool_call_internal(
                tool_name,
                {
                    "workflow_id": workflow_id,
                    "include_actions": False,
                    "include_artifacts": False,
                    "include_thoughts": False,
                    "include_memories": False
                },
                record_action=False # Internal check, don't record as agent action
            )
            # Success means the workflow was found
            return isinstance(result, dict) and result.get("success", False)
        except ToolInputError as e:
            # ToolInputError likely means "not found"
            self.logger.debug(f"Workflow {_fmt_id(workflow_id)} likely not found (ToolInputError: {e}).")
            return False
        except Exception as e:
            self.logger.error(f"Error checking workflow {_fmt_id(workflow_id)} existence: {e}", exc_info=False)
            return False # Assume not found on error

    # ------------------------------------------------ dependency check --
    async def _check_prerequisites(self, ids: List[str]) -> Tuple[bool, str]:
        """Check if all prerequisite actions are completed using get_action_details."""
        # (Keep existing v3.3.5 _check_prerequisites implementation)
        if not ids:
            return True, "No dependencies listed."

        tool_name = TOOL_GET_ACTION_DETAILS
        if not self._find_tool_server(tool_name):
            self.logger.error(f"Cannot check prerequisites: Tool {tool_name} unavailable.")
            return False, f"Tool {tool_name} unavailable."

        self.logger.debug(f"Checking prerequisites: {[_fmt_id(item_id) for item_id in ids]}")
        try:
            res = await self._execute_tool_call_internal(
                tool_name,
                {"action_ids": ids, "include_dependencies": False}, # Don't need nested deps here
                record_action=False # Internal check
            )

            if not res.get("success"):
                error_msg = res.get("error", "Unknown error during check.")
                self.logger.warning(f"Dependency check failed: {error_msg}")
                return False, f"Failed to check dependencies: {error_msg}"

            actions_found = res.get("actions", [])
            found_ids = {a.get("action_id") for a in actions_found}
            missing_ids = list(set(ids) - found_ids)
            if missing_ids:
                self.logger.warning(f"Dependency actions not found: {[_fmt_id(item_id) for item_id in missing_ids]}")
                return False, f"Dependency actions not found: {[_fmt_id(item_id) for item_id in missing_ids]}"

            incomplete_actions = []
            for action in actions_found:
                if action.get("status") != ActionStatus.COMPLETED.value:
                    incomplete_actions.append(
                        f"'{action.get('title', _fmt_id(action.get('action_id')))}' (Status: {action.get('status', 'UNKNOWN')})"
                    )

            if incomplete_actions:
                reason = f"Dependencies not completed: {', '.join(incomplete_actions)}"
                self.logger.warning(reason)
                return False, reason

            self.logger.debug("All dependencies completed.")
            return True, "All dependencies completed."

        except Exception as e:
            self.logger.error(f"Error during prerequisite check: {e}", exc_info=True)
            return False, f"Exception checking prerequisites: {str(e)}"


    # ---------------------------------------------------- action recording --
    # (Keep existing v3.3.5 _record_action_start_internal,
    # _record_action_dependencies_internal, _record_action_completion_internal)
    async def _record_action_start_internal(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        planned_dependencies: Optional[List[str]] = None, # Accept dependencies
    ) -> Optional[str]:
        """
        Persist an Action row marking the start of a tool call.
        Records dependencies if provided.
        Returns the new `action_id`, or None if the recorder tool is offline.
        """
        start_tool = TOOL_RECORD_ACTION_START
        if not self._find_tool_server(start_tool):
            self.logger.error(f"Cannot record action start: Tool '{start_tool}' unavailable.")
            return None

        # Ensure we have a workflow ID
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning("Cannot record action start: No active workflow ID in state.")
            return None

        # Prepare arguments for recording
        payload = {
            "workflow_id": current_wf_id,
            # Generate a more descriptive title if possible
            "title": f"Execute: {tool_name.split(':')[-1]}",
            "action_type": ActionType.TOOL_USE.value,
            # Status will be IN_PROGRESS by default if using record_action_start
            # "status": ActionStatus.IN_PROGRESS.value, # Often handled by the tool itself
            "tool_name": tool_name,
            "tool_args": tool_args, # Tool args are passed directly
            # Add reasoning? Maybe take from context if available? For now, keep simple.
            "reasoning": f"Agent initiated tool call: {tool_name}",
            # Add related thought ID? Requires passing more context here.
        }

        action_id: Optional[str] = None
        try:
            # Use internal call to avoid recursive recording
            res = await self._execute_tool_call_internal(
                start_tool, payload, record_action=False
            )
            if res.get("success"):
                action_id = res.get("action_id")
                if action_id:
                    self.logger.debug(f"Action started: {_fmt_id(action_id)} for tool {tool_name}")
                    # Record declared dependencies *after* successfully getting the action_id
                    if planned_dependencies:
                        await self._record_action_dependencies_internal(action_id, planned_dependencies)
                else:
                    self.logger.warning(f"Tool {start_tool} succeeded but returned no action_id.")
            else:
                self.logger.error(f"Failed to record action start for {tool_name}: {res.get('error')}")

        except Exception as e:
            self.logger.error(f"Exception recording action start for {tool_name}: {e}", exc_info=True)

        return action_id


    async def _record_action_dependencies_internal(
        self,
        source_id: str,
        target_ids: List[str],
    ) -> None:
        """
        Register edges Action->Action (SEQUENTIAL or REQUIRES) in bulk.
        """
        # Ensure source_id and target_ids are valid
        if not source_id or not target_ids:
            self.logger.debug("Skipping dependency recording: Missing source or target IDs.")
            return

        # Filter out empty strings, None, and self-references
        valid_target_ids = {tid for tid in target_ids if tid and tid != source_id}
        if not valid_target_ids:
            self.logger.debug(f"No valid dependencies to record for source action {_fmt_id(source_id)}.")
            return

        dep_tool = TOOL_ADD_ACTION_DEPENDENCY
        if not self._find_tool_server(dep_tool):
            self.logger.error(f"Cannot record dependencies: Tool '{dep_tool}' unavailable.")
            return

        # Ensure we have a workflow ID (should match source action's workflow)
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"Cannot record dependencies for action {_fmt_id(source_id)}: No active workflow ID.")
            return

        self.logger.debug(f"Recording {len(valid_target_ids)} dependencies for action {_fmt_id(source_id)}: depends on {[_fmt_id(tid) for tid in valid_target_ids]}")

        tasks = []
        for target_id in valid_target_ids:
            args = {
                "workflow_id": current_wf_id, # Pass workflow context
                "source_action_id": source_id,
                "target_action_id": target_id,
                "dependency_type": "requires", # Default dependency type
            }
            # Use internal call, don't record these meta-actions
            task = asyncio.create_task(
                self._execute_tool_call_internal(dep_tool, args, record_action=False)
            )
            tasks.append(task)

        # Gather results, log errors but don't block agent
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            target_id = list(valid_target_ids)[i] # Get corresponding target ID
            if isinstance(res, Exception):
                self.logger.error(f"Error recording dependency {_fmt_id(source_id)} -> {_fmt_id(target_id)}: {res}", exc_info=False)
            elif isinstance(res, dict) and not res.get("success"):
                self.logger.warning(f"Failed recording dependency {_fmt_id(source_id)} -> {_fmt_id(target_id)}: {res.get('error')}")
            # else: Successfully recorded


    async def _record_action_completion_internal(
        self,
        action_id: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Persist completion/failure status & result blob for an action.
        """
        completion_tool = TOOL_RECORD_ACTION_COMPLETION
        if not self._find_tool_server(completion_tool):
            self.logger.error(f"Cannot record action completion: Tool '{completion_tool}' unavailable.")
            return

        # Determine status based on the result dictionary
        status = (
            ActionStatus.COMPLETED.value
            if isinstance(result, dict) and result.get("success")
            else ActionStatus.FAILED.value
        )

        # Ensure we have a workflow ID (should match the action's workflow)
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"Cannot record completion for action {_fmt_id(action_id)}: No active workflow ID.")
            # Attempt to proceed without workflow_id? Risky. Let's skip.
            return

        payload = {
            "workflow_id": current_wf_id, # Pass workflow context
            "action_id": action_id,
            "status": status,
            # Pass the full result dictionary to be serialized by the tool
            "tool_result": result,
            # Could add summary here if available/needed
            # "summary": result.get("summary") or result.get("message")
        }

        try:
            # Use internal call, don't record this meta-action
            completion_result = await self._execute_tool_call_internal(
                completion_tool, payload, record_action=False
            )
            if completion_result.get("success"):
                self.logger.debug(f"Action completion recorded for {_fmt_id(action_id)} (Status: {status})")
            else:
                self.logger.error(f"Failed to record action completion for {_fmt_id(action_id)}: {completion_result.get('error')}")
        except Exception as e:
            self.logger.error(f"Exception recording action completion for {_fmt_id(action_id)}: {e}", exc_info=True)

    # ---------------------------------------------------- auto‑link helper --
    # (Add the integrated _run_auto_linking method from the previous step here)
    async def _run_auto_linking(
        self,
        memory_id: str,
        *,
        workflow_id: Optional[str], # Added workflow_id snapshot
        context_id: Optional[str], # Added context_id snapshot (unused here but good practice)
    ) -> None:
        """Background task to automatically link a new memory using richer link types."""
        # Abort if the workflow context changed while queued or shutdown requested
        if workflow_id != self.state.workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping auto-linking for {_fmt_id(memory_id)}: Workflow changed or shutdown.")
            return

        try:
            if not memory_id or not workflow_id:
                self.logger.debug(f"Skipping auto-linking: Missing memory_id ({_fmt_id(memory_id)}) or workflow_id ({_fmt_id(workflow_id)}).")
                return

            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS)) # Use constant
            if self._shutdown_event.is_set(): return # Check again after sleep

            self.logger.debug(f"Attempting auto-linking for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}...")

            # Get source memory details
            source_mem_details_result = await self._execute_tool_call_internal(
                TOOL_GET_MEMORY_BY_ID, {"memory_id": memory_id, "include_links": False}, record_action=False
            )
            # Check success *and* that the memory still belongs to the expected workflow
            if not source_mem_details_result.get("success") or source_mem_details_result.get("workflow_id") != workflow_id:
                self.logger.warning(f"Auto-linking failed for {_fmt_id(memory_id)}: Couldn't retrieve source memory or workflow mismatch.")
                return
            source_mem = source_mem_details_result

            # Use description or content for search query
            query_text = source_mem.get("description", "") or source_mem.get("content", "")[:200] # Limit content length for query
            if not query_text:
                self.logger.debug(f"Skipping auto-linking for {_fmt_id(memory_id)}: No description or content.")
                return

            # Prefer hybrid search if available
            search_tool = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
            if not self._find_tool_server(search_tool):
                self.logger.warning(f"Skipping auto-linking: Tool {search_tool} unavailable.")
                return

            search_args = {
                "workflow_id": workflow_id, # Use snapshotted workflow_id
                "query": query_text,
                "limit": self.auto_linking_max_links + 1, # Fetch one extra to filter self
                "threshold": self.auto_linking_threshold, # Use constant
                "include_content": False # Don't need full content for linking
            }
            if search_tool == TOOL_HYBRID_SEARCH:
                search_args.update({"semantic_weight": 0.8, "keyword_weight": 0.2}) # Prioritize semantics for linking

            similar_results = await self._execute_tool_call_internal(
                search_tool, search_args, record_action=False
            )
            if not similar_results.get("success"):
                self.logger.warning(f"Auto-linking search failed for {_fmt_id(memory_id)}: {similar_results.get('error')}")
                return

            link_count = 0
            score_key = "hybrid_score" if search_tool == TOOL_HYBRID_SEARCH else "similarity"

            for similar_mem_summary in similar_results.get("memories", []):
                if self._shutdown_event.is_set(): break # Check shutdown flag between links

                target_id = similar_mem_summary.get("memory_id")
                similarity_score = similar_mem_summary.get(score_key, 0.0)

                if not target_id or target_id == memory_id: continue

                # Get target memory details to infer better link type
                target_mem_details_result = await self._execute_tool_call_internal(
                    TOOL_GET_MEMORY_BY_ID, {"memory_id": target_id, "include_links": False}, record_action=False
                )
                # Check success *and* workflow match for the target memory
                if not target_mem_details_result.get("success") or target_mem_details_result.get("workflow_id") != workflow_id:
                    self.logger.debug(f"Skipping link target {_fmt_id(target_id)}: Not found or workflow mismatch.")
                    continue
                target_mem = target_mem_details_result

                # Infer link type based on memory types (simple example from v3.3)
                inferred_link_type = LinkType.RELATED.value
                source_type = source_mem.get("memory_type")
                target_type = target_mem.get("memory_type")

                # Add more sophisticated linking logic here based on types if desired
                if source_type == MemoryType.INSIGHT.value and target_type == MemoryType.FACT.value: inferred_link_type = LinkType.SUPPORTS.value
                elif source_type == MemoryType.FACT.value and target_type == MemoryType.INSIGHT.value: inferred_link_type = LinkType.SUPPORTS.value # Bidirectional support?
                elif source_type == MemoryType.QUESTION.value and target_type == MemoryType.FACT.value: inferred_link_type = LinkType.REFERENCES.value
                # ... add other rules as needed ...

                link_tool_name = TOOL_CREATE_LINK
                if not self._find_tool_server(link_tool_name):
                    self.logger.warning(f"Cannot create link: Tool {link_tool_name} unavailable.")
                    break # Stop trying if tool is missing

                # Create the link
                link_args = {
                    # workflow_id is usually inferred by the tool from memory IDs
                    "source_memory_id": memory_id,
                    "target_memory_id": target_id,
                    "link_type": inferred_link_type,
                    "strength": round(similarity_score, 3),
                    "description": f"Auto-link ({inferred_link_type}) based on similarity ({score_key})"
                }
                link_result = await self._execute_tool_call_internal(
                    link_tool_name, link_args, record_action=False
                )

                if link_result.get("success"):
                    link_count += 1
                    self.logger.debug(f"Auto-linked memory {_fmt_id(memory_id)} to {_fmt_id(target_id)} ({inferred_link_type}, score: {similarity_score:.2f})")
                else:
                    # Log failure, but continue trying other links
                    self.logger.warning(f"Failed to auto-create link {_fmt_id(memory_id)}->{_fmt_id(target_id)}: {link_result.get('error')}")

                if link_count >= self.auto_linking_max_links:
                    self.logger.debug(f"Reached auto-linking limit ({self.auto_linking_max_links}) for memory {_fmt_id(memory_id)}.")
                    break
                await asyncio.sleep(0.1) # Small delay between link creations

        except Exception as e:
            # Catch errors within the background task itself
            self.logger.warning(f"Error in auto-linking task for {_fmt_id(memory_id)}: {e}", exc_info=False)


    # ---------------------------------------------------- promotion helper --
    # (Add the integrated _check_and_trigger_promotion method from the previous step here)
    async def _check_and_trigger_promotion(
        self,
        memory_id: str,
        *,
        workflow_id: Optional[str], # Added workflow_id snapshot
        context_id: Optional[str], # Added context_id snapshot (unused here but good practice)
    ):
        """Checks a single memory for promotion and triggers it via TOOL_PROMOTE_MEM."""
        # Abort if the workflow context changed or shutdown requested
        if workflow_id != self.state.workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping promotion check for {_fmt_id(memory_id)}: Workflow changed or shutdown.")
            return

        promotion_tool_name = TOOL_PROMOTE_MEM
        if not memory_id or not self._find_tool_server(promotion_tool_name):
            self.logger.debug(f"Skipping promotion check for {_fmt_id(memory_id)}: Invalid ID or tool unavailable.")
            return

        try:
            # Optional slight delay to avoid thundering herd on DB after a query
            await asyncio.sleep(random.uniform(0.1, 0.4))
            if self._shutdown_event.is_set(): return # Check again after sleep

            self.logger.debug(f"Checking promotion potential for memory {_fmt_id(memory_id)} in workflow {_fmt_id(workflow_id)}...")
            # Execute the promotion check tool
            # Workflow ID is likely inferred by the tool from memory_id, but pass if needed
            promotion_result = await self._execute_tool_call_internal(
                promotion_tool_name, {"memory_id": memory_id}, record_action=False
            )

            # Log success only if promotion actually occurred
            if promotion_result.get("success"):
                if promotion_result.get("promoted"):
                    self.logger.info(f"Memory {_fmt_id(memory_id)} promoted from {promotion_result.get('previous_level')} to {promotion_result.get('new_level')}.", emoji_key="arrow_up")
                else:
                    # Log reason if not promoted but check was successful
                    self.logger.debug(f"Memory {_fmt_id(memory_id)} not promoted: {promotion_result.get('reason')}")
            else:
                 # Log if the promotion check tool itself failed
                 self.logger.warning(f"Promotion check tool failed for {_fmt_id(memory_id)}: {promotion_result.get('error')}")

        except Exception as e:
            self.logger.warning(f"Error in memory promotion check task for {_fmt_id(memory_id)}: {e}", exc_info=False)


    # ------------------------------------------------------ execute tool call --
    # (Keep the integrated _execute_tool_call_internal from the previous step here,
    # ensuring background triggers are included)
    async def _execute_tool_call_internal(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        record_action: bool = True,
        planned_dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Handles server lookup, dependency checks, execution, results, optional action recording, dependency recording, and triggers."""
        target_server = self._find_tool_server(tool_name)
        # Allow internal agent tool call without a server
        if not target_server and tool_name != AGENT_TOOL_UPDATE_PLAN:
            err = f"Tool server unavailable for {tool_name}"
            self.logger.error(err)
            # Set error details for the main loop
            self.state.last_error_details = {"tool": tool_name, "error": err, "type": "ServerUnavailable"}
            return {"success": False, "error": err, "status_code": 503}

        current_wf_id = (
            self.state.workflow_stack[-1]
            if self.state.workflow_stack
            else self.state.workflow_id
        )

        # Inject context IDs automatically if missing and relevant
        # Make a copy to avoid modifying the original arguments dict
        final_arguments = arguments.copy()
        if (
            final_arguments.get("workflow_id") is None
            and current_wf_id
            and tool_name
            not in { # Tools that don't need/take workflow_id explicitly
                TOOL_CREATE_WORKFLOW,
                TOOL_LIST_WORKFLOWS,
                "core:list_servers",
                "core:get_tool_schema",
                AGENT_TOOL_UPDATE_PLAN,
            }
        ):
            final_arguments["workflow_id"] = current_wf_id
        if (
            final_arguments.get("context_id") is None
            and self.state.context_id
            and tool_name
            in { # Tools operating on a specific cognitive context
                TOOL_GET_WORKING_MEMORY,
                TOOL_OPTIMIZE_WM,
                TOOL_AUTO_FOCUS,
                # Add others if they operate on context_id
            }
        ):
            final_arguments["context_id"] = self.state.context_id
        if (
            final_arguments.get("thought_chain_id") is None
            and self.state.current_thought_chain_id
            and tool_name == TOOL_RECORD_THOUGHT
        ):
            final_arguments["thought_chain_id"] = self.state.current_thought_chain_id

        # Dependency check before proceeding
        if planned_dependencies:
            ok, reason = await self._check_prerequisites(planned_dependencies)
            if not ok:
                err_msg = f"Prerequisites not met for {tool_name}: {reason}"
                self.logger.warning(err_msg)
                self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "DependencyNotMetError", "dependencies": planned_dependencies}
                self.state.needs_replan = True # Dependency failure requires replanning
                return {"success": False, "error": err_msg, "status_code": 412} # 412 Precondition Failed
            else:
                self.logger.info(f"Prerequisites {[_fmt_id(dep) for dep in planned_dependencies]} met for {tool_name}.")

        # Handle internal plan‑update tool directly
        if tool_name == AGENT_TOOL_UPDATE_PLAN:
            try:
                new_plan_data = final_arguments.get("plan", [])
                if not isinstance(new_plan_data, list):
                    raise ValueError("`plan` argument must be a list of step objects.")
                # Validate each step using the Pydantic model
                validated_plan = [PlanStep(**p) for p in new_plan_data]
                self.state.current_plan = validated_plan
                self.state.needs_replan = False # Plan was explicitly updated
                self.logger.info(f"Internal plan update successful. New plan has {len(validated_plan)} steps.")
                # Clear errors after successful plan update
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                return {"success": True, "message": f"Plan updated with {len(validated_plan)} steps."}
            except (ValidationError, TypeError, ValueError) as e:
                err_msg = f"Failed to validate/apply new plan: {e}"
                self.logger.error(err_msg)
                self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "PlanUpdateError"}
                # Don't increment consecutive errors for internal agent failure? Or maybe do? Let's increment.
                self.state.consecutive_error_count += 1
                self.state.needs_replan = True # Failed plan update needs replan
                return {"success": False, "error": err_msg}

        # --- Action Start Recording ---
        action_id: Optional[str] = None
        # Use the _INTERNAL_OR_META_TOOLS set defined at class level
        should_record = record_action and tool_name not in self._INTERNAL_OR_META_TOOLS
        if should_record:
            action_id = await self._record_action_start_internal(
                 tool_name, final_arguments, planned_dependencies # Pass potentially modified args
            )
            # Note: _record_action_start_internal now handles dependency recording

        # --- Define the actual tool call for retry wrapper ---
        async def _do_call():
            # Ensure None values are stripped *before* sending to MCPClient execute
            # MCPClient execute_tool likely handles this, but belt-and-suspenders
            call_args = {k: v for k, v in final_arguments.items() if v is not None}
            # Target server must be valid here because AGENT_INTERNAL was handled
            return await self.mcp_client.execute_tool(target_server, tool_name, call_args)

        record_stats = self.state.tool_usage_stats[tool_name]
        # --- Decide if tool is safe to auto‑retry ---
        idempotent = tool_name in {
            # Read-only operations are generally idempotent
            TOOL_GET_CONTEXT, TOOL_GET_MEMORY_BY_ID, TOOL_SEMANTIC_SEARCH,
            TOOL_HYBRID_SEARCH, TOOL_GET_ACTION_DETAILS, TOOL_LIST_WORKFLOWS,
            TOOL_COMPUTE_STATS, TOOL_GET_WORKING_MEMORY, TOOL_GET_LINKED_MEMORIES,
            TOOL_GET_ARTIFACTS, TOOL_GET_ARTIFACT_BY_ID, TOOL_GET_ACTION_DEPENDENCIES,
            TOOL_GET_THOUGHT_CHAIN, TOOL_GET_WORKFLOW_DETAILS,
            # Some meta operations might be considered safe
            TOOL_SUMMARIZE_TEXT,
        }

        start_ts = time.time()
        res = {} # Initialize result dict

        try:
            # Execute with retries
            raw = await self._with_retries(
                _do_call,
                max_retries=3 if idempotent else 1, # Retry only idempotent tools
                # Retry on common transient errors or specific tool errors if needed
                retry_exceptions=(ToolError, ToolInputError, asyncio.TimeoutError, ConnectionError),
            )
            latency_ms = (time.time() - start_ts) * 1000
            record_stats["latency_ms_total"] += latency_ms

            # Process result format (handle dicts vs other types)
            if isinstance(raw, dict) and ("success" in raw or "isError" in raw):
                # Assume standard MCP result format
                is_error = raw.get("isError", not raw.get("success", True))
                content = raw.get("content", raw.get("error", raw.get("data"))) # Get content or error
                if is_error:
                    res = {"success": False, "error": str(content), "status_code": raw.get("status_code")}
                else:
                    # If content itself contains success/error, use that structure
                    if isinstance(content, dict) and "success" in content:
                        res = content
                    else: # Otherwise wrap the content under a 'data' key
                        res = {"success": True, "data": content}
            elif isinstance(raw, dict): # Handle dicts without explicit success/isError
                 res = {"success": True, "data": raw} # Assume success if no error indicators
            else:
                # Handle non-dict results (e.g., simple strings, numbers)
                res = {"success": True, "data": raw}

            # --- State updates and Background Triggers on SUCCESS ---
            if res.get("success"):
                # Reset error count ONLY if successful. Heuristics handle counters.
                # self.state.consecutive_error_count = 0 # Let _apply_heuristic_plan_update handle this
                record_stats["success"] += 1

                # --- Background Triggers Integration ---
                current_wf_id_snapshot = self.state.workflow_id # Snapshot WF ID  # noqa: F841

                # Auto-linking triggers
                if tool_name in [TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY] and res.get("memory_id"):
                    self.logger.debug(f"Queueing auto-link check for memory {_fmt_id(res['memory_id'])}")
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=res["memory_id"])
                if tool_name == TOOL_RECORD_ARTIFACT and res.get("linked_memory_id"):
                    self.logger.debug(f"Queueing auto-link check for memory linked to artifact: {_fmt_id(res['linked_memory_id'])}")
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=res["linked_memory_id"])

                # Promotion check triggers (after retrieval)
                if tool_name in [TOOL_GET_MEMORY_BY_ID, TOOL_QUERY_MEMORIES, TOOL_HYBRID_SEARCH, TOOL_SEMANTIC_SEARCH, TOOL_GET_WORKING_MEMORY]:
                    mem_ids_to_check = set() # Use set to avoid duplicates
                    if tool_name == TOOL_GET_MEMORY_BY_ID:
                        # Result might be directly the memory dict or nested under 'data'
                        mem_data = res if "memory_id" in res else res.get("data", {})
                        if isinstance(mem_data, dict):
                             mem_ids_to_check.add(mem_data.get("memory_id"))
                    elif tool_name == TOOL_GET_WORKING_MEMORY:
                        wm_data = res.get("working_memories", [])
                        if isinstance(wm_data, list):
                            focus_id = res.get("focal_memory_id")
                            if focus_id: mem_ids_to_check.add(focus_id)
                            mem_ids_to_check.update(m.get("memory_id") for m in wm_data[:2] if isinstance(m,dict) and m.get("memory_id")) # Check top 2
                    else: # Query/Search results
                        # Results might be under 'memories' or 'data.memories' etc.
                        memories = res.get("memories", res.get("data", {}).get("memories") if isinstance(res.get("data"), dict) else [])
                        if isinstance(memories, list):
                            mem_ids_to_check.update(m.get("memory_id") for m in memories[:3] if isinstance(m,dict) and m.get("memory_id")) # Check top 3

                    for mem_id in filter(None, mem_ids_to_check): # Filter out None IDs
                         self.logger.debug(f"Queueing promotion check for retrieved memory {_fmt_id(mem_id)}")
                         self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id)


                # Update current thought chain ID if a new one was created
                if tool_name == TOOL_CREATE_THOUGHT_CHAIN and res.get("success"):
                    # Result might be directly the chain dict or nested under 'data'
                    chain_data = res if "thought_chain_id" in res else res.get("data", {})
                    if isinstance(chain_data, dict):
                        new_chain_id = chain_data.get("thought_chain_id")
                        if new_chain_id:
                            self.state.current_thought_chain_id = new_chain_id
                            self.logger.info(f"Switched current thought chain to newly created: {_fmt_id(new_chain_id)}")
                # --- End Background Triggers ---

            else: # Tool failed
                # Don't increment error count here; let _apply_heuristic_plan_update handle it
                record_stats["failure"] += 1
                # Ensure error details are captured from the result
                self.state.last_error_details = {
                    "tool": tool_name,
                    "args": arguments, # Log the arguments that caused failure
                    "error": res.get("error", "Unknown failure"),
                    "status_code": res.get("status_code"),
                    "type": "ToolExecutionError"
                }
                # If it was a dependency error, update type
                if res.get("status_code") == 412:
                     self.state.last_error_details["type"] = "DependencyNotMetError"

            # Update last action summary (more detailed)
            summary = ""
            if res.get("success"):
                # Try to find a meaningful summary field
                for k in ("summary", "message", "memory_id", "action_id", "artifact_id", "chain_id", "state_id", "report", "visualization"):
                    if k in res and res[k]:
                        summary = f"{k}: {_fmt_id(res[k]) if 'id' in k else str(res[k])}"
                        break
                    elif isinstance(res.get("data"), dict) and k in res["data"] and res["data"][k]:
                         summary = f"{k}: {_fmt_id(res['data'][k]) if 'id' in k else str(res['data'][k])}"
                         break
                else: # Fallback for generic success
                    data_preview = str(res.get("data", "Success"))[:70]
                    summary = f"Success. Data: {data_preview}..." if len(str(res.get("data", ""))) > 70 else f"Success. Data: {data_preview}"
            else:
                summary = f"Failed: {str(res.get('error', 'Unknown Error'))[:100]}"
                if res.get('status_code'): summary += f" (Code: {res['status_code']})"

            self.state.last_action_summary = f"{tool_name} -> {summary}"
            self.logger.info(self.state.last_action_summary, emoji_key="checkered_flag" if res.get('success') else "warning")


        except (ToolError, ToolInputError) as e:
            # Handle specific MCP exceptions caught during _with_retries or directly
            err_str = str(e); status_code = getattr(e, 'status_code', None)
            self.logger.error(f"Tool Error executing {tool_name}: {err_str}", exc_info=False)
            res = {"success": False, "error": err_str, "status_code": status_code}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": type(e).__name__, "status_code": status_code}
            self.state.last_action_summary = f"{tool_name} -> Failed: {err_str[:100]}"
            if status_code == 412: # Explicitly mark dependency failures for replanning
                self.state.last_error_details["type"] = "DependencyNotMetError"
                # self.state.needs_replan = True # Let heuristic update handle replan flag

        except asyncio.CancelledError:
             # Handle task cancellation gracefully
             err_str = "Tool execution cancelled."
             self.logger.warning(f"{tool_name} execution was cancelled.")
             res = {"success": False, "error": err_str, "status_code": 499} # Use 499 Client Closed Request
             record_stats["failure"] += 1 # Count cancellation as failure for stats
             self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "CancelledError"}
             self.state.last_action_summary = f"{tool_name} -> Cancelled"

        except Exception as e:
            # Catch any other unexpected errors during execution or retries
            err_str = str(e)
            self.logger.error(f"Unexpected Error executing {tool_name}: {err_str}", exc_info=True)
            res = {"success": False, "error": f"Unexpected error: {err_str}"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "UnexpectedExecutionError"}
            self.state.last_action_summary = f"{tool_name} -> Failed: Unexpected error."
            # self.state.needs_replan = True # Let heuristic update handle replan flag


        # --- Action Completion Recording ---
        if action_id: # Only record completion if start was recorded
            # Pass the final result 'res' to the completion recorder
            await self._record_action_completion_internal(action_id, res)


        # --- Handle workflow side effects based on the final result ---
        # (Moved *after* completion recording, using the final 'res')
        await self._handle_workflow_side_effects(tool_name, final_arguments, res)


        return res

    async def _handle_workflow_side_effects(self, tool_name: str, arguments: Dict, result_content: Dict):
        """Handles state changes after specific tool calls like workflow creation/completion."""
        # (Keep existing v3.3.5 _handle_workflow_side_effects implementation, ensure _set_default_thought_chain called)
        # --- Side effects for Workflow Creation ---
        if tool_name == TOOL_CREATE_WORKFLOW and result_content.get("success"):
            new_wf_id = result_content.get("workflow_id")
            primary_chain_id = result_content.get("primary_thought_chain_id")
            parent_id = arguments.get("parent_workflow_id") # Get parent from original args

            if new_wf_id:
                # Update primary workflow ID and context ID
                self.state.workflow_id = new_wf_id
                self.state.context_id = new_wf_id # Align context ID with new workflow

                # Manage workflow stack
                if parent_id and parent_id in self.state.workflow_stack:
                    # If created as a sub-workflow of an existing one on the stack
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
                # Reset meta-counters for new workflow? Optional, depends on desired behavior.
                # self.state.successful_actions_since_reflection = 0
                # self.state.successful_actions_since_consolidation = 0

        # --- Side effects for Workflow Status Update (Completion/Failure) ---
        elif tool_name == TOOL_UPDATE_WORKFLOW_STATUS and result_content.get("success"):
            status = arguments.get("status") # Status requested
            wf_id_updated = arguments.get("workflow_id") # Workflow that was updated

            # Check if the *current* workflow on top of the stack was finished
            if wf_id_updated and self.state.workflow_stack and wf_id_updated == self.state.workflow_stack[-1]:
                # Check if status indicates terminal state
                is_terminal = status in [
                    WorkflowStatus.COMPLETED.value,
                    WorkflowStatus.FAILED.value,
                    WorkflowStatus.ABANDONED.value
                ]

                if is_terminal:
                    finished_wf = self.state.workflow_stack.pop() # Remove finished workflow
                    if self.state.workflow_stack:
                        # Return to parent workflow
                        self.state.workflow_id = self.state.workflow_stack[-1]
                        self.state.context_id = self.state.workflow_id # Realign context ID
                        # Fetch parent's primary thought chain ID
                        await self._set_default_thought_chain_id()
                        self.logger.info(f"Sub-workflow {_fmt_id(finished_wf)} finished ({status}). Returning to parent {_fmt_id(self.state.workflow_id)}. Current chain: {_fmt_id(self.state.current_thought_chain_id)}", emoji_key="arrow_left")
                        # Force replan in parent context
                        self.state.needs_replan = True
                        self.state.current_plan = [PlanStep(description=f"Returned from sub-workflow {_fmt_id(finished_wf)} (status: {status}). Re-assess parent goal.")]
                        self.state.last_error_details = None # Clear error from sub-task
                    else:
                        # Root workflow finished
                        self.logger.info(f"Root workflow {_fmt_id(finished_wf)} finished with status: {status}.")
                        self.state.workflow_id = None
                        self.state.context_id = None
                        self.state.current_thought_chain_id = None
                        # Set goal achieved flag only if completed successfully
                        if status == WorkflowStatus.COMPLETED.value:
                             self.state.goal_achieved_flag = True
                        else:
                             self.state.goal_achieved_flag = False # Mark as not achieved if failed/abandoned
                        # Clear plan
                        self.state.current_plan = []


    # (Keep the integrated _apply_heuristic_plan_update method from the previous step here)
    async def _apply_heuristic_plan_update(self, last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        """
        Applies heuristic updates to the plan when the LLM doesn't explicitly
        call agent:update_plan. Based on v3.3 _update_plan logic.
        Also updates meta-counters based on action success.
        """
        self.logger.info("Applying heuristic plan update...", emoji_key="clipboard")

        if not self.state.current_plan:
            self.logger.warning("Plan is empty during heuristic update, adding default re-evaluation step.")
            self.state.current_plan = [PlanStep(description="Fallback: Re-evaluate situation.")]
            self.state.needs_replan = True # Force replan if plan was empty
            return

        current_step = self.state.current_plan[0]
        decision_type = last_decision.get("decision")

        action_successful = False # Flag if the last action contributing to plan step was successful
        tool_name_executed = last_decision.get("tool_name") # Get tool name if applicable

        if decision_type == "call_tool":
            # Check result success, ensure it's a dictionary
            tool_success = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            action_successful = tool_success # Tool call success directly maps to action success for plan update

            if tool_success:
                current_step.status = "completed"
                # Try to get meaningful summary or ID from result
                summary = "Success."
                if isinstance(last_tool_result_content, dict):
                     # Prioritize specific meaningful keys
                     summary_keys = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "chain_id", "state_id", "report", "visualization"]
                     for k in summary_keys:
                          if k in last_tool_result_content and last_tool_result_content[k]:
                               summary = f"{k}: {_fmt_id(last_tool_result_content[k]) if 'id' in k else str(last_tool_result_content[k])}"
                               break
                     else: # Fallback to 'data' or generic success
                          data_preview = str(last_tool_result_content.get("data", "Success."))[:70]
                          summary = f"Data: {data_preview}..." if len(str(last_tool_result_content.get('data', ''))) > 70 else f"Data: {data_preview}"

                current_step.result_summary = summary[:150] # Truncate summary
                self.state.current_plan.pop(0) # Remove completed step
                if not self.state.current_plan:
                    self.logger.info("Plan completed. Adding final analysis step.")
                    self.state.current_plan.append(PlanStep(description="Plan finished. Analyze overall result and decide if goal is met."))
                self.state.needs_replan = False # Success usually doesn't require replan unless LLM stated otherwise
            else: # Tool failed
                current_step.status = "failed"
                error_msg = "Unknown failure"
                if isinstance(last_tool_result_content, dict):
                     error_msg = str(last_tool_result_content.get('error', 'Unknown failure'))
                current_step.result_summary = f"Failure: {error_msg[:150]}"
                # Keep failed step in plan for context, insert analysis step after it
                # Avoid inserting duplicate analysis steps
                if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Analyze failure of step"):
                    self.state.current_plan.insert(1, PlanStep(description=f"Analyze failure of step '{current_step.description[:30]}...' and replan."))
                self.state.needs_replan = True # Failure requires replan

        elif decision_type == "thought_process":
            action_successful = True # Recording a thought is usually considered success for plan progression
            current_step.status = "completed" # Assume the thought fulfills the current plan step's intent
            current_step.result_summary = f"Thought Recorded: {last_decision.get('content','')[:50]}..."
            self.state.current_plan.pop(0)
            if not self.state.current_plan:
                self.logger.info("Plan completed after thought. Adding final analysis step.")
                self.state.current_plan.append(PlanStep(description="Decide next action based on recorded thought and overall goal."))
            self.state.needs_replan = False # Recording thought usually doesn't force replan

        elif decision_type == "complete":
            action_successful = True
            self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing.", status="completed")]
            self.state.needs_replan = False

        else: # Includes decision_type == "error" or unhandled agent decisions
            action_successful = False
            current_step.status = "failed"
            err_summary = self.state.last_action_summary or "Unknown agent error"
            current_step.result_summary = f"Agent/Tool Error: {err_summary[:100]}..."
            # Keep failed step, insert analysis step
            if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Re-evaluate due to agent error"):
                 self.state.current_plan.insert(1, PlanStep(description="Re-evaluate due to agent error or unclear decision."))
            self.state.needs_replan = True # Agent error requires replan

        # --- Update Meta-Counters based on Action Success ---
        if action_successful:
            self.state.consecutive_error_count = 0 # Reset error count on any success
            # Increment successful action counts *only* if the action wasn't purely internal/meta
            if tool_name_executed and tool_name_executed not in self._INTERNAL_OR_META_TOOLS:
                 # Use float increments to allow fractional progress for thoughts later if desired
                 self.state.successful_actions_since_reflection += 1.0
                 self.state.successful_actions_since_consolidation += 1.0
                 self.logger.debug(f"Incremented success counters R:{self.state.successful_actions_since_reflection}, C:{self.state.successful_actions_since_consolidation}")
            elif decision_type == "thought_process":
                 # Optional: Count thoughts as minor progress (e.g., 0.25 or 0.5)
                 # self.state.successful_actions_since_reflection += 0.5
                 # self.state.successful_actions_since_consolidation += 0.5
                 pass # Currently not counting thoughts towards thresholds

        else: # Action failed
            self.state.consecutive_error_count += 1
            self.logger.warning(f"Consecutive error count increased to: {self.state.consecutive_error_count}")
            # Reset reflection counter immediately on error to encourage reflection sooner
            if self.state.successful_actions_since_reflection > 0:
                 self.logger.info(f"Resetting reflection counter due to error (was {self.state.successful_actions_since_reflection:.1f}).")
                 self.state.successful_actions_since_reflection = 0
                 # Consider if consolidation counter should also be reset or paused on error?
                 # Let's keep consolidation counter running unless error rate is very high.


        # --- Log Final Plan State ---
        log_plan_msg = f"Plan updated heuristically. Steps remaining: {len(self.state.current_plan)}. "
        if self.state.current_plan:
            next_step = self.state.current_plan[0]
            log_plan_msg += f"Next: '{next_step.description[:60]}...' (Status: {next_step.status}, Depends: {[_fmt_id(d) for d in next_step.depends_on] if next_step.depends_on else 'None'})"
        else:
            log_plan_msg += "Plan is now empty."
        self.logger.info(log_plan_msg, emoji_key="clipboard")


    # ------------------------------------------------ adaptive thresholds --
    # (Keep existing v3.3.5 _adapt_thresholds implementation)
    def _adapt_thresholds(self, stats: Dict[str, Any]) -> None:
        """Adjusts reflection and consolidation thresholds based on memory stats."""
        if not stats or not stats.get("success"):
             self.logger.warning("Cannot adapt thresholds: Invalid or failed stats received.")
             return

        self.logger.debug(f"Attempting threshold adaptation based on stats: {stats}")
        adjustment_factor = 0.1 # How much to adjust thresholds by each time
        changed = False

        # --- Consolidation Threshold Adaptation ---
        episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
        target_episodic = BASE_CONSOLIDATION_THRESHOLD * 1.5 # Target range guidepost

        # Lower threshold if episodic count is significantly high
        if episodic_count > target_episodic * 1.5: # e.g., > 2.25 * base
            # Calculate potential new threshold, decrease by factor
            potential_new = max(MIN_CONSOLIDATION_THRESHOLD, self.state.current_consolidation_threshold - math.ceil(self.state.current_consolidation_threshold * adjustment_factor))
            if potential_new < self.state.current_consolidation_threshold:
                self.logger.info(f"High episodic count ({episodic_count}). Lowering consolidation threshold: {self.state.current_consolidation_threshold} -> {potential_new}")
                self.state.current_consolidation_threshold = potential_new
                changed = True
        # Raise threshold if episodic count is low (agent might be consolidating too often or not generating enough)
        elif episodic_count < target_episodic * 0.75: # e.g., < 1.125 * base
            # Calculate potential new threshold, increase by factor
             potential_new = min(MAX_CONSOLIDATION_THRESHOLD, self.state.current_consolidation_threshold + math.ceil(self.state.current_consolidation_threshold * adjustment_factor))
             if potential_new > self.state.current_consolidation_threshold:
                 self.logger.info(f"Low episodic count ({episodic_count}). Raising consolidation threshold: {self.state.current_consolidation_threshold} -> {potential_new}")
                 self.state.current_consolidation_threshold = potential_new
                 changed = True

        # --- Reflection Threshold Adaptation ---
        # Use tool usage stats accumulated in agent state
        total_calls = sum(v.get("success", 0) + v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        total_failures = sum(v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        # Calculate failure rate, avoid division by zero, require minimum calls for significance
        min_calls_for_rate = 5
        failure_rate = (total_failures / total_calls) if total_calls >= min_calls_for_rate else 0.0

        # Lower threshold if failure rate is high
        if failure_rate > 0.25: # Threshold for high failure rate
             potential_new = max(MIN_REFLECTION_THRESHOLD, self.state.current_reflection_threshold - math.ceil(self.state.current_reflection_threshold * adjustment_factor))
             if potential_new < self.state.current_reflection_threshold:
                 self.logger.info(f"High tool failure rate ({failure_rate:.1%}). Lowering reflection threshold: {self.state.current_reflection_threshold} -> {potential_new}")
                 self.state.current_reflection_threshold = potential_new
                 changed = True
        # Raise threshold if failure rate is very low (agent performing well)
        elif failure_rate < 0.05 and total_calls > 10: # Threshold for low rate, ensure enough data
             potential_new = min(MAX_REFLECTION_THRESHOLD, self.state.current_reflection_threshold + math.ceil(self.state.current_reflection_threshold * adjustment_factor))
             if potential_new > self.state.current_reflection_threshold:
                  self.logger.info(f"Low tool failure rate ({failure_rate:.1%}). Raising reflection threshold: {self.state.current_reflection_threshold} -> {potential_new}")
                  self.state.current_reflection_threshold = potential_new
                  changed = True

        if not changed:
             self.logger.debug("No threshold adjustments triggered based on current stats.")

    # ------------------------------------------------ periodic task runner --
    # (Keep existing v3.3.5 _run_periodic_tasks implementation, ensure it calls
    # the integrated _check_and_trigger_promotion and uses dynamic thresholds)
    async def _run_periodic_tasks(self):
        """Runs meta-cognition and maintenance tasks, including adaptive adjustments."""
        # Prevent running if no workflow or shutting down
        if not self.state.workflow_id or not self.state.context_id or self._shutdown_event.is_set():
            return

        tasks_to_run: List[Tuple[str, Dict]] = []
        trigger_reasons: List[str] = [] # Collect reasons for logging

        # Check tool availability once
        reflection_tool_available = self._find_tool_server(TOOL_REFLECTION) is not None
        consolidation_tool_available = self._find_tool_server(TOOL_CONSOLIDATION) is not None
        optimize_wm_tool_available = self._find_tool_server(TOOL_OPTIMIZE_WM) is not None
        auto_focus_tool_available = self._find_tool_server(TOOL_AUTO_FOCUS) is not None
        promote_mem_tool_available = self._find_tool_server(TOOL_PROMOTE_MEM) is not None
        stats_tool_available = self._find_tool_server(TOOL_COMPUTE_STATS) is not None
        maintenance_tool_available = self._find_tool_server(TOOL_DELETE_EXPIRED_MEMORIES) is not None

        # --- Tier 3: Stats Check & Adaptation Trigger ---
        self.state.loops_since_stats_adaptation += 1
        if self.state.loops_since_stats_adaptation >= STATS_ADAPTATION_INTERVAL:
            if stats_tool_available:
                trigger_reasons.append("StatsInterval")
                try:
                    stats = await self._execute_tool_call_internal(
                        TOOL_COMPUTE_STATS, {"workflow_id": self.state.workflow_id}, record_action=False
                    )
                    if stats.get("success"):
                        # Adapt thresholds based on the fetched stats
                        self._adapt_thresholds(stats)
                        # Example: Trigger consolidation if episodic memories are high based on stats
                        episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
                        # Trigger if count significantly exceeds current *dynamic* threshold
                        if episodic_count > (self.state.current_consolidation_threshold * 2.0) and consolidation_tool_available:
                            if not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run): # Avoid duplicates
                                self.logger.info(f"High episodic count ({episodic_count}) detected via stats, scheduling consolidation.")
                                tasks_to_run.append((TOOL_CONSOLIDATION, {"workflow_id": self.state.workflow_id, "consolidation_type": "summary", "query_filter": {"memory_level": MemoryLevel.EPISODIC.value}, "max_source_memories": self.consolidation_max_sources}))
                                trigger_reasons.append(f"HighEpisodic({episodic_count})")
                                self.state.successful_actions_since_consolidation = 0 # Reset counter as we're consolidating now
                    else:
                        self.logger.warning(f"Failed to compute stats for adaptation: {stats.get('error')}")
                except Exception as e:
                    self.logger.error(f"Error during stats computation/adaptation: {e}", exc_info=False)
                finally:
                     # Reset interval counter regardless of success/failure
                     self.state.loops_since_stats_adaptation = 0
            else:
                self.logger.warning(f"Skipping stats/adaptation: Tool {TOOL_COMPUTE_STATS} not available")


        # --- Tier 3: Maintenance Check ---
        self.state.loops_since_maintenance += 1
        if self.state.loops_since_maintenance >= MAINTENANCE_INTERVAL:
            if maintenance_tool_available:
                # Append maintenance task to the list
                tasks_to_run.append((TOOL_DELETE_EXPIRED_MEMORIES, {}))
                trigger_reasons.append("MaintenanceInterval")
                self.state.loops_since_maintenance = 0 # Reset interval counter
            else:
                self.logger.warning(f"Skipping maintenance: Tool {TOOL_DELETE_EXPIRED_MEMORIES} not available")

        # --- Existing Triggers (Using Dynamic Thresholds) ---
        # Reflection Trigger (Check replan flag OR success counter vs. dynamic threshold)
        if self.state.needs_replan or self.state.successful_actions_since_reflection >= self.state.current_reflection_threshold:
            if reflection_tool_available:
                if not any(task[0] == TOOL_REFLECTION for task in tasks_to_run): # Avoid duplicates
                    # Cycle through reflection types
                    reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
                    tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
                    reason_str = f"ReplanNeeded({self.state.needs_replan})" if self.state.needs_replan else f"SuccessCount({self.state.successful_actions_since_reflection:.1f}>={self.state.current_reflection_threshold})"
                    trigger_reasons.append(f"Reflect({reason_str})")
                    # Reset counter and advance cycle index
                    self.state.successful_actions_since_reflection = 0
                    self.state.reflection_cycle_index += 1
            else:
                self.logger.warning(f"Skipping reflection: Tool {TOOL_REFLECTION} not available")
                # Reset counter even if tool unavailable to prevent immediate re-trigger
                self.state.successful_actions_since_reflection = 0

        # Consolidation Trigger (Check success counter vs. dynamic threshold)
        if self.state.successful_actions_since_consolidation >= self.state.current_consolidation_threshold:
            if consolidation_tool_available:
                if not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run): # Avoid duplicates
                    tasks_to_run.append((TOOL_CONSOLIDATION, {"workflow_id": self.state.workflow_id, "consolidation_type": "summary", "query_filter": {"memory_level": MemoryLevel.EPISODIC.value}, "max_source_memories": self.consolidation_max_sources}))
                    trigger_reasons.append(f"ConsolidateThreshold({self.state.successful_actions_since_consolidation:.1f}>={self.state.current_consolidation_threshold})")
                    # Reset counter
                    self.state.successful_actions_since_consolidation = 0
            else:
                self.logger.warning(f"Skipping consolidation: Tool {TOOL_CONSOLIDATION} not available")
                 # Reset counter even if tool unavailable
                self.state.successful_actions_since_consolidation = 0

        # Optimization Trigger (Based on loop interval)
        self.state.loops_since_optimization += 1
        if self.state.loops_since_optimization >= OPTIMIZATION_LOOP_INTERVAL:
            if optimize_wm_tool_available:
                tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.state.context_id}))
                trigger_reasons.append("OptimizeInterval")
            else:
                self.logger.warning(f"Skipping optimization: Tool {TOOL_OPTIMIZE_WM} not available")

            if auto_focus_tool_available:
                tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.state.context_id}))
                trigger_reasons.append("FocusUpdate")
            else:
                self.logger.warning(f"Skipping auto-focus: Tool {TOOL_AUTO_FOCUS} not available")

            self.state.loops_since_optimization = 0 # Reset interval counter

        # Promotion Check Trigger (Based on loop interval)
        self.state.loops_since_promotion_check += 1
        if self.state.loops_since_promotion_check >= MEMORY_PROMOTION_LOOP_INTERVAL:
            if promote_mem_tool_available:
                # Schedule the check itself, not individual promotions
                tasks_to_run.append(("CHECK_PROMOTIONS", {}))
                trigger_reasons.append("PromotionInterval")
            else:
                self.logger.warning(f"Skipping promotion check: Tool {TOOL_PROMOTE_MEM} not available")
            self.state.loops_since_promotion_check = 0 # Reset interval counter

        # --- Execute Scheduled Tasks ---
        if tasks_to_run:
            unique_reasons_str = ', '.join(sorted(set(trigger_reasons))) # Deduplicate reasons for logging
            self.logger.info(f"Running {len(tasks_to_run)} periodic tasks (Triggers: {unique_reasons_str})...", emoji_key="brain")

            # Prioritize maintenance and stats if scheduled
            tasks_to_run.sort(key=lambda x: 0 if x[0] == TOOL_DELETE_EXPIRED_MEMORIES else 1 if x[0] == TOOL_COMPUTE_STATS else 2)

            for tool_name, args in tasks_to_run:
                # Check shutdown flag before each task
                if self._shutdown_event.is_set():
                    self.logger.info("Shutdown detected during periodic tasks, aborting remaining.")
                    break
                try:
                    if tool_name == "CHECK_PROMOTIONS":
                        # Special case: trigger the internal check method
                        await self._trigger_promotion_checks()
                        continue # Move to next task

                    # Execute standard tool calls
                    self.logger.debug(f"Executing periodic task: {tool_name} with args: {args}")
                    result_content = await self._execute_tool_call_internal(
                        tool_name, args, record_action=False # Don't record periodic tasks as agent actions
                    )

                    # --- Meta-Cognition Feedback Loop ---
                    if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and result_content.get('success'):
                        feedback = ""
                        # Extract content based on tool type
                        if tool_name == TOOL_REFLECTION:
                            feedback = result_content.get("content", "")
                        elif tool_name == TOOL_CONSOLIDATION:
                             feedback = result_content.get("consolidated_content", "")

                        # Handle nested 'data' key if necessary
                        if not feedback and isinstance(result_content.get("data"), dict):
                            if tool_name == TOOL_REFLECTION:
                                feedback = result_content["data"].get("content", "")
                            elif tool_name == TOOL_CONSOLIDATION:
                                feedback = result_content["data"].get("consolidated_content", "")

                        if feedback:
                            # Create a concise summary for the feedback state
                            feedback_summary = str(feedback).split('\n', 1)[0][:150] # First line up to 150 chars
                            self.state.last_meta_feedback = f"Feedback from {tool_name.split(':')[-1]}: {feedback_summary}..."
                            self.logger.info(f"Received meta-feedback: {self.state.last_meta_feedback}")
                            # Force replan after receiving feedback from reflection/consolidation
                            self.state.needs_replan = True
                        else:
                            self.logger.debug(f"Periodic task {tool_name} succeeded but provided no feedback content.")

                except Exception as e:
                    # Log errors but continue with other periodic tasks
                    self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)

                # Small delay between periodic tasks if needed
                await asyncio.sleep(0.1)


    async def _trigger_promotion_checks(self):
        """Checks promotion criteria for recently accessed, eligible memories."""
        # (Keep existing v3.3.5 _trigger_promotion_checks implementation)
        # Ensure this method calls the integrated _check_and_trigger_promotion
        if not self.state.workflow_id:
             self.logger.debug("Skipping promotion check: No active workflow.")
             return

        self.logger.debug("Running periodic promotion check for recent memories...")
        query_tool_name = TOOL_QUERY_MEMORIES # Use query for broader filtering
        if not self._find_tool_server(query_tool_name):
            self.logger.warning(f"Skipping promotion check: Tool {query_tool_name} unavailable.")
            return

        candidate_memory_ids = set()
        try:
            # Check recent Episodic memories
            episodic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.EPISODIC.value,
                "sort_by": "last_accessed", # Prioritize recently accessed
                "sort_order": "DESC",
                "limit": 5, # Check top 5 recent/relevant
                "include_content": False # Don't need content for check
            }
            episodic_results = await self._execute_tool_call_internal(query_tool_name, episodic_args, record_action=False)
            if episodic_results.get("success"):
                mems = episodic_results.get("memories", [])
                if isinstance(mems, list):
                    candidate_memory_ids.update(m.get('memory_id') for m in mems if isinstance(m, dict) and m.get('memory_id'))

            # Check recent Semantic memories (potential promotion to Procedural)
            semantic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.SEMANTIC.value,
                 # Optionally filter by type if only certain semantic types can become procedural
                # "memory_type": MemoryType.PROCEDURE.value, # Or check both PROCEDURE and SKILL?
                "sort_by": "last_accessed",
                "sort_order": "DESC",
                "limit": 5,
                "include_content": False
            }
            semantic_results = await self._execute_tool_call_internal(query_tool_name, semantic_args, record_action=False)
            if semantic_results.get("success"):
                mems = semantic_results.get("memories", [])
                if isinstance(mems, list):
                     # Only consider PROCEDURE or SKILL types for promotion from SEMANTIC
                     candidate_memory_ids.update(
                          m.get('memory_id') for m in mems
                          if isinstance(m, dict) and m.get('memory_id') and
                          m.get('memory_type') in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]
                     )


            if candidate_memory_ids:
                self.logger.debug(f"Checking {len(candidate_memory_ids)} memories for potential promotion: {[_fmt_id(item_id) for item_id in candidate_memory_ids]}")
                # Create tasks to check each candidate in background
                promo_tasks = []
                for mem_id in candidate_memory_ids:
                     # Use _start_background_task to handle snapshotting workflow_id etc.
                     task = self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id)
                     promo_tasks.append(task)
                # Optionally await these checks if critical, or let them run truly in background
                # await asyncio.gather(*promo_tasks, return_exceptions=True) # Uncomment to wait
            else:
                self.logger.debug("No recently accessed, eligible memories found for promotion check.")
        except Exception as e:
            self.logger.error(f"Error during periodic promotion check query: {e}", exc_info=False)


    # ================================================================= main loop --
    # (Keep the integrated run method from the previous step here)
    async def run(self, goal: str, max_loops: int = 50) -> None:
        """Main agent execution loop, integrating rich context and refined plan updates."""
        if not await self.initialize():
            self.logger.critical("Agent initialization failed – aborting.")
            return

        self.logger.info(f"Starting main loop. Goal: '{goal}' Max Loops: {max_loops}", emoji_key="arrow_forward")
        self.state.goal_achieved_flag = False

        while (
            not self.state.goal_achieved_flag
            and self.state.current_loop < max_loops
            and not self._shutdown_event.is_set()
        ):
            self.state.current_loop += 1
            self.logger.info(f"--- Agent Loop {self.state.current_loop}/{max_loops} (RefThresh: {self.state.current_reflection_threshold}, ConThresh: {self.state.current_consolidation_threshold}) ---", emoji_key="arrows_counterclockwise")

            # ---------- Error Check ----------
            if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                self.logger.error(f"Max consecutive errors ({MAX_CONSECUTIVE_ERRORS}) reached. Aborting.", emoji_key="stop_sign")
                if self.state.workflow_id:
                    # Use internal helper to update status
                    await self._execute_tool_call_internal(
                        TOOL_UPDATE_WORKFLOW_STATUS,
                        {"status": "failed", "completion_message": "Agent failed due to repeated errors."},
                        record_action=False
                    )
                break # Exit loop

            # ---------- Gather Rich Context ----------
            context = await self._gather_context() # Use the enhanced version
            if context.get("status") == "No Active Workflow":
                self.logger.warning("No active workflow. Agent must create one.")
                # Reset plan to focus on creation
                self.state.current_plan = [PlanStep(description=f"Create the primary workflow for goal: {goal}")]
                self.state.needs_replan = False # Not a replan, it's initial plan
                self.state.last_error_details = None # Clear any previous errors
            elif "errors" in context and context.get("errors"):
                self.logger.warning(f"Context gathering encountered errors: {context['errors']}. Proceeding cautiously.")
                # Potentially set needs_replan = True if context errors are critical?
                # self.state.needs_replan = True

            # ---------- LLM Decision ----------
            # Ensure context is passed correctly
            agent_decision = await self._call_agent_llm(goal, context)
            decision_type = agent_decision.get("decision")

            # ---------- Act ----------
            last_res: Optional[Dict[str, Any]] = None
            plan_updated_by_tool = False # Flag if AGENT_TOOL_UPDATE_PLAN was called

            # Get Current Plan Step and Dependencies *before* executing
            current_plan_step: Optional[PlanStep] = self.state.current_plan[0] if self.state.current_plan else None
            planned_dependencies_for_step: Optional[List[str]] = current_plan_step.depends_on if current_plan_step else None

            if decision_type == "call_tool":
                tool_name = agent_decision.get("tool_name")
                arguments = agent_decision.get("arguments", {})
                if tool_name == AGENT_TOOL_UPDATE_PLAN:
                    # Handle the specific plan update tool call
                    self.logger.info(f"Agent requests plan update via tool: {AGENT_TOOL_UPDATE_PLAN}")
                    last_res = await self._execute_tool_call_internal(
                        tool_name, arguments, record_action=False # Don't record this internal agent action
                    )
                    if last_res.get("success"):
                        plan_updated_by_tool = True
                        self.logger.info("Plan successfully updated by agent tool.")
                        # Clear errors and replan flag after successful explicit plan update
                        self.state.consecutive_error_count = 0
                        self.state.needs_replan = False
                        self.state.last_error_details = None
                    else:
                        self.logger.error(f"Agent tool {AGENT_TOOL_UPDATE_PLAN} failed: {last_res.get('error')}")
                        self.state.needs_replan = True # Failed plan update needs replan
                elif tool_name:
                    # Handle regular tool calls
                    self.logger.info(f"Agent requests tool: {tool_name} with args: {arguments}", emoji_key="wrench")
                    last_res = await self._execute_tool_call_internal(
                        tool_name, arguments, True, planned_dependencies_for_step # Pass dependencies
                    )
                    # needs_replan flag is handled by _execute_tool_call_internal on failure/dependency issues
                    if isinstance(last_res, dict) and not last_res.get("success"):
                        self.logger.warning(f"Tool execution failed or prerequisites not met for {tool_name}.")
                        # Further action (like replan) handled by heuristic update based on last_error_details
                else:
                    # Handle case where decision was 'call_tool' but tool_name was missing
                    self.logger.error("LLM requested tool call but provided no tool name.")
                    self.state.last_action_summary = "Agent error: Missing tool name."
                    self.state.last_error_details = {"agent_decision_error": "Missing tool name"}
                    # self.state.consecutive_error_count += 1 # Let heuristic update handle count
                    self.state.needs_replan = True
                    last_res = {"success": False, "error": "Missing tool name from LLM"}


            elif decision_type == "thought_process":
                content = agent_decision.get("content", "No thought content.")
                self.logger.info(f"Agent reasoning: '{content[:100]}...'. Recording.", emoji_key="thought_balloon")
                if self.state.workflow_id:
                   thought_args = {"content": content, "thought_type": ThoughtType.INFERENCE.value}
                   # workflow_id and thought_chain_id are injected by _execute_tool_call_internal
                   last_res = await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, thought_args, True)
                   # If thought fails to record, set needs_replan?
                   if not last_res.get("success"):
                        self.state.needs_replan = True
                        self.logger.error(f"Failed to record thought: {last_res.get('error')}")
                else:
                    self.logger.warning("Cannot record thought: No active workflow.")
                    last_res = {"success": True} # Treat as success for plan progression

            elif decision_type == "complete":
                 summary = agent_decision.get("summary", "Goal achieved.")
                 self.logger.info(f"Agent signals completion: {summary}", emoji_key="tada")
                 self.state.goal_achieved_flag = True
                 self.state.needs_replan = False
                 last_res = {"success": True} # Mark as success
                 if self.state.workflow_id:
                      # Record a final summary thought
                      await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"content": f"Goal Achieved: {summary}", "thought_type": ThoughtType.SUMMARY.value}, False) # Don't record this meta-action
                      # Update workflow status
                      await self._execute_tool_call_internal(TOOL_UPDATE_WORKFLOW_STATUS, {"status": "completed", "completion_message": summary}, False)
                 break # Exit loop on completion

            elif decision_type == "error":
                 error_msg = agent_decision.get("message", "Unknown agent error")
                 self.logger.error(f"Agent decision error: {error_msg}", emoji_key="x")
                 self.state.last_action_summary = f"Agent decision error: {error_msg[:100]}"
                 self.state.last_error_details = {"agent_decision_error": error_msg}
                 # self.state.consecutive_error_count += 1 # Let heuristic handle count
                 self.state.needs_replan = True
                 last_res = {"success": False, "error": f"Agent decision error: {error_msg}"}
                 # Try to record this error as a thought
                 if self.state.workflow_id:
                     await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"content": f"Agent Error: {error_msg}", "thought_type": ThoughtType.CRITIQUE.value}, False)


            else: # Should not happen
                 self.logger.warning(f"Unhandled decision: {decision_type}")
                 self.state.last_action_summary = "Unknown agent decision."
                 # self.state.consecutive_error_count += 1 # Let heuristic handle count
                 self.state.needs_replan = True
                 self.state.last_error_details = {"agent_decision_error": f"Unknown type: {decision_type}"}
                 last_res = {"success": False, "error": f"Unhandled decision type: {decision_type}"}


            # ---------- Apply Heuristic Plan Update (if not done by agent tool) ----------
            if not plan_updated_by_tool:
                 # Pass the actual decision and result to the heuristic update function
                 await self._apply_heuristic_plan_update(agent_decision, last_res)


            # ---------- Periodic Tasks ----------
            await self._run_periodic_tasks()

            # ---------- Persistence (Maybe check if state *changed* before saving?) ----------
            if self.state.current_loop % 5 == 0: # Or use a time-based interval
                await self._save_agent_state()

            # --- Loop Delay ---
            await asyncio.sleep(random.uniform(0.8, 1.2))

        # --- End of Loop ---
        self.logger.info(f"--- Agent Loop Finished (Reason: {'Goal Achieved' if self.state.goal_achieved_flag else 'Max Loops Reached' if self.state.current_loop >= max_loops else 'Shutdown Signal' if self._shutdown_event.is_set() else 'Error Limit'}) ---", emoji_key="stopwatch")
        # Final save before shutdown
        await self._save_agent_state()
        if self.state.workflow_id and not self._shutdown_event.is_set() and self.state.consecutive_error_count < MAX_CONSECUTIVE_ERRORS:
            final_status = WorkflowStatus.COMPLETED.value if self.state.goal_achieved_flag else WorkflowStatus.FAILED.value # Or maybe INCOMPLETE? Let's use FAILED if goal not achieved.
            self.logger.info(f"Workflow {_fmt_id(self.state.workflow_id)} ended with status: {final_status}")
            # Don't automatically mark failed here, let the logic inside the loop handle status updates
            # await self._generate_final_report() # Generate report only on clean exit?
        elif not self.state.workflow_id:
            self.logger.info("Loop finished with no active workflow.")

        # Final cleanup called by external runner or signal handler via shutdown()


    async def _generate_final_report(self):
        """Generates and logs a final report using the memory tool."""
        # (Keep existing v3.3.5 _generate_final_report implementation)
        if not self.state.workflow_id:
            self.logger.info("Skipping final report: No active workflow ID.")
            return

        report_tool_name = TOOL_GENERATE_REPORT
        if not self._find_tool_server(report_tool_name):
            self.logger.error(f"Cannot generate final report: Tool '{report_tool_name}' unavailable.")
            return

        self.logger.info(f"Generating final report for workflow {_fmt_id(self.state.workflow_id)}...", emoji_key="scroll")
        try:
            # Request a professional markdown report by default
            report_args = {
                "workflow_id": self.state.workflow_id,
                "report_format": "markdown",
                "style": "professional",
                "include_details": True, # Include details by default for final report
                "include_thoughts": True,
                "include_artifacts": True,
            }
            report_result = await self._execute_tool_call_internal(
                report_tool_name, report_args, record_action=False # Don't record report generation
            )

            if isinstance(report_result, dict) and report_result.get("success"):
                report_text = report_result.get("report", "Report content missing.")
                # Print report clearly demarcated
                output_lines = [
                    "\n" + "="*30 + " FINAL WORKFLOW REPORT " + "="*30,
                    f"Workflow ID: {self.state.workflow_id}",
                    f"Generated at: {datetime.now(timezone.utc).isoformat()}",
                    "-"*80,
                    report_text,
                    "="*81 # Match top length
                ]
                # Use safe_print if available from MCP Client
                printer = getattr(self.mcp_client, 'safe_print', print)
                printer("\n".join(output_lines))
            else:
                error_msg = report_result.get('error', 'Unknown error') if isinstance(report_result, dict) else "Unexpected result type"
                self.logger.error(f"Failed to generate final report: {error_msg}")
        except Exception as e:
            self.logger.error(f"Exception generating final report: {e}", exc_info=True)


# =============================================================================
# Driver helpers & CLI entry‑point
# =============================================================================
# (Keep existing v3.3.5 driver code: run_agent_process, __main__ block)
async def run_agent_process(
    mcp_server_url: str,
    anthropic_key: str,
    goal: str,
    max_loops: int,
    state_file: str,
    config_file: Optional[str],
) -> None:
    """Sets up and runs the agent process, including signal handling."""
    if not MCP_CLIENT_AVAILABLE:
        print("❌ ERROR: MCPClient dependency not met.")
        sys.exit(1)

    mcp_client_instance = None
    agent_loop_instance = None
    exit_code = 0
    # Use print initially, switch to safe_print if available
    printer = print

    try:
        printer("Instantiating MCP Client...")
        # Pass base_url explicitly during instantiation if needed by your client
        mcp_client_instance = MCPClient(base_url=mcp_server_url, config_path=config_file)
        # Switch to safe_print if the client provides it
        if hasattr(mcp_client_instance, 'safe_print') and callable(mcp_client_instance.safe_print):
            printer = mcp_client_instance.safe_print
            log.info("Using MCPClient's safe_print for output.")

        # Configure API key if not already set
        if not mcp_client_instance.config.api_key:
            if anthropic_key:
                printer("Using provided Anthropic API key.")
                mcp_client_instance.config.api_key = anthropic_key
                # Re-initialize the anthropic client instance if necessary
                mcp_client_instance.anthropic = AsyncAnthropic(api_key=anthropic_key)
            else:
                printer("❌ CRITICAL ERROR: Anthropic API key missing in config and not provided.")
                raise ValueError("Anthropic API key missing.")

        printer("Setting up MCP Client connections...")
        # Setup might involve connecting to servers defined in config/discovery
        await mcp_client_instance.setup(interactive_mode=False)

        printer("Instantiating Agent Master Loop...")
        agent_loop_instance = AgentMasterLoop(mcp_client_instance=mcp_client_instance, agent_state_file=state_file)

        # --- Signal Handling Setup ---
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event() # Use an event for clearer signaling

        def signal_handler_wrapper(signum):
            signal_name = signal.Signals(signum).name
            log.warning(f"Signal {signal_name} received. Initiating graceful shutdown.")
            # Signal the main loop to stop via the event
            stop_event.set()
            # Also trigger the agent's internal shutdown logic if possible
            if agent_loop_instance:
                 # Schedule shutdown asynchronously, don't await here
                 asyncio.create_task(agent_loop_instance.shutdown())
            # Note: loop.stop() might be too abrupt, let the agent handle shutdown

        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                # Ensure handler isn't already attached before adding
                # This check might be platform specific or require internal loop state inspection
                # Simplified: Just add, assuming loop handles duplicates if necessary
                loop.add_signal_handler(sig, signal_handler_wrapper, sig)
                log.debug(f"Registered signal handler for {sig.name}")
            except ValueError:
                log.debug(f"Signal handler for {sig.name} may already be registered.")
            except NotImplementedError:
                log.warning(f"Signal handling for {sig.name} not supported on this platform.")


        printer(f"Running Agent Loop for goal: \"{goal}\"")
        # Main execution - replace with awaiting the run method directly if preferred
        # This structure allows checking the stop_event periodically if run isn't fully async-aware
        # await agent_loop_instance.run(goal=goal, max_loops=max_loops)
        # OR, if run needs to be interruptible by the signal event:
        run_task = asyncio.create_task(agent_loop_instance.run(goal=goal, max_loops=max_loops))
        stop_task = asyncio.create_task(stop_event.wait())

        # Wait for either the agent task to finish or the stop event to be set
        done, pending = await asyncio.wait(
            {run_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED
        )

        if stop_task in done:
             printer("\n[yellow]Shutdown signal processed. Waiting for agent task to finalize...[/yellow]")
             # Give run_task a chance to finish its cleanup via agent_loop_instance.shutdown()
             if run_task in pending:
                 run_task.cancel() # Attempt cancellation if still pending
                 try:
                     await run_task
                 except asyncio.CancelledError:
                      log.info("Agent run task cancelled gracefully.")
                 except Exception as e:
                      log.error(f"Exception during agent run task finalization after signal: {e}", exc_info=True)
             exit_code = 130 # Standard exit code for Ctrl+C

        elif run_task in done:
             # Agent finished normally or hit max loops/error limit
             try:
                 run_task.result() # Check for exceptions if run_task raised one
                 log.info("Agent run task completed normally.")
             except Exception as e:
                 printer(f"\n❌ Agent loop finished with error: {e}")
                 log.error("Agent run task finished with an exception:", exc_info=True)
                 exit_code = 1


    except KeyboardInterrupt:
        # Should be caught by signal handler now, but keep as fallback
        printer("\n[yellow]KeyboardInterrupt caught (fallback).[/yellow]")
        exit_code = 130
    except ValueError as ve: # Catch specific config errors
        printer(f"\n❌ Configuration Error: {ve}")
        exit_code = 2
    except Exception as main_err:
        printer(f"\n❌ Critical error during setup or execution: {main_err}")
        log.critical("Top-level execution error", exc_info=True)
        exit_code = 1
    finally:
        printer("Initiating final shutdown sequence...")
        # Agent shutdown should have been triggered by signal handler or normal completion
        # Ensure client is closed
        if mcp_client_instance:
            printer("Closing MCP client connections...")
            try:
                await mcp_client_instance.close()
            except Exception as close_err:
                printer(f"[red]Error closing MCP client:[/red] {close_err}")
        printer("Agent execution finished.")
        # Exit only if running as the main script
        if __name__ == "__main__":
            # Short delay to allow logs/output to flush
            await asyncio.sleep(0.5)
            sys.exit(exit_code)

if __name__ == "__main__":
    # (Keep existing __main__ block for configuration loading and running)
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8013")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    AGENT_GOAL = os.environ.get(
        "AGENT_GOAL",
        "Create workflow 'Tier 3 Test': Research Quantum Computing impact on Cryptography.",
    )
    MAX_ITER = int(os.environ.get("MAX_ITERATIONS", "30"))
    STATE_FILE = os.environ.get("AGENT_STATE_FILE", AGENT_STATE_FILE)
    CONFIG_PATH = os.environ.get("MCP_CLIENT_CONFIG")

    if not ANTHROPIC_API_KEY:
        print("❌ ERROR: ANTHROPIC_API_KEY missing in environment variables.")
        sys.exit(1)
    if not MCP_CLIENT_AVAILABLE:
        # This check happens earlier now, but good practice
        print("❌ ERROR: MCPClient dependency missing.")
        sys.exit(1)

    # Display configuration before starting
    print(f"--- {AGENT_NAME} ---")
    print(f"Memory System URL: {MCP_SERVER_URL}")
    print(f"Agent Goal: {AGENT_GOAL}")
    print(f"Max Iterations: {MAX_ITER}")
    print(f"State File: {STATE_FILE}")
    print(f"Client Config: {CONFIG_PATH or 'Default internal config'}")
    print(f"Log Level: {logging.getLevelName(log.level)}")
    print("Anthropic API Key: Found")
    print("-----------------------------------------")


    # Wrapper to run the main process
    async def _main() -> None:
        await run_agent_process(
            MCP_SERVER_URL,
            ANTHROPIC_API_KEY,
            AGENT_GOAL,
            MAX_ITER,
            STATE_FILE,
            CONFIG_PATH,
        )

    # Run the async main function
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        # This might catch Ctrl+C during initial setup before signal handlers are active
        print("\n[yellow]Initial KeyboardInterrupt detected. Exiting.[/yellow]")
        sys.exit(130)
