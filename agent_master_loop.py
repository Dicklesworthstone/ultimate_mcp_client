"""
Supercharged Agent Master Loop - v3.3 (Tiers 1, 2 & 3 Integration - Complete)
===============================================================================

Enhanced orchestrator for AI agents using the Unified Memory System
via the Ultimate MCP Client. Implements structured planning, dynamic context,
dependency checking, artifact tracking, error recovery, feedback loops,
meta-cognition, richer auto-linking, hybrid search, direct memory management,
working memory context, **adaptive thresholds**, **memory maintenance**,
and **custom thought chain management**.

Designed for Claude 3.7 Sonnet (or comparable models with tool use).
"""

import asyncio
import dataclasses
import json
import logging
import math  # For adaptive threshold adjustments
import os
import random
import re
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# External Libraries
import aiofiles
import anthropic
from anthropic.types import AsyncAnthropic, Message
from pydantic import BaseModel, Field, ValidationError

# --- IMPORT YOUR ACTUAL MCP CLIENT and COMPONENTS ---
try:
    from mcp_client import (
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
        WorkflowStatus,
    )
    MCP_CLIENT_AVAILABLE = True
    log = logging.getLogger("SuperchargedAgentMasterLoop") # Use project logger if MCPClient sets it up
    if not log.handlers:
        # Basic fallback logger if MCPClient didn't configure it
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log = logging.getLogger("SuperchargedAgentMasterLoop")
        log.warning("MCPClient did not configure logger, using basic fallback.")
    log.info("Successfully imported MCPClient and required components.")
except ImportError as import_err:
    print(f"❌ CRITICAL ERROR: Could not import MCPClient or required components: {import_err}")
    print("Ensure mcp_client.py is correctly structured and in the Python path.")
    sys.exit(1)

# --- Logging Setup Refinement ---
log_level_str = os.environ.get("AGENT_LOOP_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
log.setLevel(log_level)
if log_level <= logging.DEBUG: # Use <= DEBUG to include DEBUG level
    log.info("Agent loop verbose logging enabled.")

# --- Constants ---
# File Paths & Identifiers
AGENT_STATE_FILE = "agent_loop_state_v3.3.json" # Updated state file version
AGENT_NAME = "Maestro-v3.3" # Updated agent name
# --- Meta-cognition & Maintenance Intervals/Thresholds ---
# Base Thresholds (These become the initial values and bounds)
BASE_REFLECTION_THRESHOLD = int(os.environ.get("BASE_REFLECTION_THRESHOLD", "7"))
BASE_CONSOLIDATION_THRESHOLD = int(os.environ.get("BASE_CONSOLIDATION_THRESHOLD", "12"))
# Adaptive Threshold Bounds
MIN_REFLECTION_THRESHOLD = 3
MAX_REFLECTION_THRESHOLD = 15
MIN_CONSOLIDATION_THRESHOLD = 5
MAX_CONSOLIDATION_THRESHOLD = 25
# Intervals
OPTIMIZATION_LOOP_INTERVAL = int(os.environ.get("OPTIMIZATION_INTERVAL", "8"))
MEMORY_PROMOTION_LOOP_INTERVAL = int(os.environ.get("PROMOTION_INTERVAL", "15"))
STATS_ADAPTATION_INTERVAL = int(os.environ.get("STATS_ADAPTATION_INTERVAL", "10")) # How often to check stats/adapt
MAINTENANCE_INTERVAL = int(os.environ.get("MAINTENANCE_INTERVAL", "50")) # How often to run cleanup
# Other
AUTO_LINKING_DELAY_SECS = (1.5, 3.0)
# Context & Planning
DEFAULT_PLAN_STEP = "Assess goal, gather context, formulate initial plan."
CONTEXT_RECENT_ACTIONS = 7
CONTEXT_IMPORTANT_MEMORIES = 5
CONTEXT_KEY_THOUGHTS = 5
CONTEXT_PROCEDURAL_MEMORIES = 2
CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD = 15000
CONTEXT_COMPRESSION_TARGET_TOKENS = 5000
CONTEXT_PROACTIVE_MEMORIES = 3
CONTEXT_WORKING_MEMORY_LIMIT = 10
# Error Handling
MAX_CONSECUTIVE_ERRORS = 3
# --- Tool Names (Includes Tier 1, 2 & 3) ---
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context"
TOOL_CREATE_WORKFLOW = "unified_memory:create_workflow"
TOOL_UPDATE_WORKFLOW_STATUS = "unified_memory:update_workflow_status"
TOOL_RECORD_ACTION_START = "unified_memory:record_action_start"
TOOL_RECORD_ACTION_COMPLETION = "unified_memory:record_action_completion"
TOOL_GET_ACTION_DETAILS = "unified_memory:get_action_details"
# Action Dependency Tools (Tier 1)
TOOL_ADD_ACTION_DEPENDENCY = "unified_memory:add_action_dependency"
TOOL_GET_ACTION_DEPENDENCIES = "unified_memory:get_action_dependencies"
# Artifact Tracking Tools (Tier 1)
TOOL_RECORD_ARTIFACT = "unified_memory:record_artifact"
TOOL_GET_ARTIFACTS = "unified_memory:get_artifacts"
TOOL_GET_ARTIFACT_BY_ID = "unified_memory:get_artifact_by_id"
# Core Memory Tools (Tier 2 additions)
TOOL_HYBRID_SEARCH = "unified_memory:hybrid_search_memories"
TOOL_STORE_MEMORY = "unified_memory:store_memory"
TOOL_UPDATE_MEMORY = "unified_memory:update_memory"
TOOL_GET_WORKING_MEMORY = "unified_memory:get_working_memory"
# Custom Thought Chain Tools (Tier 3)
TOOL_CREATE_THOUGHT_CHAIN = "unified_memory:create_thought_chain"
TOOL_GET_THOUGHT_CHAIN = "unified_memory:get_thought_chain"
# Maintenance Tool (Tier 3)
TOOL_DELETE_EXPIRED_MEMORIES = "unified_memory:delete_expired_memories"
# Statistics Tool (Tier 3)
TOOL_COMPUTE_STATS = "unified_memory:compute_memory_statistics"
# Other Memory Tools
TOOL_RECORD_THOUGHT = "unified_memory:record_thought"
TOOL_REFLECTION = "unified_memory:generate_reflection"
TOOL_CONSOLIDATION = "unified_memory:consolidate_memories"
TOOL_OPTIMIZE_WM = "unified_memory:optimize_working_memory"
TOOL_AUTO_FOCUS = "unified_memory:auto_update_focus"
TOOL_PROMOTE_MEM = "unified_memory:promote_memory_level"
TOOL_QUERY_MEMORIES = "unified_memory:query_memories"
TOOL_SEMANTIC_SEARCH = "unified_memory:search_semantic_memories"
TOOL_CREATE_LINK = "unified_memory:create_memory_link"
TOOL_GET_MEMORY_BY_ID = "unified_memory:get_memory_by_id"
TOOL_GET_LINKED_MEMORIES = "unified_memory:get_linked_memories"
TOOL_LIST_WORKFLOWS = "unified_memory:list_workflows"
TOOL_GENERATE_REPORT = "unified_memory:generate_workflow_report"
TOOL_SUMMARIZE_TEXT = "unified_memory:summarize_text"


# --- Structured Plan Model ---
class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: f"step-{MemoryUtils.generate_id()[:8]}")
    description: str
    status: str = Field(default="planned", description="Status: planned, in_progress, completed, failed, skipped")
    depends_on: List[str] = Field(default_factory=list, description="List of action IDs this step requires")
    assigned_tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    is_parallel_group: Optional[str] = None

# --- Agent State Dataclass (Added Tier 3 State) ---
def _default_tool_stats():
    return defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0})

@dataclass
class AgentState:
    # Core State
    workflow_id: Optional[str] = None
    context_id: Optional[str] = None
    workflow_stack: List[str] = field(default_factory=list)
    current_plan: List[PlanStep] = field(default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)])
    current_sub_goal_id: Optional[str] = None # ID for the active goal/sub-goal
    current_thought_chain_id: Optional[str] = None # Track active thought chain (Tier 3)
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False
    # Error & Replanning State
    consecutive_error_count: int = 0
    needs_replan: bool = False
    last_error_details: Optional[Dict] = None
    # Meta-Cognition State
    successful_actions_since_reflection: int = 0
    successful_actions_since_consolidation: int = 0
    loops_since_optimization: int = 0
    loops_since_promotion_check: int = 0
    loops_since_stats_adaptation: int = 0 # Tier 3
    loops_since_maintenance: int = 0 # Tier 3
    reflection_cycle_index: int = 0
    last_meta_feedback: Optional[str] = None
    # --- Adaptive Thresholds (Tier 3) ---
    current_reflection_threshold: int = BASE_REFLECTION_THRESHOLD
    current_consolidation_threshold: int = BASE_CONSOLIDATION_THRESHOLD
    # Stats & Tracking
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=_default_tool_stats)
    # Background task tracking (transient, not saved)
    background_tasks: Set[asyncio.Task] = field(default_factory=set, init=False, repr=False)


# --- Agent Loop Class (Modified for Tier 1, 2 & 3) ---
class AgentMasterLoop:
    """Supercharged orchestrator implementing Tier 1, 2 & 3 UMS enhancements."""

    def __init__(self, mcp_client_instance: MCPClient, agent_state_file: str = AGENT_STATE_FILE):
        if not MCP_CLIENT_AVAILABLE: raise RuntimeError("MCPClient class unavailable.")

        self.mcp_client = mcp_client_instance
        self.anthropic_client = self.mcp_client.anthropic
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        # Config attributes (Base values, dynamic ones are in state)
        self.consolidation_memory_level = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources = 10
        self.auto_linking_threshold = 0.7
        self.auto_linking_max_links = 3
        self.reflection_type_sequence = ["summary", "progress", "gaps", "strengths", "plan"]

        if not self.anthropic_client:
            self.logger.critical("Anthropic client unavailable! Agent cannot function.")
            raise ValueError("Anthropic client required.")

        self.state = AgentState() # Initialize state here
        self._shutdown_event = asyncio.Event()
        self.tool_schemas: List[Dict[str, Any]] = []
        self._active_tasks: Set[asyncio.Task] = set()

    async def initialize(self) -> bool:
        """Initializes loop state, loads previous state, verifies client setup, including Tier 1, 2 & 3 tools."""
        self.logger.info("Initializing agent loop...", emoji_key="gear")
        await self._load_agent_state()
        if self.state.workflow_id and not self.state.context_id:
            self.state.context_id = self.state.workflow_id
            self.logger.info(f"Set context_id to match loaded workflow_id: {self.state.workflow_id}")

        try:
            if not self.mcp_client.server_manager:
                self.logger.error("MCP Client Server Manager not initialized.")
                return False

            # Fetch and filter tool schemas
            all_tools_for_api = self.mcp_client.server_manager.format_tools_for_anthropic()
            self.tool_schemas = [
                schema for schema in all_tools_for_api
                if self.mcp_client.server_manager.sanitized_to_original.get(schema['name'], '').startswith("unified_memory:")
            ]
            loaded_tool_names = {self.mcp_client.server_manager.sanitized_to_original.get(s['name']) for s in self.tool_schemas}
            self.logger.info(f"Loaded {len(self.tool_schemas)} unified_memory tool schemas: {loaded_tool_names}", emoji_key="clipboard")

            # Verify essential tools (Added Tier 1, 2 & 3)
            essential_tools = [
                TOOL_GET_CONTEXT, TOOL_CREATE_WORKFLOW, TOOL_RECORD_THOUGHT,
                TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION,
                TOOL_ADD_ACTION_DEPENDENCY, TOOL_RECORD_ARTIFACT, TOOL_GET_ACTION_DETAILS,
                TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY, TOOL_GET_WORKING_MEMORY,
                TOOL_HYBRID_SEARCH,
                TOOL_CREATE_THOUGHT_CHAIN, TOOL_GET_THOUGHT_CHAIN, # Tier 3
                TOOL_COMPUTE_STATS, TOOL_DELETE_EXPIRED_MEMORIES # Tier 3
            ]
            missing_essential = [t for t in essential_tools if not self._find_tool_server(t)]
            if missing_essential:
                self.logger.error(f"Missing essential tools: {missing_essential}. Agent functionality WILL BE impaired.")

            # Check workflow validity
            current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
            if current_workflow_id and not await self._check_workflow_exists(current_workflow_id):
                self.logger.warning(f"Loaded workflow {current_workflow_id} not found. Resetting state.")
                await self._reset_state_to_defaults()
                await self._save_agent_state()

            # Initialize current thought chain ID if workflow exists but chain ID is missing
            if self.state.workflow_id and not self.state.current_thought_chain_id:
                 await self._set_default_thought_chain_id()

            self.logger.info("Agent loop initialized successfully.")
            return True
        except Exception as e:
            self.logger.critical(f"Agent loop initialization failed: {e}", exc_info=True)
            return False

    async def _set_default_thought_chain_id(self):
         """Sets the current_thought_chain_id to the primary chain of the current workflow."""
         current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
         if not current_wf_id: return # No workflow active

         # Use the get_workflow_details tool to find the primary chain
         get_details_tool = "unified_memory:get_workflow_details" # Use the correct tool name constant if defined, or the string directly

         # Check if the tool is available
         if self._find_tool_server(get_details_tool):
             try:
                 details = await self._execute_tool_call_internal(
                     get_details_tool,
                     {
                         "workflow_id": current_wf_id,
                         "include_thoughts": True, # Need thoughts to get chain ID
                         "include_actions": False,
                         "include_artifacts": False,
                         "include_memories": False
                     },
                     record_action=False
                 )
                 # Check successful execution AND if thought_chains exist in the result
                 if details.get("success") and isinstance(details.get("thought_chains"), list) and details["thought_chains"]:
                     # Assume the first chain listed is the primary one
                     first_chain = details["thought_chains"][0]
                     chain_id = first_chain.get("thought_chain_id")
                     if chain_id:
                         self.state.current_thought_chain_id = chain_id
                         self.logger.info(f"Set current_thought_chain_id to primary chain: {self.state.current_thought_chain_id}")
                         return # Success

                 self.logger.warning(f"Could not find primary thought chain in details for workflow {current_wf_id}. Using default logic.")

             except Exception as e:
                 self.logger.error(f"Error fetching workflow details to set default thought chain ID: {e}", exc_info=False)
         else:
             self.logger.warning(f"Cannot set default thought chain ID: Tool '{get_details_tool}' unavailable.")

         # Fallback message if chain couldn't be set
         self.logger.info("Could not determine primary thought chain ID on init/load. Will use default on first thought recording.")

    async def _estimate_tokens_anthropic(self, data: Any) -> int:
        """Estimates token count for arbitrary data structures using the Anthropic client."""
        if data is None: return 0
        if not self.anthropic_client:
             self.logger.warning("Cannot estimate tokens: Anthropic client not available.")
             try: return len(json.dumps(data, default=str)) // 4
             except Exception: return 0
        token_count = 0
        try:
            if isinstance(data, str): text_representation = data
            else: text_representation = json.dumps(data, ensure_ascii=False, default=str)
            token_count = await self.anthropic_client.count_tokens(text_representation)
            return token_count
        except anthropic.APIError as e: self.logger.warning(f"Anthropic API error during token counting: {e}. Using fallback estimate.")
        except Exception as e: self.logger.warning(f"Token estimation failed for data type {type(data)}: {e}. Using fallback estimate.")
        try:
             text_representation = json.dumps(data, default=str) if not isinstance(data, str) else data
             return len(text_representation) // 4
        except Exception: return 0

    async def _save_agent_state(self):
        """Saves the agent loop's state to a JSON file."""
        state_dict = dataclasses.asdict(self.state)
        state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        state_dict.pop("background_tasks", None)
        state_dict["tool_usage_stats"] = {k: dict(v) for k, v in self.state.tool_usage_stats.items()}
        state_dict["current_plan"] = [step.model_dump() for step in self.state.current_plan]
        try:
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.agent_state_file, 'w') as f: await f.write(json.dumps(state_dict, indent=2))
            self.logger.debug(f"Agent state saved to {self.agent_state_file}")
        except Exception as e: self.logger.error(f"Failed to save agent state: {e}", exc_info=True)

    async def _load_agent_state(self):
        """Loads state, converting plan back to PlanStep objects and setting dynamic thresholds."""
        if not self.agent_state_file.exists():
            self.logger.info("No previous agent state file found. Using default state.")
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            ) # Initialize dynamic thresholds
            return
        try:
            async with aiofiles.open(self.agent_state_file, 'r') as f: state_data = json.loads(await f.read())
            kwargs = {}
            for field_info in dataclasses.fields(AgentState):
                 if field_info.name in state_data:
                     if field_info.name == "current_plan":
                         try: kwargs["current_plan"] = [PlanStep(**step_data) for step_data in state_data["current_plan"]]
                         except (ValidationError, TypeError) as plan_err: log.warning(f"Failed to parse saved plan, resetting: {plan_err}"); kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                     elif field_info.name == "tool_usage_stats":
                          stats_dict = state_data["tool_usage_stats"]
                          recreated_stats = defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0})
                          if isinstance(stats_dict, dict):
                              for k, v in stats_dict.items():
                                  if isinstance(v, dict): recreated_stats[k] = {"success": v.get("success", 0), "failure": v.get("failure", 0), "latency_ms_total": v.get("latency_ms_total", 0.0)}
                          kwargs["tool_usage_stats"] = recreated_stats
                     else: kwargs[field_info.name] = state_data[field_info.name]
                 else:
                     # Initialize dynamic thresholds if not present in saved state
                     if field_info.name == "current_reflection_threshold": kwargs[field_info.name] = BASE_REFLECTION_THRESHOLD
                     elif field_info.name == "current_consolidation_threshold": kwargs[field_info.name] = BASE_CONSOLIDATION_THRESHOLD
                     elif field_info.default_factory is not dataclasses.MISSING: kwargs[field_info.name] = field_info.default_factory()
                     elif field_info.default is not dataclasses.MISSING: kwargs[field_info.name] = field_info.default

            # Ensure dynamic thresholds exist even if loading older state file
            if "current_reflection_threshold" not in kwargs: kwargs["current_reflection_threshold"] = BASE_REFLECTION_THRESHOLD
            if "current_consolidation_threshold" not in kwargs: kwargs["current_consolidation_threshold"] = BASE_CONSOLIDATION_THRESHOLD


            self.state = AgentState(**kwargs)
            self.logger.info(f"Agent state loaded from {self.agent_state_file}. Loop {self.state.current_loop}. WF: {self.state.workflow_id}. Dyn Thresh: R={self.state.current_reflection_threshold}/C={self.state.current_consolidation_threshold}")
        except Exception as e: self.logger.error(f"Failed to load/parse agent state: {e}. Resetting.", exc_info=True); await self._reset_state_to_defaults()

    async def _reset_state_to_defaults(self):
        self.state = AgentState(
             current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
             current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
        )
        self.logger.warning("Agent state has been reset to defaults.")

    # --- Context Gathering (Enhanced for Tier 2) ---
    async def _gather_context(self) -> Dict[str, Any]:
        """Gathers comprehensive context for the agent LLM, including working memory and using hybrid search."""
        self.logger.info("Gathering context...", emoji_key="satellite")
        base_context = {
            "current_loop": self.state.current_loop,
            "current_plan": [step.model_dump() for step in self.state.current_plan],
            "last_action_summary": self.state.last_action_summary,
            "consecutive_errors": self.state.consecutive_error_count,
            "last_error_details": self.state.last_error_details,
            "workflow_stack": self.state.workflow_stack,
            "meta_feedback": self.state.last_meta_feedback,
            "current_thought_chain_id": self.state.current_thought_chain_id, # Added Tier 3
            "core_context": None,
            "current_working_memory": [], # Added Tier 2
            "proactive_memories": [],
            "relevant_procedures": [],
            "contextual_links": None,
            "compression_summary": None,
            "status": "Gathering...",
            "errors": []
        }
        self.state.last_meta_feedback = None

        current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_context_id = self.state.context_id

        if not current_workflow_id:
            base_context["status"] = "No Active Workflow"; base_context["message"] = "Create/load workflow."; return base_context

        # --- 0. Get Current Working Memory (Tier 2) ---
        if current_context_id and self._find_tool_server(TOOL_GET_WORKING_MEMORY):
             try:
                 wm_result = await self._execute_tool_call_internal(TOOL_GET_WORKING_MEMORY, {"context_id": current_context_id, "include_content": False, "include_links": False}, record_action=False)
                 if wm_result.get("success"):
                     wm_mems = wm_result.get("working_memories", []); base_context["current_working_memory"] = [{"memory_id": m.get("memory_id"), "description": m.get("description"), "type": m.get("memory_type"), "importance": m.get("importance")} for m in wm_mems[:CONTEXT_WORKING_MEMORY_LIMIT]]; self.logger.info(f"Retrieved {len(base_context['current_working_memory'])} items from working memory for context {current_context_id}.")
                 else: base_context["errors"].append(f"Working memory retrieval failed: {wm_result.get('error')}")
             except Exception as e: self.logger.warning(f"Working memory retrieval exception: {e}"); base_context["errors"].append(f"Working memory retrieval exception: {e}")
        elif not current_context_id: self.logger.debug("Skipping working memory retrieval: No context_id set.")
        else: self.logger.debug(f"Skipping working memory retrieval: Tool '{TOOL_GET_WORKING_MEMORY}' unavailable.")

        # --- 1. Goal-Directed Proactive Memory Retrieval (Using Hybrid Search - Tier 2) ---
        active_plan_step_desc = self.state.current_plan[0].description if self.state.current_plan else "Achieve main goal"
        proactive_query = f"Information relevant to planning or executing: {active_plan_step_desc}"
        search_tool_proactive = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
        if self._find_tool_server(search_tool_proactive):
            search_args = {"workflow_id": current_workflow_id, "query": proactive_query, "limit": CONTEXT_PROACTIVE_MEMORIES, "include_content": False}
            if search_tool_proactive == TOOL_HYBRID_SEARCH: search_args.update({"semantic_weight": 0.7, "keyword_weight": 0.3})
            try:
                result_content = await self._execute_tool_call_internal(search_tool_proactive, search_args, record_action=False)
                if result_content.get("success"):
                    proactive_mems = result_content.get("memories", []); score_key = "hybrid_score" if search_tool_proactive == TOOL_HYBRID_SEARCH else "similarity"
                    base_context["proactive_memories"] = [{"memory_id": m.get("memory_id"), "description": m.get("description"), "score": m.get(score_key), "type": m.get("memory_type")} for m in proactive_mems]
                    if base_context["proactive_memories"]: self.logger.info(f"Retrieved {len(base_context['proactive_memories'])} proactive memories using {search_tool_proactive.split(':')[-1]}.")
                else: base_context["errors"].append(f"Proactive memory search failed: {result_content.get('error')}")
            except Exception as e: self.logger.warning(f"Proactive memory search exception: {e}"); base_context["errors"].append(f"Proactive search exception: {e}")
        else: self.logger.warning("Skipping proactive memory search: No suitable search tool available.")

        # --- 2. Fetch Core Context via Tool ---
        if self._find_tool_server(TOOL_GET_CONTEXT):
            try:
                core_context_result = await self._execute_tool_call_internal(TOOL_GET_CONTEXT, {"workflow_id": current_workflow_id, "recent_actions_limit": CONTEXT_RECENT_ACTIONS, "important_memories_limit": CONTEXT_IMPORTANT_MEMORIES, "key_thoughts_limit": CONTEXT_KEY_THOUGHTS}, record_action=False)
                if core_context_result.get("success"): base_context["core_context"] = core_context_result; base_context["core_context"].pop("success", None); base_context["core_context"].pop("processing_time", None); self.logger.info("Core context retrieved.")
                else: base_context["errors"].append(f"Core context retrieval failed: {core_context_result.get('error')}")
            except Exception as e: self.logger.warning(f"Core context retrieval exception: {e}"); base_context["errors"].append(f"Core context exception: {e}")
        else: self.logger.warning(f"Skipping core context retrieval: Tool '{TOOL_GET_CONTEXT}' unavailable.")

        # --- 3. Fetch Relevant Procedural Memories (Using Hybrid Search - Tier 2) ---
        search_tool_proc = TOOL_HYBRID_SEARCH if self._find_tool_server(TOOL_HYBRID_SEARCH) else TOOL_SEMANTIC_SEARCH
        if self._find_tool_server(search_tool_proc):
             proc_query = f"How to accomplish: {active_plan_step_desc}"
             search_args = {"workflow_id": current_workflow_id, "query": proc_query, "limit": CONTEXT_PROCEDURAL_MEMORIES, "memory_level": MemoryLevel.PROCEDURAL.value, "include_content": False}
             if search_tool_proc == TOOL_HYBRID_SEARCH: search_args.update({"semantic_weight": 0.6, "keyword_weight": 0.4})
             try:
                 proc_result = await self._execute_tool_call_internal(search_tool_proc, search_args, record_action=False)
                 if proc_result.get("success"):
                     proc_mems = proc_result.get("memories", []); score_key = "hybrid_score" if search_tool_proc == TOOL_HYBRID_SEARCH else "similarity"
                     base_context["relevant_procedures"] = [{"memory_id": m.get("memory_id"), "description": m.get("description"), "score": m.get(score_key)} for m in proc_mems]
                     if base_context["relevant_procedures"]: self.logger.info(f"Retrieved {len(base_context['relevant_procedures'])} relevant procedures using {search_tool_proc.split(':')[-1]}.")
                 else: base_context["errors"].append(f"Procedure search failed: {proc_result.get('error')}")
             except Exception as e: self.logger.warning(f"Procedure search exception: {e}"); base_context["errors"].append(f"Procedure search exception: {e}")
        else: self.logger.warning("Skipping procedure search: No suitable search tool available.")

        # --- 4. Context Compression (Check) ---
        try:
            estimated_tokens = await self._estimate_tokens_anthropic(base_context)
            if estimated_tokens > CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD:
                self.logger.warning(f"Context ({estimated_tokens} tokens) exceeds threshold {CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD}. Attempting compression.")
                if self._find_tool_server(TOOL_SUMMARIZE_TEXT):
                    actions_text = json.dumps(base_context.get("core_context", {}).get("recent_actions", []), indent=2, default=str)
                    if len(actions_text) > 500:
                        summary_result = await self._execute_tool_call_internal( TOOL_SUMMARIZE_TEXT, {"text_to_summarize": actions_text, "target_tokens": CONTEXT_COMPRESSION_TARGET_TOKENS, "workflow_id": current_workflow_id, "record_summary": False}, record_action=False)
                        if summary_result.get("success"):
                            base_context["compression_summary"] = f"Summary of recent actions: {summary_result.get('summary', 'Summary failed.')[:150]}..."
                            if base_context.get("core_context"): base_context["core_context"].pop("recent_actions", None)
                            self.logger.info(f"Compressed recent actions. New context size: {await self._estimate_tokens_anthropic(base_context)} est. tokens")
                        else: base_context["errors"].append(f"Context compression failed: {summary_result.get('error')}")
                else: self.logger.warning(f"Cannot compress context: Tool '{TOOL_SUMMARIZE_TEXT}' unavailable.")
        except Exception as e: self.logger.error(f"Error during context compression check: {e}", exc_info=False); base_context["errors"].append(f"Compression exception: {e}")

        # --- 5. Contextual Link Traversal ---
        base_context["contextual_links"] = None
        get_linked_memories_tool = TOOL_GET_LINKED_MEMORIES
        if self._find_tool_server(get_linked_memories_tool):
            mem_id_to_traverse = None
            # Prioritize focus memory from working memory if available
            wm_list = base_context.get("current_working_memory", [])
            if wm_list: mem_id_to_traverse = wm_list[0].get("memory_id") # Simple: pick first WM item
            # Fallback to important memories
            if not mem_id_to_traverse:
                 important_mem_list = base_context.get("core_context", {}).get("important_memories", [])
                 if important_mem_list and isinstance(important_mem_list, list) and len(important_mem_list) > 0:
                     first_mem = important_mem_list[0]
                     if isinstance(first_mem, dict): mem_id_to_traverse = first_mem.get("memory_id")

            if mem_id_to_traverse:
                self.logger.debug(f"Attempting link traversal from relevant memory: {mem_id_to_traverse[:8]}...")
                try:
                    links_result_content = await self._execute_tool_call_internal(get_linked_memories_tool, {"memory_id": mem_id_to_traverse, "direction": "both", "limit": 3}, record_action=False)
                    if links_result_content.get("success"):
                        links_data = links_result_content.get("links", {}); outgoing_links = links_data.get("outgoing", []); incoming_links = links_data.get("incoming", [])
                        link_summary = {"source_memory_id": mem_id_to_traverse, "outgoing_count": len(outgoing_links), "incoming_count": len(incoming_links), "top_links_summary": []}
                        for link in outgoing_links[:2]: link_summary["top_links_summary"].append(f"OUT: {link.get('link_type', 'related')} -> {link.get('target_type','Mem')} '{str(link.get('target_description','?'))[:30]}...' (ID: {str(link.get('target_memory_id','?'))[:6]}...)")
                        for link in incoming_links[:2]: link_summary["top_links_summary"].append(f"IN: {link.get('link_type', 'related')} <- {link.get('source_type','Mem')} '{str(link.get('source_description','?'))[:30]}...' (ID: {str(link.get('source_memory_id','?'))[:6]}...)")
                        base_context["contextual_links"] = link_summary
                        self.logger.info(f"Retrieved {len(outgoing_links)} outgoing, {len(incoming_links)} incoming links for memory {mem_id_to_traverse[:8]}...")
                    else: err_msg = f"Link retrieval tool failed: {links_result_content.get('error', 'Unknown')}"; base_context["errors"].append(err_msg); self.logger.warning(err_msg)
                except Exception as e: err_msg = f"Link retrieval exception: {e}"; self.logger.warning(err_msg, exc_info=False); base_context["errors"].append(err_msg)
            else: self.logger.debug("No relevant memory found (working or important) to perform link traversal from.")
        else: self.logger.debug(f"Skipping link traversal: Tool '{get_linked_memories_tool}' unavailable.")

        base_context["status"] = "Ready" if not base_context["errors"] else "Ready with Errors"
        return base_context

    # --- Prompt Construction (Updated for Tier 3 Tools & Context) ---
    def _construct_agent_prompt(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Constructs the prompt for the LLM, including Tier 1, 2 & 3 tools and instructions."""
        system_prompt = f"""You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System. Achieve the Overall Goal by strategically using the provided memory tools.

Overall Goal: {goal}

Available Unified Memory Tools (Use ONLY these):
"""
        # Add tool descriptions to prompt
        if not self.tool_schemas: system_prompt += "- CRITICAL WARNING: No tools loaded.\n"
        else:
             for schema in self.tool_schemas:
                 sanitized = schema['name']; original = self.mcp_client.server_manager.sanitized_to_original.get(sanitized, 'Unknown')
                 # Highlight Tier 1, 2 & 3 tools
                 is_new_or_essential = original in [
                     TOOL_ADD_ACTION_DEPENDENCY, TOOL_GET_ACTION_DEPENDENCIES,
                     TOOL_RECORD_ARTIFACT, TOOL_GET_ARTIFACTS, TOOL_GET_ARTIFACT_BY_ID,
                     TOOL_CREATE_LINK, TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION,
                     TOOL_RECORD_THOUGHT, TOOL_GET_ACTION_DETAILS,
                     TOOL_HYBRID_SEARCH, TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY,
                     TOOL_GET_WORKING_MEMORY,
                     TOOL_CREATE_THOUGHT_CHAIN, TOOL_GET_THOUGHT_CHAIN, # Tier 3
                     TOOL_COMPUTE_STATS, TOOL_DELETE_EXPIRED_MEMORIES # Tier 3
                 ]
                 prefix = "**" if is_new_or_essential else ""
                 system_prompt += f"\n- {prefix}Name: `{sanitized}` (Represents: `{original}`){prefix}\n"
                 system_prompt += f"  Desc: {schema.get('description', 'N/A')}\n"; system_prompt += f"  Schema: {json.dumps(schema['input_schema'])}\n"

        # Add Tier 1, 2 & 3 Instructions
        system_prompt += """
Your Process:
1.  Context Analysis: Deeply analyze 'Current Context'. Note workflow status, errors (`last_error_details`), recent actions, memories (`core_context`, `proactive_memories`), thoughts, `current_plan`, `relevant_procedures`, `current_working_memory` (most active memories), `current_thought_chain_id`, and `meta_feedback`. Pay attention to memory `importance`/`confidence`.
2.  Error Handling: If `last_error_details` exists, **FIRST** reason about the error and propose a recovery strategy in your Reasoning & Planning step. Check if it was a dependency failure.
3.  Reasoning & Planning:
    a. State step-by-step reasoning towards the Goal/Sub-goal, integrating context and feedback. Consider `current_working_memory` for immediate context. Record thoughts using `record_thought` and specify the `thought_chain_id` if different from `current_thought_chain_id`.
    b. Evaluate `current_plan`. Is it valid? Does it address errors? Are dependencies (`depends_on`) likely met?
    c. **Action Dependencies:** If planning Step B requires output from Step A (action ID 'a123'), include `"depends_on": ["a123"]` in Step B's plan object.
    d. **Artifact Tracking:** If planning to use a tool that creates a file/data, plan a subsequent step to call `record_artifact`. If needing a previously created artifact, plan to use `get_artifacts` or `get_artifact_by_id` first.
    e. **Direct Memory Management:** If you synthesize a critical new fact, insight, or piece of knowledge, plan to use `store_memory` to explicitly save it. If you find strong evidence contradicting a stored memory, plan to use `update_memory` to correct it. Provide clear `content`, `memory_type`, `importance`, and `confidence`.
    f. **Custom Thought Chains:** If tackling a distinct sub-problem or exploring a complex tangent, consider creating a new reasoning thread using `create_thought_chain`. Provide a clear `title`. Subsequent related thoughts should specify the new `thought_chain_id`. The loop will automatically track the `current_thought_chain_id`.
    g. **Linking:** Identify potential memory relationships (causal, supportive, contradictory). Plan to use `create_memory_link` with specific `link_type`s.
    h. **Search:** Prefer `hybrid_search_memories` for mixed queries. Use `search_semantic_memories` for pure conceptual similarity.
    i. Propose an **Updated Plan** (1-3 structured `PlanStep` JSON objects). Explain reasoning for changes. Use `record_thought(thought_type='plan')` for complex planning.
4.  Action Decision: Choose **ONE** action based on the *first planned step* in your Updated Plan:
    *   Call Memory Tool: Select the most precise `unified_memory:*` tool (or other available tool). Provide args per schema. **Mandatory:** Call `create_workflow` if context shows 'No Active Workflow'.
    *   Record Thought: Use `record_thought` for logging reasoning, questions, etc. Specify `thought_chain_id` if not the default/current one.
    *   Signal Completion: If Overall Goal is MET, respond ONLY with "Goal Achieved:" and summary.
5.  Output Format: Respond **ONLY** with the valid JSON for the chosen tool call OR "Goal Achieved:" text. Include the updated plan JSON within your reasoning text using the format `Updated Plan:\n```json\n[...plan steps...]\n````.

Key Considerations:
*   Use memory confidence. Update memories via `update_memory` if needed.
*   Store important learned info using `store_memory`.
*   Use `current_working_memory` for immediate relevance.
*   Dependencies: Ensure `depends_on` actions are likely complete. Use `get_action_details`.
*   Artifacts: Track outputs (`record_artifact`), retrieve inputs (`get_artifacts`/`get_artifact_by_id`).
*   Thought Chains: Use `create_thought_chain` for complex sub-problems. Record subsequent thoughts using the correct `thought_chain_id`.
*   Linking: Use specific `link_type`s.
"""
        # Prepare context string
        context_str = json.dumps(context, indent=2, default=str, ensure_ascii=False); max_context_len = 25000
        if len(context_str) > max_context_len: context_str = context_str[:max_context_len] + "\n... (Context Truncated)\n}"; self.logger.warning("Truncated context string sent to LLM.")

        user_prompt = f"Current Context:\n```json\n{context_str}\n```\n\n"
        user_prompt += f"My Current Plan (Structured):\n```json\n{json.dumps([s.model_dump() for s in self.state.current_plan], indent=2)}\n```\n\n"
        user_prompt += f"Last Action Summary:\n{self.state.last_action_summary}\n\n"
        if self.state.last_error_details: user_prompt += f"**CRITICAL: Address Last Error:**\n```json\n{json.dumps(self.state.last_error_details, indent=2)}\n```\n\n"
        if self.state.last_meta_feedback: user_prompt += f"**Meta-Cognitive Feedback:**\n{self.state.last_meta_feedback}\n\n"
        user_prompt += f"Overall Goal: {goal}\n\n"
        user_prompt += "**Instruction:** Analyze context & errors. Reason step-by-step. Update plan (output structured JSON plan steps in reasoning text). Decide ONE action based on the *first* planned step (Tool JSON or 'Goal Achieved:'). Focus on dependencies, artifacts, explicit memory storage/updates, custom thought chains, linking, and using working memory context."
        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]

    async def _call_agent_llm(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Claude 3.7 Sonnet, includes structured plan parsing."""
        self.logger.info("Calling Agent LLM (Claude 3.7 Sonnet) for decision/plan...", emoji_key="robot_face")
        if not self.anthropic_client: return {"decision": "error", "message": "Anthropic client unavailable."}
        messages = self._construct_agent_prompt(goal, context)
        api_tools = self.tool_schemas

        try:
            response: Message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620", max_tokens=4000,
                messages=messages, tools=api_tools, tool_choice={"type": "auto"}, temperature=0.4
            )
            self.logger.debug(f"LLM Raw Response Stop Reason: {response.stop_reason}")

            decision = {"decision": "error", "message": "LLM provided no actionable output."}
            text_parts = []
            tool_call = None
            updated_plan_steps = None

            for block in response.content:
                if block.type == "text": text_parts.append(block.text)
                elif block.type == "tool_use": tool_call = block

            full_text = "".join(text_parts).strip()

            # Parse Updated Plan from Text
            plan_match = re.search(r"Updated Plan:\s*```json\s*([\s\S]+?)\s*```", full_text, re.IGNORECASE)
            if plan_match:
                plan_json_str = plan_match.group(1).strip()
                try:
                    plan_data = json.loads(plan_json_str)
                    if isinstance(plan_data, list):
                        validated_plan = [PlanStep(**step_data) for step_data in plan_data]
                        updated_plan_steps = validated_plan
                        self.logger.info(f"LLM proposed updated plan with {len(updated_plan_steps)} steps.")
                    else: self.logger.warning("LLM plan update was not a list.")
                except (json.JSONDecodeError, ValidationError, TypeError) as e:
                    self.logger.warning(f"Failed to parse structured plan from LLM response: {e}")
            else: self.logger.debug("No structured 'Updated Plan:' block found in LLM text.")

            # Determine Final Decision
            if tool_call:
                tool_name_sanitized = tool_call.name; tool_input = tool_call.input or {}
                original_tool_name = self.mcp_client.server_manager.sanitized_to_original.get(tool_name_sanitized, tool_name_sanitized)
                self.logger.info(f"LLM chose tool: {original_tool_name}", emoji_key="hammer_and_wrench")
                decision = {"decision": "call_tool", "tool_name": original_tool_name, "arguments": tool_input}
            elif full_text.startswith("Goal Achieved:"):
                decision = {"decision": "complete", "summary": full_text.replace("Goal Achieved:", "").strip()}
            elif full_text: # No tool, no completion -> treat as reasoning/thought
                 decision = {"decision": "thought_process", "content": full_text}
                 self.logger.info("LLM provided text reasoning/thought.")
            # else: decision remains default error

            # Attach parsed plan to the decision object
            if updated_plan_steps: decision["updated_plan_steps"] = updated_plan_steps

            self.logger.debug(f"Agent Decision Parsed: {decision}")
            return decision

        except anthropic.APIConnectionError as e: msg = f"API Connection Error: {e}"; self.logger.error(msg, exc_info=True)
        except anthropic.RateLimitError: msg = "Rate limit exceeded."; self.logger.error(msg, exc_info=True); await asyncio.sleep(random.uniform(5, 10))
        except anthropic.APIStatusError as e: msg = f"API Error {e.status_code}: {e.message}"; self.logger.error(f"Anthropic API status error: {e.status_code} - {e.response}", exc_info=True)
        except Exception as e: msg = f"Unexpected LLM interaction error: {e}"; self.logger.error(msg, exc_info=True)
        return {"decision": "error", "message": msg}

    async def _run_auto_linking(self, memory_id: str):
        """Background task to automatically link a new memory using richer link types."""
        try:
            if not memory_id or not self.state.workflow_id: return
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            self.logger.debug(f"Attempting auto-linking for memory {memory_id[:8]}...")

            source_mem_details_result = await self._execute_tool_call_internal(TOOL_GET_MEMORY_BY_ID, {"memory_id": memory_id, "include_links": False}, record_action=False)
            if not source_mem_details_result.get("success"): self.logger.warning(f"Auto-linking failed: couldn't retrieve source memory {memory_id}"); return
            source_mem = source_mem_details_result

            query_text = source_mem.get("description", "") or source_mem.get("content", "")[:200]
            if not query_text: return

            search_tool = TOOL_SEMANTIC_SEARCH
            if not self._find_tool_server(search_tool): self.logger.warning(f"Skipping auto-linking: Tool {search_tool} unavailable."); return

            similar_results = await self._execute_tool_call_internal(search_tool, {"workflow_id": self.state.workflow_id, "query": query_text, "limit": self.auto_linking_max_links + 1, "threshold": self.auto_linking_threshold }, record_action=False)
            if not similar_results.get("success"): return

            link_count = 0
            for similar_mem_summary in similar_results.get("memories", []):
                target_id = similar_mem_summary.get("memory_id")
                if not target_id or target_id == memory_id: continue

                target_mem_details_result = await self._execute_tool_call_internal(TOOL_GET_MEMORY_BY_ID, {"memory_id": target_id, "include_links": False}, record_action=False)
                if not target_mem_details_result.get("success"): continue
                target_mem = target_mem_details_result

                inferred_link_type = LinkType.RELATED.value
                source_type = source_mem.get("memory_type"); target_type = target_mem.get("memory_type")
                if source_type == MemoryType.INSIGHT.value and target_type == MemoryType.FACT.value: inferred_link_type = LinkType.SUPPORTS.value
                elif source_type == MemoryType.FACT.value and target_type == MemoryType.INSIGHT.value: inferred_link_type = LinkType.SUPPORTS.value
                elif source_type == MemoryType.EVIDENCE.value and target_type == MemoryType.HYPOTHESIS.value: inferred_link_type = LinkType.SUPPORTS.value
                elif source_type == MemoryType.HYPOTHESIS.value and target_type == MemoryType.EVIDENCE.value: inferred_link_type = LinkType.SUPPORTS.value

                if not self._find_tool_server(TOOL_CREATE_LINK): self.logger.warning(f"Cannot create link: Tool {TOOL_CREATE_LINK} unavailable."); break

                await self._execute_tool_call_internal(TOOL_CREATE_LINK, {"source_memory_id": memory_id, "target_memory_id": target_id, "link_type": inferred_link_type, "strength": similar_mem_summary.get("similarity", 0.7), "description": f"Auto-link ({inferred_link_type}) based on similarity"}, record_action=False)
                link_count += 1; self.logger.debug(f"Auto-linked memory {memory_id[:8]} to {target_id[:8]} ({inferred_link_type}, similarity: {similar_mem_summary.get('similarity', 0):.2f})")
                if link_count >= self.auto_linking_max_links: break
        except Exception as e: self.logger.warning(f"Error in auto-linking task for {memory_id}: {e}", exc_info=False)

    async def _check_prerequisites(self, dependency_ids: List[str]) -> Tuple[bool, str]:
        """Check if all prerequisite actions are completed using get_action_details."""
        if not dependency_ids: return True, "No dependencies"
        self.logger.debug(f"Checking prerequisites: {dependency_ids}")
        if not self._find_tool_server(TOOL_GET_ACTION_DETAILS): return False, f"Cannot check: Tool {TOOL_GET_ACTION_DETAILS} unavailable."
        try:
            dep_details_result = await self._execute_tool_call_internal(TOOL_GET_ACTION_DETAILS, {"action_ids": dependency_ids, "include_dependencies": False}, record_action=False)
            if not dep_details_result.get("success"): return False, f"Failed to check dependencies: {dep_details_result.get('error', 'Unknown error')}"
            actions = dep_details_result.get("actions", []); found_ids = {a.get("action_id") for a in actions}; missing = list(set(dependency_ids) - found_ids)
            if missing: return False, f"Dependency actions not found: {missing}"
            incomplete = [a.get("action_id") for a in actions if a.get("status") != ActionStatus.COMPLETED.value]
            if incomplete: incomplete_titles = [f"'{a.get('title', a.get('action_id')[:8])}' ({a.get('status')})" for a in actions if a.get('action_id') in incomplete]; return False, f"Dependencies not completed: {', '.join(incomplete_titles)}"
            return True, "All dependencies completed"
        except Exception as e: self.logger.error(f"Error checking prerequisites: {e}", exc_info=False); return False, f"Error checking prerequisites: {str(e)}"

    async def _execute_tool_call_internal(
        self, tool_name: str, arguments: Dict[str, Any],
        record_action: bool = True,
        planned_dependencies: Optional[List[str]] = None
        ) -> Dict[str, Any]:
        """Handles server lookup, dependency checks, execution, results, optional action recording, dependency recording, and triggers."""
        action_id = None
        tool_result_content = {"success": False, "error": "Execution error."}
        start_time = time.time()

        # 1. Find Server
        target_server = self._find_tool_server(tool_name)
        if not target_server: err_msg = f"Tool/server unavailable: {tool_name}"; self.logger.error(err_msg); self.state.last_error_details = {"tool": tool_name, "error": err_msg}; return {"success": False, "error": err_msg, "status_code": 503}

        # 2. Dependency Check
        if planned_dependencies:
            met, reason = await self._check_prerequisites(planned_dependencies)
            if not met: err_msg = f"Prerequisites not met for {tool_name}: {reason}"; self.logger.warning(err_msg); self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "dependency_failure", "dependencies": planned_dependencies}; self.state.needs_replan = True; return {"success": False, "error": err_msg, "status_code": 412}
            self.logger.info(f"Prerequisites {planned_dependencies} met for {tool_name}.")

        # 3. Record Action Start (Optional)
        # Determine if this tool call represents a significant agent action
        should_record_start = record_action and tool_name not in [
            TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION, # Meta-actions
            TOOL_GET_CONTEXT, TOOL_GET_WORKING_MEMORY, # Context gathering
            TOOL_SEMANTIC_SEARCH, TOOL_HYBRID_SEARCH, TOOL_QUERY_MEMORIES, # Searches
            TOOL_GET_MEMORY_BY_ID, TOOL_GET_LINKED_MEMORIES, # Retrievals
            TOOL_GET_ACTION_DETAILS, TOOL_GET_ARTIFACTS, TOOL_GET_ARTIFACT_BY_ID, # Retrievals
            TOOL_ADD_ACTION_DEPENDENCY, TOOL_CREATE_LINK, # Internal linking/dep mgmt
            TOOL_LIST_WORKFLOWS, TOOL_COMPUTE_STATS, TOOL_SUMMARIZE_TEXT, # Meta/utility
            TOOL_OPTIMIZE_WM, TOOL_AUTO_FOCUS, TOOL_PROMOTE_MEM, # Periodic/internal cognitive
            TOOL_REFLECTION, TOOL_CONSOLIDATION, TOOL_DELETE_EXPIRED_MEMORIES # Periodic/meta
        ]
        if should_record_start:
            action_id = await self._record_action_start_internal(tool_name, arguments, target_server)
            # 3.5 Record Dependencies AFTER starting the action
            if action_id and planned_dependencies:
                await self._record_action_dependencies_internal(action_id, planned_dependencies)

        # 4. Execute Tool
        try:
            current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
            # Inject workflow_id automatically
            if 'workflow_id' not in arguments and current_wf_id and tool_name not in [TOOL_CREATE_WORKFLOW, TOOL_LIST_WORKFLOWS, 'core:list_servers', 'core:get_tool_schema']: arguments['workflow_id'] = current_wf_id
            # Inject context_id if needed
            if 'context_id' not in arguments and self.state.context_id and tool_name in [TOOL_GET_WORKING_MEMORY, TOOL_OPTIMIZE_WM, TOOL_AUTO_FOCUS]: arguments['context_id'] = self.state.context_id
            # Inject current thought chain ID if needed and available
            if 'thought_chain_id' not in arguments and self.state.current_thought_chain_id and tool_name == TOOL_RECORD_THOUGHT: arguments['thought_chain_id'] = self.state.current_thought_chain_id

            clean_args = {k: v for k, v in arguments.items() if v is not None}

            call_tool_result = await self.mcp_client.execute_tool(target_server, tool_name, clean_args)
            latency_ms = (time.time() - start_time) * 1000
            self.state.tool_usage_stats[tool_name]["latency_ms_total"] += latency_ms

            # Process result
            if isinstance(call_tool_result, dict):
                is_error = call_tool_result.get("isError", False); content = call_tool_result.get("content")
                if is_error or (content is None and "success" not in call_tool_result): error_msg = str(content or call_tool_result.get("error", "Unknown tool error.")); tool_result_content = {"success": False, "error": error_msg}
                elif isinstance(content, dict) and "success" in content: tool_result_content = content
                else: tool_result_content = {"success": True, "data": content}
            else: tool_result_content = {"success": False, "error": f"Unexpected result type: {type(call_tool_result)}"}

            log_msg = f"Tool {tool_name} executed. Success: {tool_result_content.get('success')} ({latency_ms:.0f}ms)"
            self.logger.info(log_msg, emoji_key="checkered_flag" if tool_result_content.get('success') else "warning")
            self.state.last_action_summary = log_msg
            if not tool_result_content.get('success'):
                 err_detail = str(tool_result_content.get('error', 'Unknown'))[:150]; self.state.last_action_summary += f" Error: {err_detail}"; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_detail, "result": tool_result_content}; self.state.tool_usage_stats[tool_name]["failure"] += 1
            else:
                 self.state.last_error_details = None; self.state.consecutive_error_count = 0; self.state.tool_usage_stats[tool_name]["success"] += 1
                 # Trigger Post-Success Actions
                 if tool_name in [TOOL_STORE_MEMORY, TOOL_UPDATE_MEMORY] and tool_result_content.get("memory_id"): self._start_background_task(self._run_auto_linking(tool_result_content["memory_id"]))
                 if tool_name == TOOL_RECORD_ARTIFACT and tool_result_content.get("linked_memory_id"): self._start_background_task(self._run_auto_linking(tool_result_content["linked_memory_id"]))
                 if tool_name in [TOOL_GET_MEMORY_BY_ID, TOOL_QUERY_MEMORIES, TOOL_HYBRID_SEARCH]:
                    mem_ids_to_check = []
                    if tool_name == TOOL_GET_MEMORY_BY_ID: mem_ids_to_check = [arguments.get("memory_id")]
                    else: memories = tool_result_content.get("memories", []) if isinstance(tool_result_content, dict) else []; mem_ids_to_check = [m.get("memory_id") for m in memories[:3]]
                    for mem_id in filter(None, mem_ids_to_check): self._start_background_task(self._check_and_trigger_promotion(mem_id))
                 # Update current thought chain ID if a new one was created (Tier 3)
                 if tool_name == TOOL_CREATE_THOUGHT_CHAIN and tool_result_content.get("success"):
                      new_chain_id = tool_result_content.get("thought_chain_id")
                      if new_chain_id:
                           self.state.current_thought_chain_id = new_chain_id
                           self.logger.info(f"Switched current thought chain to newly created: {new_chain_id}")


        except (ToolError, ToolInputError) as e:
             err_str = str(e); status_code = getattr(e, 'status_code', None); self.logger.error(f"Tool Error executing {tool_name}: {e}", exc_info=False); tool_result_content = {"success": False, "error": err_str, "status_code": status_code}; self.state.last_action_summary = f"Tool {tool_name} Error: {err_str[:100]}"; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": type(e).__name__}; self.state.tool_usage_stats[tool_name]["failure"] += 1
             if status_code == 412: self.state.last_error_details["type"] = "dependency_failure"; self.state.needs_replan = True
        except Exception as e: err_str = str(e); self.logger.error(f"Unexpected Error executing {tool_name}: {e}", exc_info=True); tool_result_content = {"success": False, "error": f"Unexpected error: {err_str}"}; self.state.last_action_summary = f"Execution failed: Unexpected error."; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "Unexpected"}; self.state.tool_usage_stats[tool_name]["failure"] += 1

        # 5. Record Action Completion (Optional)
        if should_record_start and action_id: # Only record completion if start was recorded
             await self._record_action_completion_internal(action_id, tool_result_content)

        # 6. Handle Workflow Side Effects
        await self._handle_workflow_side_effects(tool_name, arguments, tool_result_content)

        return tool_result_content

    async def _record_action_dependencies_internal(self, source_action_id: str, target_action_ids: List[str]):
        """Records dependencies using the add_action_dependency tool."""
        if not source_action_id or not target_action_ids: return
        self.logger.debug(f"Recording dependencies for action {source_action_id[:8]}: depends on {target_action_ids}")
        dep_tool_name = TOOL_ADD_ACTION_DEPENDENCY
        if not self._find_tool_server(dep_tool_name): self.logger.error(f"Cannot record dependency: Tool '{dep_tool_name}' unavailable."); return

        dep_tasks = []; unique_target_ids = set(target_action_ids)
        for target_id in unique_target_ids:
            if target_id == source_action_id: self.logger.warning(f"Skipping self-dependency for action {source_action_id}"); continue
            args = {"source_action_id": source_action_id, "target_action_id": target_id, "dependency_type": "requires"}
            task = asyncio.create_task(self._execute_tool_call_internal(dep_tool_name, args, record_action=False, planned_dependencies=None))
            dep_tasks.append(task)
        results = await asyncio.gather(*dep_tasks, return_exceptions=True)
        valid_target_ids = [tid for tid in unique_target_ids if tid != source_action_id]
        for i, res in enumerate(results):
            if i >= len(valid_target_ids): break
            target_id = valid_target_ids[i]
            if isinstance(res, Exception): self.logger.error(f"Error recording dependency {source_action_id[:8]} -> {target_id[:8]}: {res}", exc_info=False)
            elif isinstance(res, dict) and not res.get("success"): self.logger.warning(f"Failed recording dependency {source_action_id[:8]} -> {target_id[:8]}: {res.get('error')}")

    async def _record_action_start_internal(self, primary_tool_name: str, primary_tool_args: Dict[str, Any], primary_target_server: str) -> Optional[str]:
         """Internal helper to record action start."""
         action_id = None; start_title = f"Exec: {primary_tool_name.split(':')[-1]}"; start_reasoning = f"Agent initiated tool: {primary_tool_name}"; current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
         if not current_wf_id: self.logger.warning("Cannot record start: No active workflow"); return None
         start_tool_name = TOOL_RECORD_ACTION_START
         if not self._find_tool_server(start_tool_name): self.logger.error(f"Cannot record start: Tool '{start_tool_name}' unavailable."); return None
         try:
             safe_tool_args = json.loads(json.dumps(primary_tool_args, default=str))
             start_args = {"workflow_id": current_wf_id, "action_type": ActionType.TOOL_USE.value, "title": start_title, "reasoning": start_reasoning, "tool_name": primary_tool_name, "tool_args": safe_tool_args}
             start_result_content = await self._execute_tool_call_internal(start_tool_name, start_args, record_action=False)
             if start_result_content.get("success"):
                 action_id = start_result_content.get("action_id")
                 if action_id: self.logger.debug(f"Action {action_id} started for {primary_tool_name}.")
                 else: self.logger.warning(f"Record action start succeeded but returned no action ID.")
             else: error_msg = start_result_content.get('error', 'Unknown'); self.logger.warning(f"Failed recording start for {primary_tool_name}: {error_msg}")
         except Exception as e: self.logger.error(f"Exception recording start for {primary_tool_name}: {e}", exc_info=True)
         return action_id

    async def _record_action_completion_internal(self, action_id: str, tool_result_content: Dict):
         """Internal helper to record action completion."""
         status = ActionStatus.COMPLETED.value if tool_result_content.get("success") else ActionStatus.FAILED.value
         comp_tool_name = TOOL_RECORD_ACTION_COMPLETION
         if not self._find_tool_server(comp_tool_name): self.logger.error(f"Cannot record completion: Tool '{comp_tool_name}' unavailable."); return
         try:
             safe_result = json.loads(json.dumps(tool_result_content, default=str))
             completion_args = {"action_id": action_id, "status": status, "tool_result": safe_result}
             comp_result_content = await self._execute_tool_call_internal(comp_tool_name, completion_args, record_action=False)
             if not comp_result_content.get("success"): error_msg = comp_result_content.get('error', 'Unknown'); self.logger.warning(f"Failed recording completion for {action_id}: {error_msg}")
             else: self.logger.debug(f"Action {action_id} completion recorded ({status})")
         except Exception as e: self.logger.error(f"Error recording completion for {action_id}: {e}", exc_info=True)

    async def _handle_workflow_side_effects(self, tool_name: str, arguments: Dict, result_content: Dict):
        """Handles state changes after specific tool calls."""
        if tool_name == TOOL_CREATE_WORKFLOW and result_content.get("success"):
            new_wf_id = result_content.get("workflow_id"); parent_id = arguments.get("parent_workflow_id")
            if new_wf_id:
                self.state.workflow_id = new_wf_id; self.state.context_id = new_wf_id
                if parent_id: self.state.workflow_stack.append(new_wf_id)
                else: self.state.workflow_stack = [new_wf_id]
                # --- Set current_thought_chain_id for new workflow (Tier 3) ---
                self.state.current_thought_chain_id = result_content.get("primary_thought_chain_id")
                self.logger.info(f"Switched to {'sub-' if parent_id else 'new'} workflow: {new_wf_id}. Current chain: {self.state.current_thought_chain_id}", emoji_key="label")
                self.state.current_plan = [PlanStep(description=f"Start new workflow: {result_content.get('title', 'Untitled')}. Goal: {result_content.get('goal', 'Not specified')}.")]; self.state.consecutive_error_count = 0; self.state.needs_replan = False
        elif tool_name == TOOL_UPDATE_WORKFLOW_STATUS and result_content.get("success"):
            status = arguments.get("status"); wf_id = arguments.get("workflow_id")
            if status in [s.value for s in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.ABANDONED]] and self.state.workflow_stack and wf_id == self.state.workflow_stack[-1]:
                 finished_wf = self.state.workflow_stack.pop()
                 if self.state.workflow_stack:
                      self.state.workflow_id = self.state.workflow_stack[-1]; self.state.context_id = self.state.workflow_id
                      # Fetch parent's primary thought chain ID
                      await self._set_default_thought_chain_id()
                      self.logger.info(f"Sub-workflow {finished_wf} finished. Returning to parent {self.state.workflow_id}. Current chain: {self.state.current_thought_chain_id}", emoji_key="arrow_left")
                      self.state.needs_replan = True; self.state.current_plan = [PlanStep(description=f"Returned from sub-workflow {finished_wf} (status: {status}). Re-assess parent goal.")]
                 else: self.state.workflow_id = None; self.state.context_id = None; self.state.current_thought_chain_id = None; self.logger.info(f"Root workflow {finished_wf} finished."); self.state.goal_achieved_flag = True

    async def _update_plan(self, context: Dict[str, Any], last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        """Updates the plan based on LLM proposal or heuristics."""
        self.logger.info("Updating agent plan...", emoji_key="clipboard")
        llm_proposed_plan = last_decision.get("updated_plan_steps")

        if llm_proposed_plan and isinstance(llm_proposed_plan, list):
            try:
                 validated_plan = [PlanStep(**step) if isinstance(step, dict) else step for step in llm_proposed_plan]
                 if validated_plan and all(isinstance(step, PlanStep) for step in validated_plan):
                     self.state.current_plan = validated_plan; self.logger.info(f"Plan updated by LLM with {len(validated_plan)} steps. First step: '{validated_plan[0].description[:50]}...'"); self.state.needs_replan = False
                     if self.state.last_error_details: self.state.consecutive_error_count = 0
                     if last_decision.get("decision") == "call_tool" and isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success"): self.state.successful_actions_since_reflection += 1; self.state.successful_actions_since_consolidation += 1
                     return
                 else: self.logger.warning("LLM provided invalid or empty plan structure. Falling back to heuristic.")
            except (ValidationError, TypeError) as e: self.logger.warning(f"Failed to validate LLM plan: {e}. Falling back.")

        # --- Fallback to Heuristic Plan Update ---
        if not self.state.current_plan: self.logger.warning("Plan is empty, adding default re-evaluation step."); self.state.current_plan = [PlanStep(description="Fallback: Re-evaluate situation.")]; self.state.needs_replan = True; return
        current_step = self.state.current_plan[0]

        if last_decision.get("decision") == "call_tool":
            tool_success = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            if tool_success:
                 current_step.status = "completed"; action_id_from_result = None
                 if isinstance(last_tool_result_content, dict): action_id_from_result = last_tool_result_content.get('action_id') or (last_tool_result_content.get('data') or {}).get('action_id')
                 summary_text = f"Success: {str(last_tool_result_content)[:100]}..."
                 if action_id_from_result and last_decision.get("tool_name") == TOOL_RECORD_ACTION_START : summary_text += f" (ActionID: {action_id_from_result[:8]})"
                 current_step.result_summary = summary_text; self.state.current_plan.pop(0)
                 if not self.state.current_plan: self.state.current_plan.append(PlanStep(description="Analyze successful tool output and plan next steps."))
                 self.state.consecutive_error_count = 0; self.state.needs_replan = False; self.state.successful_actions_since_reflection += 1; self.state.successful_actions_since_consolidation += 1
            else:
                 current_step.status = "failed"; error_msg = str(last_tool_result_content.get('error', 'Unknown failure'))[:150]; current_step.result_summary = f"Failure: {error_msg}"
                 self.state.current_plan = [current_step] + self.state.current_plan[1:]
                 if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Analyze failure"): self.state.current_plan.insert(1, PlanStep(description=f"Analyze failure of step '{current_step.description[:30]}...' and replan."))
                 self.state.consecutive_error_count += 1; self.state.needs_replan = True; self.state.successful_actions_since_reflection = self.state.current_reflection_threshold # Use dynamic threshold
        elif last_decision.get("decision") == "thought_process":
             current_step.status = "completed"; current_step.result_summary = f"Thought Recorded: {last_decision.get('content','')[:50]}..."
             self.state.current_plan.pop(0)
             if not self.state.current_plan: self.state.current_plan.append(PlanStep(description="Decide next action based on recorded thought."))
             self.state.consecutive_error_count = 0; self.state.needs_replan = False
        elif last_decision.get("decision") == "complete": self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing.", status="completed")]; self.state.consecutive_error_count = 0; self.state.needs_replan = False
        else:
             current_step.status = "failed"; current_step.result_summary = f"Agent/Tool Error: {self.state.last_action_summary[:100]}..."
             self.state.current_plan = [current_step] + self.state.current_plan[1:]
             if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.startswith("Re-evaluate due"): self.state.current_plan.insert(1, PlanStep(description="Re-evaluate due to agent error or unclear decision."))
             self.state.consecutive_error_count += 1; self.state.needs_replan = True
        log_plan = f"Plan updated (Heuristic). Steps: {len(self.state.current_plan)}. Next: '{self.state.current_plan[0].description[:60]}...'"
        self.logger.info(log_plan)

    # --- Periodic Tasks (Enhanced for Tier 3) ---
    async def _run_periodic_tasks(self):
        """Runs meta-cognition and maintenance tasks, including adaptive adjustments."""
        if not self.state.workflow_id or not self.state.context_id or self._shutdown_event.is_set(): return

        tasks_to_run: List[Tuple[str, Dict]] = []; trigger_reason = []
        reflection_tool_available = self._find_tool_server(TOOL_REFLECTION) is not None
        consolidation_tool_available = self._find_tool_server(TOOL_CONSOLIDATION) is not None
        optimize_wm_tool_available = self._find_tool_server(TOOL_OPTIMIZE_WM) is not None
        auto_focus_tool_available = self._find_tool_server(TOOL_AUTO_FOCUS) is not None
        promote_mem_tool_available = self._find_tool_server(TOOL_PROMOTE_MEM) is not None
        stats_tool_available = self._find_tool_server(TOOL_COMPUTE_STATS) is not None
        maintenance_tool_available = self._find_tool_server(TOOL_DELETE_EXPIRED_MEMORIES) is not None

        # --- Tier 3: Stats Check & Adaptation ---
        self.state.loops_since_stats_adaptation += 1
        if self.state.loops_since_stats_adaptation >= STATS_ADAPTATION_INTERVAL:
             if stats_tool_available:
                 trigger_reason.append("StatsInterval")
                 try:
                     stats = await self._execute_tool_call_internal(
                         TOOL_COMPUTE_STATS, {"workflow_id": self.state.workflow_id}, record_action=False
                     )
                     if stats.get("success"):
                         self._adapt_thresholds(stats)
                         # Trigger consolidation if episodic memories are high
                         episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
                         # Example: Trigger if > 2x the consolidation threshold, regardless of success count
                         if episodic_count > (self.state.current_consolidation_threshold * 2) and consolidation_tool_available:
                             if not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run): # Avoid duplicate scheduling
                                 tasks_to_run.append((TOOL_CONSOLIDATION, {"workflow_id": self.state.workflow_id, "consolidation_type": "summary", "query_filter": {"memory_level": MemoryLevel.EPISODIC.value}, "max_source_memories": self.consolidation_max_sources}))
                                 trigger_reason.append(f"HighEpisodic({episodic_count})")
                                 self.state.successful_actions_since_consolidation = 0 # Reset counter as we're consolidating now
                     else: self.logger.warning(f"Failed to compute stats for adaptation: {stats.get('error')}")
                 except Exception as e: self.logger.error(f"Error during stats computation/adaptation: {e}", exc_info=False)
                 self.state.loops_since_stats_adaptation = 0 # Reset interval counter
             else: self.logger.warning(f"Skipping stats/adaptation: Tool {TOOL_COMPUTE_STATS} not available")


        # --- Tier 3: Maintenance Check ---
        self.state.loops_since_maintenance += 1
        if self.state.loops_since_maintenance >= MAINTENANCE_INTERVAL:
             if maintenance_tool_available:
                 tasks_to_run.append((TOOL_DELETE_EXPIRED_MEMORIES, {})); trigger_reason.append("MaintenanceInterval")
                 self.state.loops_since_maintenance = 0 # Reset interval counter
             else: self.logger.warning(f"Skipping maintenance: Tool {TOOL_DELETE_EXPIRED_MEMORIES} not available")

        # --- Existing Triggers (Now use dynamic thresholds) ---
        # Reflection Trigger
        if self.state.needs_replan or self.state.successful_actions_since_reflection >= self.state.current_reflection_threshold:
            if reflection_tool_available:
                if not any(task[0] == TOOL_REFLECTION for task in tasks_to_run): # Avoid duplicates if scheduled by stats
                     reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]; tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type})); trigger_reason.append(f"ReplanNeeded({self.state.needs_replan})" if self.state.needs_replan else f"SuccessCount({self.state.successful_actions_since_reflection}>={self.state.current_reflection_threshold})"); self.state.successful_actions_since_reflection = 0; self.state.reflection_cycle_index += 1
            else: self.logger.warning(f"Skipping reflection: Tool {TOOL_REFLECTION} not available")
        # Consolidation Trigger
        if self.state.successful_actions_since_consolidation >= self.state.current_consolidation_threshold:
             if consolidation_tool_available:
                 if not any(task[0] == TOOL_CONSOLIDATION for task in tasks_to_run): # Avoid duplicates if scheduled by stats
                      tasks_to_run.append((TOOL_CONSOLIDATION, {"workflow_id": self.state.workflow_id, "consolidation_type": "summary", "query_filter": {"memory_level": MemoryLevel.EPISODIC.value}, "max_source_memories": self.consolidation_max_sources})); trigger_reason.append(f"ConsolidateThreshold({self.state.successful_actions_since_consolidation}>={self.state.current_consolidation_threshold})"); self.state.successful_actions_since_consolidation = 0
             else: self.logger.warning(f"Skipping consolidation: Tool {TOOL_CONSOLIDATION} not available")
        # Optimization Trigger
        self.state.loops_since_optimization += 1
        if self.state.loops_since_optimization >= OPTIMIZATION_LOOP_INTERVAL: # Use constant interval
             if optimize_wm_tool_available: tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.state.context_id})); trigger_reason.append("OptimizeInterval")
             else: self.logger.warning(f"Skipping optimization: Tool {TOOL_OPTIMIZE_WM} not available")
             if auto_focus_tool_available: tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.state.context_id})); trigger_reason.append("FocusUpdate")
             else: self.logger.warning(f"Skipping auto-focus: Tool {TOOL_AUTO_FOCUS} not available")
             self.state.loops_since_optimization = 0
        # Promotion Check Trigger
        self.state.loops_since_promotion_check += 1
        if self.state.loops_since_promotion_check >= MEMORY_PROMOTION_LOOP_INTERVAL: # Use constant interval
             if promote_mem_tool_available: tasks_to_run.append(("CHECK_PROMOTIONS", {})); trigger_reason.append("PromotionInterval")
             else: self.logger.warning(f"Skipping promotion check: Tool {TOOL_PROMOTE_MEM} not available")
             self.state.loops_since_promotion_check = 0

        # Execute Scheduled Tasks
        if tasks_to_run:
            unique_reasons = sorted(set(trigger_reason)) # Deduplicate reasons for logging
            self.logger.info(f"Running {len(tasks_to_run)} periodic tasks (Triggers: {', '.join(unique_reasons)})...", emoji_key="brain")
            # Prioritize maintenance and stats if scheduled
            tasks_to_run.sort(key=lambda x: 0 if x[0] == TOOL_DELETE_EXPIRED_MEMORIES else 1 if x[0] == TOOL_COMPUTE_STATS else 2)
            for tool_name, args in tasks_to_run:
                 if self._shutdown_event.is_set(): break
                 try:
                     if tool_name == "CHECK_PROMOTIONS": await self._trigger_promotion_checks(); continue
                     self.logger.debug(f"Executing periodic task: {tool_name} with args: {args}")
                     result_content = await self._execute_tool_call_internal(tool_name, args, record_action=False)
                     # Meta-Cognition Feedback Loop
                     if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and result_content.get('success'):
                          content_key = "reflection_content" if tool_name == TOOL_REFLECTION else "consolidated_content"; feedback = result_content.get(content_key, "") or result_content.get("data", "")
                          if feedback: feedback_summary = str(feedback).split('\n')[0][:150]; self.state.last_meta_feedback = f"Feedback from {tool_name.split(':')[-1]}: {feedback_summary}..."; self.logger.info(self.state.last_meta_feedback); self.state.needs_replan = True
                 except Exception as e: self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)
                 await asyncio.sleep(0.1) # Small delay

    # --- Tier 3: Adaptive Threshold Logic ---
    def _adapt_thresholds(self, stats: Dict[str, Any]):
        """Adjusts reflection and consolidation thresholds based on memory stats."""
        self.logger.debug(f"Adapting thresholds based on stats: {stats}")
        adjustment_factor = 0.1 # How much to adjust thresholds by each time

        # Example Heuristic 1: High episodic count -> Lower consolidation threshold (consolidate sooner)
        episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
        target_episodic = BASE_CONSOLIDATION_THRESHOLD * 1.5 # Target range
        if episodic_count > target_episodic * 1.5: # Significantly over target
            new_threshold = max(MIN_CONSOLIDATION_THRESHOLD, self.state.current_consolidation_threshold - math.ceil(self.state.current_consolidation_threshold * adjustment_factor))
            if new_threshold != self.state.current_consolidation_threshold:
                self.logger.info(f"High episodic count ({episodic_count}). Lowering consolidation threshold: {self.state.current_consolidation_threshold} -> {new_threshold}")
                self.state.current_consolidation_threshold = new_threshold
        elif episodic_count < target_episodic * 0.75: # Well below target
             new_threshold = min(MAX_CONSOLIDATION_THRESHOLD, self.state.current_consolidation_threshold + math.ceil(self.state.current_consolidation_threshold * adjustment_factor))
             if new_threshold != self.state.current_consolidation_threshold:
                 self.logger.info(f"Low episodic count ({episodic_count}). Raising consolidation threshold: {self.state.current_consolidation_threshold} -> {new_threshold}")
                 self.state.current_consolidation_threshold = new_threshold


        # Example Heuristic 2: High tool failure rate -> Lower reflection threshold (reflect sooner)
        total_calls = sum(sum(v.values()) for k, v in self.state.tool_usage_stats.items() if isinstance(v, dict) and k != 'latency_ms_total')
        total_failures = sum(v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        failure_rate = (total_failures / total_calls) if total_calls > 5 else 0.0 # Require minimum calls

        if failure_rate > 0.25: # High failure rate
             new_threshold = max(MIN_REFLECTION_THRESHOLD, self.state.current_reflection_threshold - math.ceil(self.state.current_reflection_threshold * adjustment_factor))
             if new_threshold != self.state.current_reflection_threshold:
                 self.logger.info(f"High tool failure rate ({failure_rate:.1%}). Lowering reflection threshold: {self.state.current_reflection_threshold} -> {new_threshold}")
                 self.state.current_reflection_threshold = new_threshold
        elif failure_rate < 0.05 and total_calls > 10: # Very low failure rate
             new_threshold = min(MAX_REFLECTION_THRESHOLD, self.state.current_reflection_threshold + math.ceil(self.state.current_reflection_threshold * adjustment_factor))
             if new_threshold != self.state.current_reflection_threshold:
                  self.logger.info(f"Low tool failure rate ({failure_rate:.1%}). Raising reflection threshold: {self.state.current_reflection_threshold} -> {new_threshold}")
                  self.state.current_reflection_threshold = new_threshold


    async def _trigger_promotion_checks(self):
        """Checks promotion criteria for recently accessed, eligible memories."""
        self.logger.debug("Running periodic promotion check...")
        tool_name_query = TOOL_QUERY_MEMORIES
        if not self._find_tool_server(tool_name_query): self.logger.warning(f"Skipping promotion check: Tool {tool_name_query} unavailable."); return

        candidate_memory_ids = set()
        try:
            episodic_results = await self._execute_tool_call_internal(tool_name_query, {"workflow_id": self.state.workflow_id, "memory_level": MemoryLevel.EPISODIC.value, "sort_by": "last_accessed", "sort_order": "DESC", "limit": 5, "include_content": False}, record_action=False)
            if episodic_results.get("success"): candidate_memory_ids.update(m.get('memory_id') for m in episodic_results.get("memories", []) if m.get('memory_id'))
            semantic_results = await self._execute_tool_call_internal(tool_name_query, {"workflow_id": self.state.workflow_id, "memory_level": MemoryLevel.SEMANTIC.value, "sort_by": "last_accessed", "sort_order": "DESC", "limit": 5, "include_content": False}, record_action=False)
            if semantic_results.get("success"): candidate_memory_ids.update(m.get('memory_id') for m in semantic_results.get("memories", []) if m.get('memory_id'))
            if candidate_memory_ids: self.logger.debug(f"Checking {len(candidate_memory_ids)} memories for promotion"); promo_tasks = [self._check_and_trigger_promotion(mem_id) for mem_id in candidate_memory_ids]; await asyncio.gather(*promo_tasks, return_exceptions=True)
            else: self.logger.debug("No recent eligible memories found for promotion check.")
        except Exception as e: self.logger.error(f"Error during periodic promotion check query: {e}", exc_info=False)

    async def _check_and_trigger_promotion(self, memory_id: str):
        """Checks a single memory for promotion and triggers it."""
        if not memory_id or not self._find_tool_server(TOOL_PROMOTE_MEM): return
        try:
            await asyncio.sleep(random.uniform(0.1, 0.5))
            promotion_result = await self._execute_tool_call_internal(TOOL_PROMOTE_MEM, {"memory_id": memory_id}, record_action=False)
            if promotion_result.get("success") and promotion_result.get("promoted"): self.logger.info(f"Memory {memory_id[:8]} promoted from {promotion_result.get('previous_level')} to {promotion_result.get('new_level')}", emoji_key="arrow_up")
        except Exception as e: self.logger.warning(f"Error in memory promotion check for {memory_id}: {e}", exc_info=False)

    # --- Run Method (Main Loop - Incorporates Tier 1, 2 & 3) ---
    async def run(self, goal: str, max_loops: int = 50):
        """Main agent execution loop."""
        if not await self.initialize(): self.logger.critical("Agent initialization failed."); return

        self.logger.info(f"Starting main loop. Goal: '{goal}' Max Loops: {max_loops}", emoji_key="arrow_forward")
        self.state.goal_achieved_flag = False

        while not self.state.goal_achieved_flag and self.state.current_loop < max_loops:
             if self._shutdown_event.is_set(): self.logger.info("Shutdown signal detected, exiting loop."); break
             self.state.current_loop += 1
             self.logger.info(f"--- Agent Loop {self.state.current_loop}/{max_loops} (RefThresh: {self.state.current_reflection_threshold}, ConThresh: {self.state.current_consolidation_threshold}) ---", emoji_key="arrows_counterclockwise")

             # Error Check
             if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                  self.logger.error(f"Max consecutive errors ({MAX_CONSECUTIVE_ERRORS}) reached. Aborting.", emoji_key="stop_sign")
                  if self.state.workflow_id: await self._update_workflow_status_internal("failed", "Agent failed due to repeated errors.")
                  break

             # 1. Gather Context
             context = await self._gather_context()
             if context.get("status") == "No Active Workflow":
                  self.logger.warning("No active workflow. Agent must create one.")
                  self.state.current_plan = [PlanStep(description=f"Create the primary workflow for goal: {goal}")]
                  self.state.needs_replan = False
             elif "errors" in context and context.get("errors"):
                  self.logger.warning(f"Context gathering encountered errors: {context['errors']}. Proceeding cautiously.")

             # 2. Decide
             agent_decision = await self._call_agent_llm(goal, context)

             # 3. Act
             decision_type = agent_decision.get("decision")
             last_tool_result_content = None

             # Get Current Plan Step and Dependencies
             current_plan_step: Optional[PlanStep] = self.state.current_plan[0] if self.state.current_plan else None
             planned_dependencies_for_step: Optional[List[str]] = current_plan_step.depends_on if current_plan_step else None

             # Update Plan based on LLM suggestion FIRST
             if agent_decision.get("updated_plan_steps"):
                  proposed_plan = agent_decision["updated_plan_steps"]
                  if isinstance(proposed_plan, list) and all(isinstance(step, PlanStep) for step in proposed_plan):
                      self.state.current_plan = proposed_plan; self.logger.info(f"Plan updated by LLM with {len(self.state.current_plan)} steps."); self.state.needs_replan = False
                      current_plan_step = self.state.current_plan[0] if self.state.current_plan else None; planned_dependencies_for_step = current_plan_step.depends_on if current_plan_step else None
                  else: self.logger.warning("LLM provided updated_plan_steps in unexpected format, ignoring.")

             # Execute Action
             if decision_type == "call_tool":
                 tool_name = agent_decision.get("tool_name"); arguments = agent_decision.get("arguments", {})
                 if not tool_name: self.logger.error("LLM requested tool call but provided no tool name."); self.state.last_action_summary = "Agent error: Missing tool name."; self.state.last_error_details = {"agent_decision_error": "Missing tool name"}; self.state.consecutive_error_count += 1; self.state.needs_replan = True
                 else:
                     self.logger.info(f"Agent requests tool: {tool_name} with args: {arguments}", emoji_key="wrench")
                     last_tool_result_content = await self._execute_tool_call_internal(tool_name, arguments, True, planned_dependencies_for_step)
                     if isinstance(last_tool_result_content, dict) and not last_tool_result_content.get("success"):
                          self.state.needs_replan = True
                          if last_tool_result_content.get("status_code") == 412: self.logger.warning(f"Tool execution failed due to unmet prerequisites: {last_tool_result_content.get('error')}")
                          else: self.logger.warning(f"Tool execution failed: {last_tool_result_content.get('error')}")

             elif decision_type == "thought_process":
                  thought_content = agent_decision.get("content", "No thought content provided.")
                  self.logger.info(f"Agent reasoning: '{thought_content[:100]}...'. Recording.", emoji_key="thought_balloon")
                  if self.state.workflow_id:
                      # Use current_thought_chain_id if available
                      thought_args = {"workflow_id": self.state.workflow_id, "content": thought_content, "thought_type": ThoughtType.INFERENCE.value}
                      if self.state.current_thought_chain_id: thought_args["thought_chain_id"] = self.state.current_thought_chain_id
                      try: thought_result = await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, thought_args, True); assert thought_result.get("success")
                      except Exception as e: self.logger.error(f"Failed to record thought: {e}", exc_info=False); self.state.consecutive_error_count += 1; self.state.last_action_summary = f"Error recording thought: {str(e)[:100]}"; self.state.needs_replan = True; self.state.last_error_details = {"tool": TOOL_RECORD_THOUGHT, "error": str(e)}
                  else: self.logger.warning("No workflow to record thought."); self.state.last_action_summary = "Agent provided reasoning, but no workflow active."

             elif decision_type == "complete":
                  summary = agent_decision.get("summary", "Goal achieved."); self.logger.info(f"Agent signals completion: {summary}", emoji_key="tada"); self.state.goal_achieved_flag = True; self.state.needs_replan = False
                  if self.state.workflow_id: await self._update_workflow_status_internal("completed", summary)
                  break

             elif decision_type == "error":
                  error_msg = agent_decision.get("message", "Unknown agent error"); self.logger.error(f"Agent decision error: {error_msg}", emoji_key="x"); self.state.last_action_summary = f"Agent decision error: {error_msg[:100]}"; self.state.last_error_details = {"agent_decision_error": error_msg}; self.state.consecutive_error_count += 1; self.state.needs_replan = True
                  if self.state.workflow_id: 
                      try: 
                        await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"workflow_id": self.state.workflow_id, "content": f"Agent Error: {error_msg}", "thought_type": ThoughtType.CRITIQUE.value}, False) 
                      except Exception: pass

             else: self.logger.warning(f"Unhandled decision: {decision_type}"); self.state.last_action_summary = "Unknown agent decision."; self.state.consecutive_error_count += 1; self.state.needs_replan = True; self.state.last_error_details = {"agent_decision_error": f"Unknown type: {decision_type}"}

             # 4. Update Plan (Fallback if LLM didn't provide one)
             if not agent_decision.get("updated_plan_steps"):
                 await self._update_plan(context, agent_decision, last_tool_result_content)

             # 5. Periodic Tasks (Enhanced for Tier 3)
             await self._run_periodic_tasks()

             # 6. Save State Periodically
             if self.state.current_loop % 5 == 0: await self._save_agent_state()

             # 7. Loop Delay
             await asyncio.sleep(random.uniform(0.8, 1.2))

        # --- End of Loop ---
        self.logger.info("--- Agent Loop Finished ---", emoji_key="stopwatch")
        await self._cleanup_background_tasks(); await self._save_agent_state()
        if self.state.workflow_id and not self._shutdown_event.is_set():
            final_status = "completed" if self.state.goal_achieved_flag else "incomplete"
            self.logger.info(f"Workflow ended with status: {final_status}")
            await self._generate_final_report()
        elif not self.state.workflow_id: self.logger.info("Loop finished with no active workflow.")

    def _start_background_task(self, coro):
        """Creates and tracks a background task."""
        task = asyncio.create_task(coro); self.state.background_tasks.add(task); task.add_done_callback(self.state.background_tasks.discard)

    async def _cleanup_background_tasks(self):
        """Cancels and awaits completion of any running background tasks."""
        if self.state.background_tasks:
            self.logger.info(f"Cleaning up {len(self.state.background_tasks)} background tasks...")
            cancelled_tasks = []; [task.cancel() for task in list(self.state.background_tasks) if not task.done() and cancelled_tasks.append(task)]
            if cancelled_tasks: await asyncio.gather(*cancelled_tasks, return_exceptions=True); self.logger.debug(f"Cancelled {len(cancelled_tasks)} background tasks.")
            self.logger.info("Background tasks cleaned up."); self.state.background_tasks.clear()

    async def signal_shutdown(self):
        """Initiates graceful shutdown."""
        self.logger.info("Graceful shutdown signal received.", emoji_key="wave"); self._shutdown_event.set(); await self._cleanup_background_tasks()

    async def shutdown(self):
        """Performs final cleanup and state saving."""
        self.logger.info("Shutting down agent loop...", emoji_key="power_button"); self._shutdown_event.set(); await self._cleanup_background_tasks(); await self._save_agent_state(); self.logger.info("Agent loop shutdown complete.", emoji_key="checkered_flag")

    async def _update_workflow_status_internal(self, status: str, message: Optional[str] = None):
        """Internal helper to update workflow status via tool call."""
        if not self.state.workflow_id: return
        try: status_value = WorkflowStatus(status.lower()).value
        except ValueError: self.logger.warning(f"Invalid workflow status '{status}'. Using 'failed'."); status_value = WorkflowStatus.FAILED.value
        tool_name = TOOL_UPDATE_WORKFLOW_STATUS
        if not self._find_tool_server(tool_name): self.logger.error(f"Cannot update status: Tool {tool_name} unavailable."); return
        try: await self._execute_tool_call_internal(tool_name, {"workflow_id": self.state.workflow_id, "status": status_value, "completion_message": message}, record_action=False)
        except Exception as e: self.logger.error(f"Error marking workflow {self.state.workflow_id} as {status_value}: {e}", exc_info=False)

    async def _generate_final_report(self):
        """Generates and logs a final report using the memory tool."""
        if not self.state.workflow_id: return
        self.logger.info(f"Generating final report for workflow {self.state.workflow_id}...", emoji_key="scroll")
        tool_name = TOOL_GENERATE_REPORT
        if not self._find_tool_server(tool_name): self.logger.error(f"Cannot generate report: Tool {tool_name} unavailable."); return
        try:
            report_result_content = await self._execute_tool_call_internal(tool_name, {"workflow_id": self.state.workflow_id, "report_format": "markdown", "style": "professional"}, record_action=False)
            if isinstance(report_result_content, dict) and report_result_content.get("success"): report_text = report_result_content.get("report", "Report content missing."); self.mcp_client.safe_print("\n--- FINAL WORKFLOW REPORT ---\n" + report_text + "\n--- END REPORT ---")
            else: self.logger.error(f"Failed to generate final report: {report_result_content.get('error', 'Unknown error')}")
        except Exception as e: self.logger.error(f"Exception generating final report: {e}", exc_info=True)

    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Finds an active server providing the specified tool."""
        if self.mcp_client and self.mcp_client.server_manager:
            if tool_name in self.mcp_client.server_manager.tools:
                 server_name = self.mcp_client.server_manager.tools[tool_name].server_name
                 if server_name in self.mcp_client.server_manager.active_sessions: return server_name
                 else: self.logger.debug(f"Server '{server_name}' for tool '{tool_name}' is not active.")
            elif tool_name.startswith("core:") and "CORE" in self.mcp_client.server_manager.active_sessions: return "CORE"
        self.logger.debug(f"Tool '{tool_name}' not found on any active server.")
        return None

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """Checks if a workflow ID exists using list_workflows tool."""
        self.logger.debug(f"Checking existence of workflow {workflow_id} using list_workflows (potentially inefficient).")
        tool_name = TOOL_LIST_WORKFLOWS
        if not self._find_tool_server(tool_name): self.logger.error(f"Cannot check workflow: Tool {tool_name} unavailable."); return False
        try:
             result = await self._execute_tool_call_internal(tool_name, {"limit": 500}, record_action=False)
             if isinstance(result, dict) and result.get("success"): wf_list = result.get("workflows", []); return any(wf.get("workflow_id") == workflow_id for wf in wf_list)
             return False
        except Exception as e: self.logger.error(f"Error checking WF {workflow_id}: {e}"); return False

# --- Main Execution Block ---
async def run_agent_process(mcp_server_url: str, anthropic_key: str, goal: str, max_loops: int, state_file: str, config_file: Optional[str]):
    """Sets up and runs the agent process."""
    if not MCP_CLIENT_AVAILABLE: print("❌ ERROR: MCPClient dependency not met."); sys.exit(1)
    mcp_client_instance = None; agent_loop_instance = None; exit_code = 0; printer = print

    try:
        printer("Instantiating MCP Client...")
        mcp_client_instance = MCPClient(config_path=config_file)
        if hasattr(mcp_client_instance, 'safe_print'): printer = mcp_client_instance.safe_print
        if not mcp_client_instance.config.api_key:
            if anthropic_key: printer("Using provided Anthropic API key."); mcp_client_instance.config.api_key = anthropic_key; mcp_client_instance.anthropic = AsyncAnthropic(api_key=anthropic_key)
            else: raise ValueError("Anthropic API key missing.")
        printer("Setting up MCP Client...")
        await mcp_client_instance.setup(interactive_mode=False)
        printer("Instantiating Agent Master Loop...")
        agent_loop_instance = AgentMasterLoop(mcp_client_instance=mcp_client_instance, agent_state_file=state_file)

        loop = asyncio.get_running_loop()
        def signal_handler_wrapper(signum, frame):
            log.warning(f"Signal {signal.Signals(signum).name} received. Initiating graceful shutdown.")
            if agent_loop_instance: asyncio.create_task(agent_loop_instance.signal_shutdown())
            else: loop.stop()
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try: loop.add_signal_handler(sig, signal_handler_wrapper, sig, None)
            except ValueError: log.debug(f"Signal handler for {sig} already exists.")
            except NotImplementedError: log.warning(f"Signal handling for {sig} not supported on this platform.")

        printer("Running Agent Loop...")
        await agent_loop_instance.run(goal=goal, max_loops=max_loops)

    except KeyboardInterrupt: printer("\n[yellow]Agent loop interrupt handled by signal handler.[/yellow]"); exit_code = 130
    except Exception as main_err: printer(f"\n❌ Critical error: {main_err}"); log.critical("Top-level execution error", exc_info=True); exit_code = 1
    finally:
        printer("Initiating final shutdown sequence...")
        if agent_loop_instance: printer("Shutting down agent loop..."); await agent_loop_instance.shutdown()
        if mcp_client_instance: printer("Closing MCP client..."); await mcp_client_instance.close()
        printer("Agent execution finished.")
        if __name__ == "__main__": await asyncio.sleep(0.5); sys.exit(exit_code)

if __name__ == "__main__":
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8013")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    # --- Updated Goal for Tier 3 Testing ---
    AGENT_GOAL = os.environ.get("AGENT_GOAL",
        "Create workflow 'Tier 3 Test'. Goal: Research 'Quantum Computing impact on Cryptography'. "
        "Plan: 1. Create a new thought chain for 'Cryptography Research'. 2. Search memory for existing info (hybrid search). 3. Perform simulated web search (store results as memory). 4. Consolidate findings. 5. Reflect on progress and potential gaps. 6. Mark workflow complete."
    )
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30")) # Increased slightly for more complex goal
    AGENT_STATE_FILENAME = os.environ.get("AGENT_STATE_FILE", AGENT_STATE_FILE)
    MCP_CLIENT_CONFIG_FILE = os.environ.get("MCP_CLIENT_CONFIG")

    if not ANTHROPIC_API_KEY: print("❌ ERROR: ANTHROPIC_API_KEY missing."); sys.exit(1)
    if not MCP_CLIENT_AVAILABLE: print("❌ ERROR: MCPClient dependency missing."); sys.exit(1)

    print(f"--- {AGENT_NAME} (Tier 1, 2 & 3) ---") # Updated name
    print(f"Memory System URL: {MCP_SERVER_URL}")
    print(f"Agent Goal: {AGENT_GOAL}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"State File: {AGENT_STATE_FILENAME}")
    print(f"Client Config: {MCP_CLIENT_CONFIG_FILE or 'Default'}")
    print(f"Log Level: {log.level}")
    print("Anthropic API Key: Found")
    print("-----------------------------------------")

    # --- Tool Simulation Setup ---
    async def simulate_web_search(query: str):
        log.info(f"[SIMULATED] Searching web for: {query}")
        await asyncio.sleep(0.5)
        # Simulate finding some relevant snippets
        results = [
            f"Quantum computers threaten RSA encryption due to Shor's algorithm. (Source: Tech Journal)",
            f"Post-quantum cryptography (PQC) standards are being developed by NIST. (Source: NIST website)",
            f"Lattice-based cryptography is a leading candidate for PQC. (Source: Crypto Conf paper)"
        ]
        return {"success": True, "search_results": results}

    async def setup_and_run():
        """Wrapper to setup client and potentially register simulated tools."""
        # Placeholder for tool registration (adapt to your MCPClient)
        # client = MCPClient(...)
        # await client.register_tool_function("simulate:web_search", simulate_web_search)
        # await client.setup(...)
        # await run_agent_process(...) using this client instance
        await run_agent_process(MCP_SERVER_URL, ANTHROPIC_API_KEY, AGENT_GOAL, MAX_ITERATIONS, AGENT_STATE_FILENAME, MCP_CLIENT_CONFIG_FILE)

        asyncio.run(setup_and_run())
