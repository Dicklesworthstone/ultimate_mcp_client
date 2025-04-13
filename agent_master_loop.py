"""
Supercharged Agent Master Loop - v3.0 (Production Candidate)
===========================================================

Enhanced orchestrator for AI agents using the Unified Memory System
via the Ultimate MCP Client. Implements structured planning, dynamic context,
dependency checking, error recovery, feedback loops, and meta-cognition.

Designed for Claude 3.7 Sonnet (or comparable models with tool use).
"""

import asyncio
import dataclasses
import json
import logging
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
from anthropic.types import AsyncAnthropic, Message  # Removed ToolUseBlock - process via content list
from pydantic import BaseModel, Field, ValidationError  # For structured plans

# --- IMPORT YOUR ACTUAL MCP CLIENT and COMPONENTS ---
try:
    from mcp_client import (
        ActionStatus,
        LinkType,  # noqa: F401
        MCPClient,
        MemoryLevel,
        MemoryType,  # noqa: F401
        MemoryUtils,
        ThoughtType,  # noqa: F401
        ToolError,
        ToolInputError,
        WorkflowStatus,
    )
    MCP_CLIENT_AVAILABLE = True
    log = logging.getLogger("SuperchargedAgentMasterLoop") # Use project logger if MCPClient sets it up
    if not log.handlers: raise ImportError("Logger not configured by MCPClient import")
    log.info("Successfully imported MCPClient and required components.")
except ImportError as import_err:
    print(f"❌ CRITICAL ERROR: Could not import MCPClient or required components: {import_err}")
    print("Ensure mcp_client.py is correctly structured and in the Python path.")
    sys.exit(1) # Exit if client cannot be imported

# --- Logging Setup Refinement ---
# Ensure log level is set based on environment variable
log_level_str = os.environ.get("AGENT_LOOP_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
log.setLevel(log_level)
if log_level == logging.DEBUG:
    log.info("Agent loop verbose logging enabled.")

# --- Constants ---
# File Paths & Identifiers
AGENT_STATE_FILE = "agent_loop_state_v3.0.json"
AGENT_NAME = "Maestro-v3"
# Meta-cognition & Maintenance Intervals/Thresholds
REFLECTION_SUCCESS_THRESHOLD = int(os.environ.get("REFLECTION_THRESHOLD", "7")) # Reduced default
CONSOLIDATION_SUCCESS_THRESHOLD = int(os.environ.get("CONSOLIDATION_THRESHOLD", "12")) # Reduced default
OPTIMIZATION_LOOP_INTERVAL = int(os.environ.get("OPTIMIZATION_INTERVAL", "8")) # Reduced default
MEMORY_PROMOTION_LOOP_INTERVAL = int(os.environ.get("PROMOTION_INTERVAL", "15"))
AUTO_LINKING_DELAY_SECS = (1.5, 3.0) # Range for random delay before auto-linking
# Context & Planning
DEFAULT_PLAN_STEP = "Assess goal, gather context, formulate initial plan."
CONTEXT_RECENT_ACTIONS = 7
CONTEXT_IMPORTANT_MEMORIES = 5
CONTEXT_KEY_THOUGHTS = 5
CONTEXT_PROCEDURAL_MEMORIES = 2 # How many relevant procedures to fetch
CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD = 15000 # Approx token limit before trying compression
CONTEXT_COMPRESSION_TARGET_TOKENS = 5000
# Error Handling
MAX_CONSECUTIVE_ERRORS = 3
# Tool Names (Ensure these match registration in your MCP Client / Memory Server)
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context"
TOOL_CREATE_WORKFLOW = "unified_memory:create_workflow"
TOOL_UPDATE_WORKFLOW_STATUS = "unified_memory:update_workflow_status"
TOOL_RECORD_ACTION_START = "unified_memory:record_action_start"
TOOL_RECORD_ACTION_COMPLETION = "unified_memory:record_action_completion"
TOOL_RECORD_THOUGHT = "unified_memory:record_thought"
TOOL_ADD_DEPENDENCY = "unified_memory:add_action_dependency"
TOOL_GET_DEPENDENCIES = "unified_memory:get_action_dependencies"
TOOL_REFLECTION = "unified_memory:generate_reflection"
TOOL_CONSOLIDATION = "unified_memory:consolidate_memories"
TOOL_OPTIMIZE_WM = "unified_memory:optimize_working_memory"
TOOL_AUTO_FOCUS = "unified_memory:auto_update_focus"
TOOL_PROMOTE_MEM = "unified_memory:promote_memory_level"
TOOL_QUERY_MEMORIES = "unified_memory:query_memories"
TOOL_SEMANTIC_SEARCH = "unified_memory:search_semantic_memories"
TOOL_STORE_MEMORY = "unified_memory:store_memory"
TOOL_CREATE_LINK = "unified_memory:create_memory_link"
TOOL_GET_MEMORY_BY_ID = "unified_memory:get_memory_by_id"
TOOL_GET_ACTION_DETAILS = "unified_memory:get_action_details" # Assumed needed for dep check
TOOL_GET_LINKED_MEMORIES = "unified_memory:get_linked_memories" # For context expansion
TOOL_LIST_WORKFLOWS = "unified_memory:list_workflows"
TOOL_GENERATE_REPORT = "unified_memory:generate_workflow_report"
TOOL_COMPUTE_STATS = "unified_memory:compute_memory_statistics" # For adaptive meta-cognition
TOOL_SUMMARIZE_TEXT = "unified_memory:summarize_text"
CONTEXT_PROACTIVE_MEMORIES = 3 # How many memories to fetch proactively based on plan

# --- Structured Plan Model (#1) ---
class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: f"step-{MemoryUtils.generate_id()[:8]}")
    description: str
    status: str = Field(default="planned", description="Status: planned, in_progress, completed, failed, skipped")
    depends_on: List[str] = Field(default_factory=list, description="List of action IDs this step requires")
    assigned_tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    is_parallel_group: Optional[str] = None # ID linking parallel steps

# --- Agent State Dataclass (Enhanced) ---
def _default_tool_stats():
    return defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0})

@dataclass
class AgentState:
    # Core State
    workflow_id: Optional[str] = None
    context_id: Optional[str] = None # Usually mirrors workflow_id
    workflow_stack: List[str] = field(default_factory=list) # For sub-workflows (#2)
    current_plan: List[PlanStep] = field(default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)]) # Structured plan (#1)
    current_sub_goal_id: Optional[str] = None # Track sub-goal focus (#2)
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False
    # Error & Replanning State
    consecutive_error_count: int = 0
    needs_replan: bool = False
    last_error_details: Optional[Dict] = None # Store structured error info (#10)
    # Meta-Cognition State
    successful_actions_since_reflection: int = 0
    successful_actions_since_consolidation: int = 0
    loops_since_optimization: int = 0
    loops_since_promotion_check: int = 0
    reflection_cycle_index: int = 0
    last_meta_feedback: Optional[str] = None # Store feedback from reflection/consolidation (#13)
    # Stats & Tracking
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=_default_tool_stats) # Enhanced stats (#5)
    # Background task tracking (transient, not saved)
    background_tasks: Set[asyncio.Task] = field(default_factory=set, init=False, repr=False)


# --- Agent Loop Class ---
class AgentMasterLoop:
    """Supercharged orchestrator using Unified Memory and MCP Client."""

    def __init__(self, mcp_client_instance: MCPClient, agent_state_file: str = AGENT_STATE_FILE):
        if not MCP_CLIENT_AVAILABLE: raise RuntimeError("MCPClient class unavailable.")

        self.mcp_client = mcp_client_instance
        self.anthropic_client = self.mcp_client.anthropic
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        # Add all required configuration attributes
        self.reflection_threshold = REFLECTION_SUCCESS_THRESHOLD
        self.consolidation_threshold = CONSOLIDATION_SUCCESS_THRESHOLD
        self.optimization_interval = OPTIMIZATION_LOOP_INTERVAL
        self.promotion_interval = MEMORY_PROMOTION_LOOP_INTERVAL
        
        # Define reflection type sequence
        self.reflection_type_sequence = ["summary", "progress", "gaps", "strengths", "plan"]
        
        # Define consolidation settings
        self.consolidation_memory_level = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources = 10
        
        # Auto-linking settings
        self.auto_linking_threshold = 0.7
        self.auto_linking_max_links = 3
        
        if not self.anthropic_client:
            self.logger.critical("Anthropic client unavailable! Agent cannot function.")
            raise ValueError("Anthropic client required.")

        self.state = AgentState()
        if self.state.workflow_id and not self.state.context_id:
            self.state.context_id = self.state.workflow_id
        self._shutdown_event = asyncio.Event()
        self.tool_schemas: List[Dict[str, Any]] = []
        self._active_tasks: Set[asyncio.Task] = set()  # Track all running tasks

    async def initialize(self) -> bool:
        """Initializes loop state, loads previous state, verifies client setup."""
        self.logger.info("Initializing agent loop...", emoji_key="gear")
        # Load persistent state first
        await self._load_agent_state()
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
            if not self.tool_schemas: self.logger.warning("No 'unified_memory:*' tools loaded.", emoji_key="warning")
            else: self.logger.info(f"Loaded {len(self.tool_schemas)} unified_memory tool schemas.", emoji_key="clipboard")

            # Verify essential tools
            essential_tools = [TOOL_GET_CONTEXT, TOOL_CREATE_WORKFLOW, TOOL_RECORD_THOUGHT, TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION]
            missing_essential = [t for t in essential_tools if not self._find_tool_server(t)]
            if missing_essential:
                self.logger.error(f"Missing essential tools: {missing_essential}. Agent cannot function reliably.")
                # Decide if this is fatal. Let's make it fatal for now.
                # return False
                self.logger.warning("Proceeding without essential tools - functionality may be impaired.")


            # Check workflow validity
            current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
            if current_workflow_id and not await self._check_workflow_exists(current_workflow_id):
                self.logger.warning(f"Loaded workflow {current_workflow_id} not found. Resetting state.")
                await self._reset_state_to_defaults()
                await self._save_agent_state() # Save reset state

            self.logger.info("Agent loop initialized successfully.")
            return True
        except Exception as e:
            self.logger.critical(f"Agent loop initialization failed: {e}", exc_info=True)
            return False

    # --- Accurate Token Counting Helper (Using Anthropic Client) ---
    async def _estimate_tokens_anthropic(self, data: Any) -> int:
        """Estimates token count for arbitrary data structures using the Anthropic client."""
        if data is None: return 0
        if not self.anthropic_client:
             self.logger.warning("Cannot estimate tokens: Anthropic client not available.")
             # Fallback to very rough character estimate if client missing
             try: 
                 return len(json.dumps(data, default=str)) // 4
             except Exception: 
                 return 0

        token_count = 0
        try:
            # Convert data to a text representation suitable for counting.
            # JSON is a reasonable approach for structured data.
            # For simple strings, count directly.
            if isinstance(data, str):
                text_representation = data
            else:
                text_representation = json.dumps(data, ensure_ascii=False, default=str)

            # Use the Anthropic client's count_tokens method
            token_count = await self.anthropic_client.count_tokens(text_representation)
            # self.logger.debug(f"Estimated tokens for data (type {type(data)}): {token_count}")
            return token_count
        except anthropic.APIError as e:
             self.logger.warning(f"Anthropic API error during token counting: {e}. Using fallback estimate.")
        except Exception as e:
            self.logger.warning(f"Token estimation failed for data type {type(data)}: {e}. Using fallback estimate.")

        # Fallback to character estimate if API call fails
        try:
             text_representation = json.dumps(data, default=str) if not isinstance(data, str) else data
             return len(text_representation) // 4
        except Exception:
             return 0 # Final fallback
        
    async def _save_agent_state(self):
        """Saves the agent loop's state to a JSON file."""
        state_dict = dataclasses.asdict(self.state)
        state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Convert complex types for saving
        state_dict.pop("background_tasks", None) # Don't save tasks
        state_dict["tool_usage_stats"] = dict(state_dict["tool_usage_stats"])
        # Convert PlanStep objects to dicts
        state_dict["current_plan"] = [step.model_dump() for step in self.state.current_plan]

        try:
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.agent_state_file, 'w') as f:
                await f.write(json.dumps(state_dict, indent=2))
            self.logger.debug(f"Agent state saved to {self.agent_state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)

    async def _load_agent_state(self):
        """Loads state, converting plan back to PlanStep objects."""
        if not self.agent_state_file.exists():
            self.logger.info("No previous agent state file found. Using default state.")
            self.state = AgentState()
            return
        try:
            async with aiofiles.open(self.agent_state_file, 'r') as f:
                state_data = json.loads(await f.read())

            # Create AgentState instance, handling potential missing keys
            kwargs = {}
            for field_info in dataclasses.fields(AgentState):
                 if field_info.name in state_data:
                     # Special handling for nested structures
                     if field_info.name == "current_plan":
                         try:
                             # Convert list of dicts back to PlanStep objects
                             kwargs["current_plan"] = [PlanStep(**step_data) for step_data in state_data["current_plan"]]
                         except (ValidationError, TypeError) as plan_err:
                              log.warning(f"Failed to parse saved plan, resetting: {plan_err}")
                              kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)] # Default plan
                     elif field_info.name == "tool_usage_stats":
                          kwargs["tool_usage_stats"] = defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0}, state_data["tool_usage_stats"])
                     else:
                          kwargs[field_info.name] = state_data[field_info.name]
                 else:
                     # Use default factory if key missing
                     if field_info.default_factory is not dataclasses.MISSING:
                         kwargs[field_info.name] = field_info.default_factory()
                     elif field_info.default is not dataclasses.MISSING:
                         kwargs[field_info.name] = field_info.default
                     # else: # No default, will be None or raise error if required later

            self.state = AgentState(**kwargs)
            self.logger.info(f"Agent state loaded from {self.agent_state_file}. Loop {self.state.current_loop}. WF: {self.state.workflow_id}")

        except Exception as e:
            self.logger.error(f"Failed to load/parse agent state: {e}. Resetting.", exc_info=True)
            await self._reset_state_to_defaults()

    async def _reset_state_to_defaults(self):
        self.state = AgentState() # Re-init with defaults
        self.logger.warning("Agent state has been reset to defaults.")

    # --- Enhanced Context Gathering (v3.0 - Corrected Tool Names & Token Counting) ---
    async def _gather_context(self) -> Dict[str, Any]:
        """Gathers comprehensive context for the agent LLM, using exact tool names
           and accurate token counting."""
        self.logger.info("Gathering context...", emoji_key="satellite")
        base_context = {
            # Core agent state (unchanged)
            "current_loop": self.state.current_loop,
            "current_plan": [step.model_dump() for step in self.state.current_plan],
            "last_action_summary": self.state.last_action_summary,
            "consecutive_errors": self.state.consecutive_error_count,
            "last_error_details": self.state.last_error_details,
            "workflow_stack": self.state.workflow_stack,
            "meta_feedback": self.state.last_meta_feedback,
            # Placeholders
            "core_context": None, "proactive_memories": [], "relevant_procedures": [],
            "contextual_links": None, "compression_summary": None,
            "status": "Gathering...", "errors": []
        }
        self.state.last_meta_feedback = None # Clear after reading

        current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_workflow_id:
            base_context["status"] = "No Active Workflow"; base_context["message"] = "Create/load workflow."; return base_context

        # Define exact tool names (matching unified_memory.py __all__)
        get_context_tool = "unified_memory:get_workflow_context"
        search_semantic_tool = "unified_memory:search_semantic_memories"
        summarize_tool = TOOL_SUMMARIZE_TEXT
        # get_linked_memories_tool = "unified_memory:get_linked_memories" # If you implement #15

        # --- 1. Goal-Directed Proactive Memory Retrieval (#7) ---
        active_plan_step_desc = self.state.current_plan[0].description if self.state.current_plan else "Achieve main goal"
        proactive_query = f"Information relevant to planning or executing: {active_plan_step_desc}"
        try:
            result_content = await self._execute_tool_call_internal(
                search_semantic_tool, # Use exact name
                {"workflow_id": current_workflow_id, "query": proactive_query, "limit": CONTEXT_PROACTIVE_MEMORIES, "include_content": False},
                record_action=False
            )
            # Assuming 'memories' list is directly in result_content on success
            if result_content.get("success"):
                proactive_mems = result_content.get("memories", [])
                base_context["proactive_memories"] = [
                    {"memory_id": m.get("memory_id"), "description": m.get("description"), "similarity": m.get("similarity"), "type": m.get("memory_type")}
                    for m in proactive_mems
                ]
                if base_context["proactive_memories"]: self.logger.info(f"Retrieved {len(base_context['proactive_memories'])} proactive memories.")
            else: base_context["errors"].append(f"Proactive memory search failed: {result_content.get('error')}")
        except Exception as e: self.logger.warning(f"Proactive memory search exception: {e}"); base_context["errors"].append(f"Proactive search exception: {e}")

        # --- 2. Fetch Core Context via Tool ---
        try:
            core_context_result = await self._execute_tool_call_internal(
                get_context_tool, { # Use exact name
                    "workflow_id": current_workflow_id,
                    "recent_actions_limit": CONTEXT_RECENT_ACTIONS,
                    "important_memories_limit": CONTEXT_IMPORTANT_MEMORIES,
                    "key_thoughts_limit": CONTEXT_KEY_THOUGHTS
                }, record_action=False
            )
            if core_context_result.get("success"):
                # Store the actual *content* returned by the tool
                base_context["core_context"] = core_context_result # Contains the nested data
                base_context["core_context"].pop("success", None) # Remove redundant keys if present
                base_context["core_context"].pop("processing_time", None)
                # Workflow ID is already known, no need to store it again here
                self.logger.info("Core context retrieved.")
            else: base_context["errors"].append(f"Core context retrieval failed: {core_context_result.get('error')}")
        except Exception as e: self.logger.warning(f"Core context retrieval exception: {e}"); base_context["errors"].append(f"Core context exception: {e}")

        # --- 3. Fetch Relevant Procedural Memories (#6) ---
        try:
             proc_query = f"How to accomplish: {active_plan_step_desc}"
             proc_result = await self._execute_tool_call_internal(
                  search_semantic_tool, { # Use exact name
                       "workflow_id": current_workflow_id, "query": proc_query, "limit": CONTEXT_PROCEDURAL_MEMORIES,
                       "memory_level": MemoryLevel.PROCEDURAL.value, # Use Enum value
                       "include_content": False
                  }, record_action=False
             )
             # Assuming 'memories' list is directly in proc_result on success
             if proc_result.get("success"):
                 proc_mems = proc_result.get("memories", [])
                 base_context["relevant_procedures"] = [
                     {"memory_id": m.get("memory_id"), "description": m.get("description"), "similarity": m.get("similarity")}
                     for m in proc_mems
                 ]
                 if base_context["relevant_procedures"]: self.logger.info(f"Retrieved {len(base_context['relevant_procedures'])} relevant procedures.")
             else: base_context["errors"].append(f"Procedure search failed: {proc_result.get('error')}")
        except Exception as e: self.logger.warning(f"Procedure search exception: {e}"); base_context["errors"].append(f"Procedure search exception: {e}")

        # --- 4. Context Compression (#9 - Requires summarize_text tool) ---
        try:
            # Use the *accurate* token counter
            estimated_tokens = await self._estimate_tokens_anthropic(base_context)
            if estimated_tokens > CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD:
                self.logger.warning(f"Context ({estimated_tokens} tokens) exceeds threshold {CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD}. Attempting compression.")
                # Check if the summarization tool is available
                if self._find_tool_server(summarize_tool):
                    # Identify parts to summarize (e.g., older actions)
                    actions_to_summarize = []
                    if base_context.get("core_context") and isinstance(base_context["core_context"].get("recent_actions"), list):
                        actions = base_context["core_context"]["recent_actions"]
                        if len(actions) > 3:
                            actions_to_summarize = actions[len(actions)//2:]
                            # Create text block from actions
                            text_block = f"Recent Actions Log (Older Half):\n"
                            for action in actions_to_summarize:
                                 text_block += f"- Seq {action.get('sequence_number')}: {action.get('title')} ({action.get('status')})\n"
                                 if action.get('reasoning'): text_block += f"  Reason: {action.get('reasoning')[:100]}...\n"
                                 if action.get('tool_result'): text_block += f"  Result: {str(action.get('tool_result'))[:100]}...\n"

                            # Call the summarization tool
                            summary_result = await self._execute_tool_call_internal(
                                TOOL_SUMMARIZE_TEXT, {
                                    "text_to_summarize": text_block,
                                    "target_tokens": CONTEXT_COMPRESSION_TARGET_TOKENS,
                                    "workflow_id": current_workflow_id,  # Add this parameter
                                    "record_summary": False  # Set to false for temporary summaries
                                }, record_action=False
                            )
                            if summary_result.get("success"):
                                # Replace older actions with summary note
                                if base_context.get("core_context"):
                                     base_context["core_context"]["recent_actions"] = actions[:len(actions)//2] # Keep newer half
                                base_context["compression_summary"] = f"Summary of older actions: {summary_result.get('summary', 'Summary failed.')[:150]}..."
                                self.logger.info("Compressed older context actions.")
                                # Re-estimate tokens after compression
                                estimated_tokens = await self._estimate_tokens_anthropic(base_context) # After potential compression
                                self.logger.info(f"Context size after compression: {estimated_tokens} est. tokens")
                            else: base_context["errors"].append(f"Context compression failed: {summary_result.get('error')}")
                else:
                    self.logger.warning(f"Cannot compress context: Tool '{summarize_tool}' unavailable.")
        except Exception as e:
            self.logger.error(f"Error during context compression check: {e}", exc_info=False)
            base_context["errors"].append(f"Compression exception: {e}")

        # --- 5. Contextual Link Traversal (#15 - Implemented) ---
        base_context["contextual_links"] = None # Initialize
        get_linked_memories_tool = "unified_memory:get_linked_memories" # Use exact tool name

        # Check if the tool is available
        if self._find_tool_server(get_linked_memories_tool):
            # Identify the most important memory from the core context results
            mem_id_to_traverse = None
            important_mem_list = base_context.get("core_context", {}).get("important_memories", [])
            if important_mem_list and isinstance(important_mem_list, list) and len(important_mem_list) > 0:
                # Ensure we access the dictionary correctly
                first_mem = important_mem_list[0]
                if isinstance(first_mem, dict):
                    mem_id_to_traverse = first_mem.get("memory_id")

            if mem_id_to_traverse:
                self.logger.debug(f"Attempting link traversal from important memory: {mem_id_to_traverse[:8]}...")
                try:
                    links_result_content = await self._execute_tool_call_internal(
                        get_linked_memories_tool,
                        { # Arguments based on the expected tool schema
                            "memory_id": mem_id_to_traverse,
                            "direction": "both",
                            "limit": 3 # Fetch a few neighbors in each direction
                        },
                        record_action=False # Don't log context fetching as primary action
                    )

                    if links_result_content.get("success"):
                        # Parse the links from the result content
                        # Assuming the tool returns {'success': True, 'links': {'outgoing': [...], 'incoming': [...]}}
                        links_data = links_result_content.get("links", {})
                        outgoing_links = links_data.get("outgoing", [])
                        incoming_links = links_data.get("incoming", [])

                        # Format a concise summary for the context prompt
                        link_summary = {
                            "source_memory_id": mem_id_to_traverse,
                            "outgoing_count": len(outgoing_links),
                            "incoming_count": len(incoming_links),
                            # Include brief details of top links (e.g., type and target/source description)
                            "top_links_summary": []
                        }

                        # Add top outgoing links summary
                        for link in outgoing_links[:2]: # Limit details shown
                            link_summary["top_links_summary"].append(
                                f"OUT: {link.get('link_type', 'related')} -> "
                                f"{link.get('target_type','Mem')} '{str(link.get('target_description','?'))[:30]}...' "
                                f"(ID: {str(link.get('target_memory_id','?'))[:6]}...)"
                            )
                        # Add top incoming links summary
                        for link in incoming_links[:2]: # Limit details shown
                            link_summary["top_links_summary"].append(
                                f"IN: {link.get('link_type', 'related')} <- "
                                f"{link.get('source_type','Mem')} '{str(link.get('source_description','?'))[:30]}...' "
                                f"(ID: {str(link.get('source_memory_id','?'))[:6]}...)"
                            )

                        base_context["contextual_links"] = link_summary # Add the formatted summary
                        self.logger.info(f"Retrieved {len(outgoing_links)} outgoing, {len(incoming_links)} incoming links for memory {mem_id_to_traverse[:8]}...")
                    else:
                        err_msg = f"Link retrieval tool failed: {links_result_content.get('error', 'Unknown')}"
                        base_context["errors"].append(err_msg)
                        self.logger.warning(err_msg)
                except Exception as e:
                    err_msg = f"Link retrieval exception: {e}"
                    self.logger.warning(err_msg, exc_info=False)
                    base_context["errors"].append(err_msg)
            else:
                self.logger.debug("No important memory found in core context to perform link traversal from.")
        else:
            self.logger.debug(f"Skipping link traversal: Tool '{get_linked_memories_tool}' unavailable.")


    async def _fetch_proactive_memories(self, workflow_id: str, query: str) -> List[Dict]:
        """Fetches potentially relevant memories based on current plan/query."""
        try:
            search_result = await self._execute_tool_call_internal(
                TOOL_SEMANTIC_SEARCH,
                {"workflow_id": workflow_id, "query": query, "limit": CONTEXT_PROACTIVE_MEMORIES, "include_content": False},
                record_action=False
            )
            if search_result.get("success"):
                # Extract only essential info for context prompt
                return [
                    {"memory_id": m.get("memory_id"), "description": m.get("description"), "similarity": m.get("similarity")}
                    for m in search_result.get("memories", [])
                ]
        except Exception as e: self.logger.warning(f"Proactive memory search failed: {e}")
        return []

    # --- Prompt Construction (Enhanced) ---
    def _construct_agent_prompt(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        system_prompt = f"""You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System. Achieve the Overall Goal by strategically using the provided memory tools.

Overall Goal: {goal}

Available Unified Memory Tools (Use ONLY these):
"""
        if not self.tool_schemas: system_prompt += "- CRITICAL WARNING: No tools loaded.\n"
        else:
             for schema in self.tool_schemas:
                 sanitized = schema['name']; original = self.mcp_client.server_manager.sanitized_to_original.get(sanitized, 'Unknown')
                 system_prompt += f"\n- Name: `{sanitized}` (Represents: `{original}`)\n"
                 system_prompt += f"  Desc: {schema.get('description', 'N/A')}\n"; system_prompt += f"  Schema: {json.dumps(schema['input_schema'])}\n"
        system_prompt += """
Your Process:
1.  Context Analysis: Deeply analyze 'Current Context'. Note workflow status, errors (`last_error_details`), recent actions, memories (note `importance`/`confidence`), thoughts, `current_plan` (structured steps), `proactive_memories`, and `meta_feedback`.
2.  Error Handling: If `last_error_details` exists, **FIRST** reason about the error and propose a recovery strategy in your Reasoning & Planning step.
3.  Reasoning & Planning:
    a. State step-by-step reasoning towards the Goal/Sub-goal, integrating context and feedback.
    b. Evaluate `current_plan`. Is it valid? Does it address errors? Are dependencies met?
    c. Propose an **Updated Plan** (1-3 structured `PlanStep` JSON objects: {"id": "optional_id", "description": "...", "status": "planned", "depends_on": ["action_id_1"]}). If replanning, explain why. Use `record_thought(thought_type='plan')` for complex planning.
4.  Action Decision: Choose **ONE** action based on the *first planned step* in your Updated Plan:
    *   Call Memory Tool: Select the most precise `unified_memory:*` tool. Provide args per schema. **Mandatory:** Call `create_workflow` if context shows 'No Active Workflow'. Consider `add_action_dependency` for planned dependencies. Consider `get_action_dependencies` before critical steps.
    *   Record Thought: Use `record_thought` for logging reasoning, questions, hypotheses, critiques.
    *   Signal Completion: If Overall Goal is MET, respond ONLY with "Goal Achieved:" and summary.
5.  Output Format: Respond **ONLY** with the valid JSON for the chosen tool call OR "Goal Achieved:" text.

Key Considerations:
*   Use memory confidence (#8): Be cautious with low-confidence memories.
*   Consider known procedures/patterns from context (#6).
*   Suggest memory links (`create_memory_link`) when you identify relationships (#15).
*   Dependencies (#3): Ensure `depends_on` actions are likely complete before planning dependent steps.
"""
        # Prepare context string carefully
        context_str = json.dumps(context, indent=2, default=str, ensure_ascii=False); max_context_len = 18000 # ~6k tokens
        if len(context_str) > max_context_len: context_str = context_str[:max_context_len] + "\n... (Context Truncated)\n}"; self.logger.warning("Truncated context string sent to LLM.")
        user_prompt = f"Current Context:\n```json\n{context_str}\n```\n\n"; user_prompt += f"My Current Plan (Structured):\n```json\n{json.dumps([s.model_dump() for s in self.state.current_plan], indent=2)}\n```\n\n"; user_prompt += f"Last Action Summary:\n{self.state.last_action_summary}\n\n"
        if self.state.last_error_details: user_prompt += f"**CRITICAL: Address Last Error:**\n```json\n{json.dumps(self.state.last_error_details, indent=2)}\n```\n\n"
        if self.state.last_meta_feedback: user_prompt += f"**Meta-Cognitive Feedback:**\n{self.state.last_meta_feedback}\n\n"
        user_prompt += f"Overall Goal: {goal}\n\n"; user_prompt += "**Instruction:** Analyze context & errors. Reason step-by-step. Update plan (output structured JSON plan steps). Decide ONE action based on the *first* planned step (Tool JSON or 'Goal Achieved:')."
        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]

    async def _call_agent_llm(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Claude 3.7 Sonnet, includes structured plan parsing."""
        # (LLM call logic mostly unchanged from v2.1)
        # --- Modification: Parse structured plan from text response ---
        self.logger.info("Calling Agent LLM (Claude 3.7 Sonnet) for decision/plan...", emoji_key="robot_face")
        if not self.anthropic_client: return {"decision": "error", "message": "Anthropic client unavailable."}
        messages = self._construct_agent_prompt(goal, context)
        api_tools = self.tool_schemas # Assumes already filtered

        try:
            response: Message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620", max_tokens=4000, # Increased further
                messages=messages, tools=api_tools, tool_choice={"type": "auto"}, temperature=0.4 # Slightly lower temp
            )
            self.logger.debug(f"LLM Raw Response Stop Reason: {response.stop_reason}")

            decision = {"decision": "error", "message": "LLM provided no actionable output."}; text_parts = []; tool_call = None; updated_plan_steps = None

            for block in response.content:
                if block.type == "text": text_parts.append(block.text)
                elif block.type == "tool_use": tool_call = block; break # Prioritize tool call

            full_text = "".join(text_parts).strip()

            # --- Parse Updated Plan from Text (#1) ---
            # Look for a JSON block representing the plan steps
            plan_match = re.search(r"Updated Plan:\s*```json\s*([\s\S]+?)\s*```", full_text, re.IGNORECASE)
            if plan_match:
                plan_json_str = plan_match.group(1).strip()
                try:
                    plan_data = json.loads(plan_json_str)
                    # Validate and convert to PlanStep objects
                    if isinstance(plan_data, list):
                        validated_plan = [PlanStep(**step_data) for step_data in plan_data]
                        updated_plan_steps = validated_plan
                        self.logger.info(f"LLM proposed updated plan with {len(updated_plan_steps)} steps.")
                    else: self.logger.warning("LLM plan update was not a list.")
                except (json.JSONDecodeError, ValidationError, TypeError) as e:
                    self.logger.warning(f"Failed to parse structured plan from LLM response: {e}")
            else: self.logger.debug("No structured 'Updated Plan:' block found in LLM text.")

            # --- Determine Final Decision ---
            if tool_call:
                tool_name_sanitized = tool_call.name; tool_input = tool_call.input or {}
                original_tool_name = self.mcp_client.server_manager.sanitized_to_original.get(tool_name_sanitized, tool_name_sanitized)
                self.logger.info(f"LLM chose tool: {original_tool_name}", emoji_key="hammer_and_wrench")
                if original_tool_name.startswith("unified_memory:"):
                    decision = {"decision": "call_tool", "tool_name": original_tool_name, "arguments": tool_input}
                else: # Non-memory tool or error
                    self.logger.warning(f"LLM called non-unified_memory tool '{original_tool_name}'. Treating as thought.")
                    decision = {"decision": "thought_process", "content": full_text}
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

        # (Error Handling Unchanged)
        except anthropic.APIConnectionError as e: msg = f"API Connection Error: {e}"; self.logger.error(msg, exc_info=True)
        except anthropic.RateLimitError: msg = "Rate limit exceeded."; self.logger.error(msg, exc_info=True); await asyncio.sleep(random.uniform(5, 10))
        except anthropic.APIStatusError as e: msg = f"API Error {e.status_code}: {e.message}"; self.logger.error(f"Anthropic API status error: {e.status_code} - {e.response}", exc_info=True)
        except Exception as e: msg = f"Unexpected LLM interaction error: {e}"; self.logger.error(msg, exc_info=True)
        return {"decision": "error", "message": msg}

    async def _check_and_trigger_promotion(self, memory_id: str):
        """Check if a memory meets criteria for promotion and trigger if it does."""
        try:
            if not memory_id:
                return
                
            # Add a small delay to avoid overwhelming the system
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Use the promote_memory_level tool
            promotion_result = await self._execute_tool_call_internal(
                TOOL_PROMOTE_MEM,
                {
                    "memory_id": memory_id,
                    # Let the server decide if criteria are met based on defaults
                    "min_access_count_episodic": 5,  # Example threshold
                    "min_confidence_episodic": 0.8
                },
                record_action=False
            )
            
            if promotion_result.get("success") and promotion_result.get("promoted"):
                self.logger.info(
                    f"Memory {memory_id[:8]} promoted from {promotion_result.get('previous_level')} to {promotion_result.get('new_level')}",
                    emoji_key="arrow_up"
                )
                
        except Exception as e:
            self.logger.warning(f"Error in memory promotion check: {e}", exc_info=False)

    async def _run_auto_linking(self, memory_id: str):
        """Background task to automatically link a new memory to relevant existing memories."""
        try:
            if not memory_id or not self.state.workflow_id:
                return
                
            # Add a small random delay to avoid overwhelming the system
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            
            # Use semantic search to find similar memories
            self.logger.debug(f"Attempting auto-linking for memory {memory_id[:8]}...")
            
            # First, get the content of the memory to use as query
            mem_details = await self._execute_tool_call_internal(
                TOOL_GET_MEMORY_BY_ID,
                {"memory_id": memory_id, "include_links": False},
                record_action=False
            )
            
            if not mem_details.get("success"):
                self.logger.warning(f"Auto-linking failed: couldn't retrieve memory {memory_id}")
                return
                
            # Prepare search query from memory content/description
            query_text = mem_details.get("description", "") or mem_details.get("content", "")[:200]
            if not query_text:
                return
                
            # Search for similar memories (excluding self)
            similar_results = await self._execute_tool_call_internal(
                TOOL_SEMANTIC_SEARCH,
                {
                    "workflow_id": self.state.workflow_id,
                    "query": query_text,
                    "limit": 3,  # Limit to top matches
                    "threshold": 0.7  # Only high-quality matches
                },
                record_action=False
            )
            
            if not similar_results.get("success"):
                return
                
            # Create links to similar memories
            for similar_mem in similar_results.get("memories", []):
                similar_id = similar_mem.get("memory_id")
                similarity = similar_mem.get("similarity", 0)
                
                # Skip self or if similarity too low
                if similar_id == memory_id or similarity < 0.7:
                    continue
                    
                # Create bidirectional links
                await self._execute_tool_call_internal(
                    TOOL_CREATE_LINK,
                    {
                        "source_memory_id": memory_id,
                        "target_memory_id": similar_id,
                        "link_type": "related",
                        "strength": similarity,
                        "description": "Auto-generated link based on semantic similarity"
                    },
                    record_action=False
                )
                
                self.logger.debug(f"Auto-linked memory {memory_id[:8]} to {similar_id[:8]} (similarity: {similarity:.2f})")
                
        except Exception as e:
            self.logger.warning(f"Error in auto-linking task: {e}", exc_info=False)

    async def _check_prerequisites(self, dependency_ids: List[str]) -> Tuple[bool, str]:
        """Check if all prerequisite actions are completed."""
        if not dependency_ids:
            return True, "No dependencies"
            
        try:
            # Get details for all dependencies
            dep_details_result = await self._execute_tool_call_internal(
                TOOL_GET_ACTION_DETAILS,
                {"action_ids": dependency_ids},
                record_action=False
            )
            
            if not dep_details_result.get("success"):
                return False, f"Failed to check dependencies: {dep_details_result.get('error', 'Unknown error')}"
                
            # Check status of all dependencies
            actions = dep_details_result.get("actions", [])
            incomplete = []
            
            for action in actions:
                if action.get("status") != ActionStatus.COMPLETED.value:
                    incomplete.append(action.get("action_id", "unknown"))
                    
            if incomplete:
                return False, f"Dependencies not completed: {', '.join(incomplete[:3])}" + (f" and {len(incomplete)-3} more" if len(incomplete) > 3 else "")
                
            return True, "All dependencies completed"
            
        except Exception as e:
            self.logger.error(f"Error checking prerequisites: {e}", exc_info=False)
            return False, f"Error checking prerequisites: {str(e)}"
        
    # --- Enhanced Tool Execution Helper (#3, #7, #10, #12) ---
    async def _execute_tool_call_internal(
        self, tool_name: str, arguments: Dict[str, Any],
        record_action: bool = True, action_dependencies: Optional[List[str]] = None
        ) -> Dict[str, Any]:
        """Handles server lookup, dependency checks, execution, results, optional action recording, and triggers."""
        action_id = None
        tool_result_content = {"success": False, "error": "Execution error."}
        start_time = time.time()

        if tool_name in [TOOL_CREATE_WORKFLOW, TOOL_UPDATE_WORKFLOW_STATUS, TOOL_RECORD_ACTION_START]:
            if not tool_result_content.get("success"):
                self.logger.error(f"Critical tool operation failed: {tool_name} - {tool_result_content.get('error')}")
                # Possibly increase error count or take recovery action
                if tool_name == TOOL_CREATE_WORKFLOW:
                    self.logger.critical("Failed to create workflow - agent cannot continue without a valid workflow.")
                    
        # 1. Find Server
        target_server = self._find_tool_server(tool_name)
        if not target_server:
            err_msg = f"Tool/server unavailable: {tool_name}"; self.logger.error(err_msg)
            self.state.last_error_details = {"tool": tool_name, "error": err_msg}; return {"success": False, "error": err_msg}

        # 2. Dependency Check (#3)
        if action_dependencies:
            met, reason = await self._check_prerequisites(action_dependencies)
            if not met:
                err_msg = f"Prerequisites not met for {tool_name}: {reason}"; self.logger.warning(err_msg)
                self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "dependency_failure"}; self.state.needs_replan = True
                return {"success": False, "error": err_msg}
            self.logger.info(f"Prerequisites met for {tool_name}.")

        # 3. Record Action Start (Optional)
        if record_action and tool_name not in [TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION]:
            action_id = await self._record_action_start_internal(tool_name, arguments, target_server)

        # 4. Execute Tool
        try:
            current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
            if 'workflow_id' not in arguments and current_wf_id and tool_name not in [TOOL_CREATE_WORKFLOW, TOOL_LIST_WORKFLOWS]: arguments['workflow_id'] = current_wf_id

            call_tool_result = await self.mcp_client.execute_tool(target_server, tool_name, arguments)

            latency_ms = (time.time() - start_time) * 1000 # Calculate latency here
            # Update tool stats (#5)
            self.state.tool_usage_stats[tool_name]["latency_ms_total"] += latency_ms

            # Process result (Unchanged parsing logic from v2.1)
            if isinstance(call_tool_result, dict):
                is_error = call_tool_result.get("isError", True); content = call_tool_result.get("content")
                if is_error: tool_result_content = {"success": False, "error": str(content)}
                elif isinstance(content, dict) and "success" in content: tool_result_content = content # Assume tool returns dict with success flag
                else: tool_result_content = {"success": True, "data": content} # Wrap other results
            else: tool_result_content = {"success": False, "error": f"Unexpected result type: {type(call_tool_result)}"}

            log_msg = f"Tool {tool_name} executed. Success: {tool_result_content.get('success')} ({latency_ms:.0f}ms)"
            self.logger.info(log_msg, emoji_key="checkered_flag" if tool_result_content.get('success') else "warning")
            self.state.last_action_summary = log_msg
            if not tool_result_content.get('success'):
                 err_detail = str(tool_result_content.get('error', 'Unknown'))[:150]
                 self.state.last_action_summary += f" Error: {err_detail}"; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_detail, "result": tool_result_content}
                 self.state.tool_usage_stats[tool_name]["failure"] += 1
            else:
                 self.state.last_error_details = None # Clear last error on success
                 self.state.tool_usage_stats[tool_name]["success"] += 1
                 # --- Trigger Post-Success Actions ---
                 # Auto-Linking (#8)
                 if tool_name == TOOL_STORE_MEMORY and tool_result_content.get("memory_id"):
                      self._start_background_task(self._run_auto_linking(tool_result_content["memory_id"]))
                 # Access-Triggered Promotion (#12)
                 if tool_name in [TOOL_GET_MEMORY_BY_ID, TOOL_QUERY_MEMORIES]:
                    mem_ids_to_check = []
                    if tool_name == TOOL_GET_MEMORY_BY_ID: 
                        mem_ids_to_check = [arguments.get("memory_id")]
                    elif tool_name == TOOL_QUERY_MEMORIES:
                        # Direct access to memories field, not through "data"
                        memories = tool_result_content.get("memories", [])
                        mem_ids_to_check = [m.get("memory_id") for m in memories[:3]]
                    for mem_id in filter(None, mem_ids_to_check):
                        self._start_background_task(self._check_and_trigger_promotion(mem_id))

        # (Error Handling Unchanged from v2.1)
        except (ToolError, ToolInputError) as e: err_str = str(e); self.logger.error(f"Tool Error executing {tool_name}: {e}", exc_info=False); tool_result_content = {"success": False, "error": err_str}; self.state.last_action_summary = f"Tool {tool_name} Error: {err_str[:100]}"; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": type(e).__name__}; self.state.tool_usage_stats[tool_name]["failure"] += 1
        except Exception as e: err_str = str(e); self.logger.error(f"Unexpected Error executing {tool_name}: {e}", exc_info=True); tool_result_content = {"success": False, "error": f"Unexpected error: {err_str}"}; self.state.last_action_summary = f"Execution failed: Unexpected error."; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "Unexpected"}; self.state.tool_usage_stats[tool_name]["failure"] += 1

        # 5. Record Action Completion (Optional)
        if record_action and action_id:
             await self._record_action_completion_internal(action_id, tool_result_content)

        # 6. Handle Workflow Side Effects (#2)
        await self._handle_workflow_side_effects(tool_name, arguments, tool_result_content)

        return tool_result_content

    def _start_background_task(self, coro):
        """Creates and tracks a background task."""
        task = asyncio.create_task(coro)
        self.state.background_tasks.add(task)
        task.add_done_callback(self.state.background_tasks.discard) # Auto-remove when done

    async def _record_action_start_internal(self, primary_tool_name: str, primary_tool_args: Dict[str, Any], primary_target_server: str) -> Optional[str]:
         """Internal helper to record action start."""
         action_id = None
         start_title = f"Exec: {primary_tool_name.split(':')[-1]}"
         start_reasoning = f"Agent initiated: {primary_tool_name}"
         current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
         if not current_wf_id: self.logger.warning("Cannot record start: No active workflow"); return None

         start_tool_name = TOOL_RECORD_ACTION_START
         start_server = self._find_tool_server(start_tool_name)
         if not start_server: self.logger.error(f"Cannot record start: Tool '{start_tool_name}' unavailable."); return None

         try:
             start_args = {"workflow_id": current_wf_id, "action_type": "tool_use", "title": start_title, "reasoning": start_reasoning, "tool_name": primary_tool_name, "tool_args": primary_tool_args}
             start_result = await self.mcp_client.execute_tool(start_server, start_tool_name, start_args)
             content = start_result.get("content") if isinstance(start_result, dict) else None
             if isinstance(content, dict) and content.get("success"): action_id = content.get("action_id"); self.logger.debug(f"Action {action_id} started for {primary_tool_name}.")
             else: self.logger.warning(f"Failed recording start for {primary_tool_name}: {content.get('error', 'Unknown') if isinstance(content, dict) else 'Invalid result'}")
         except Exception as e: self.logger.error(f"Exception recording start for {primary_tool_name}: {e}", exc_info=True)
         return action_id

    async def _record_action_completion_internal(self, action_id: str, tool_result_content: Dict):
         """Internal helper to record action completion."""
         status = ActionStatus.COMPLETED.value if tool_result_content.get("success") else ActionStatus.FAILED.value
         comp_tool_name = TOOL_RECORD_ACTION_COMPLETION
         comp_server = self._find_tool_server(comp_tool_name)
         if not comp_server: self.logger.error(f"Cannot record completion: Tool '{comp_tool_name}' unavailable."); return

         try:
             await self.mcp_client.execute_tool(comp_server, comp_tool_name, {"action_id": action_id, "status": status, "tool_result": tool_result_content})
             self.logger.debug(f"Action {action_id} completion recorded ({status})")
         except Exception as e: self.logger.error(f"Error recording completion for {action_id}: {e}", exc_info=True)

    async def _handle_workflow_side_effects(self, tool_name: str, arguments: Dict, result_content: Dict):
         """Handles state changes after specific tool calls."""
         if tool_name == TOOL_CREATE_WORKFLOW and result_content.get("success"):
             new_wf_id = result_content.get("workflow_id")
             parent_id = arguments.get("parent_workflow_id")
             if new_wf_id:
                 self.state.workflow_id = new_wf_id; self.state.context_id = new_wf_id
                 if parent_id: self.state.workflow_stack.append(new_wf_id)
                 else: self.state.workflow_stack = [new_wf_id]
                 self.logger.info(f"Switched to {'sub-' if parent_id else 'new'} workflow: {new_wf_id}", emoji_key="label")
                 self.state.current_plan = [PlanStep(description=f"Start new workflow: {result_content.get('title', 'Untitled')}. Goal: {result_content.get('goal', 'Not specified')}.")] # Reset plan for new workflow
         elif tool_name == TOOL_UPDATE_WORKFLOW_STATUS and result_content.get("success"):
             status = arguments.get("status"); wf_id = arguments.get("workflow_id")
             if status in [s.value for s in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.ABANDONED]] and \
                self.state.workflow_stack and wf_id == self.state.workflow_stack[-1]:
                  finished_wf = self.state.workflow_stack.pop()
                  if self.state.workflow_stack:
                      self.state.workflow_id = self.state.workflow_stack[-1]; self.state.context_id = self.state.workflow_id
                      self.logger.info(f"Sub-workflow {finished_wf} finished. Returning to parent {self.state.workflow_id}.", emoji_key="arrow_left")
                      self.state.needs_replan = True; self.state.current_plan = [PlanStep(description=f"Returned from sub-workflow {finished_wf}. Re-assess parent goal.")]
                  else: # Completed the root workflow
                      self.state.workflow_id = None; self.state.context_id = None; self.logger.info(f"Root workflow {finished_wf} finished.")
                      self.state.goal_achieved_flag = True # Explicitly set flag

    # --- Enhanced Plan Update (#1) ---
    async def _update_plan(self, context: Dict[str, Any], last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        self.logger.info("Updating agent plan...", emoji_key="clipboard")
        llm_proposed_plan = last_decision.get("updated_plan_steps") # Expecting list of PlanStep objects (or dicts)

        if llm_proposed_plan and isinstance(llm_proposed_plan, list):
            try:
                 # Validate and convert dictionaries back to PlanStep if needed
                 validated_plan = [PlanStep(**step) if isinstance(step, dict) else step for step in llm_proposed_plan]
                 # Basic check: Ensure it's not empty and steps are valid
                 if validated_plan and all(isinstance(step, PlanStep) for step in validated_plan):
                     self.state.current_plan = validated_plan
                     self.logger.info(f"Plan updated by LLM with {len(validated_plan)} steps. First step: '{validated_plan[0].description[:50]}...'")
                     self.state.needs_replan = False
                     if self.state.last_error_details: self.state.consecutive_error_count = 0 # Assume LLM handled error in plan
                     # Update success counters based on the action that *led* to this plan update
                     if last_decision.get("decision") == "call_tool":
                         if isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success"):
                             self.state.successful_actions_since_reflection += 1; self.state.successful_actions_since_consolidation += 1
                     return # Successfully updated plan from LLM
                 else: self.logger.warning("LLM provided invalid or empty plan structure. Falling back to heuristic.")
            except (ValidationError, TypeError) as e: self.logger.warning(f"Failed to validate LLM plan: {e}. Falling back.")

        # --- Fallback to Heuristic Plan Update ---
        # (Logic unchanged from v2.1, operates on self.state.current_plan list)
        current_step = self.state.current_plan[0] if self.state.current_plan else PlanStep(description="Fallback: Re-evaluate situation.")

        if last_decision.get("decision") == "call_tool":
            tool_success = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            if tool_success:
                 current_step.status = "completed"
                 current_step.result_summary = f"Success: {str(last_tool_result_content)[:100]}..."
                 # Remove completed step, add default next step if plan is now empty
                 self.state.current_plan.pop(0)
                 if not self.state.current_plan: self.state.current_plan.append(PlanStep(description="Analyze successful tool output and plan next steps."))
                 self.state.consecutive_error_count = 0; self.state.needs_replan = False
                 self.state.successful_actions_since_reflection += 1; self.state.successful_actions_since_consolidation += 1
            else: # Tool failed
                 current_step.status = "failed"
                 error_msg = str(last_tool_result_content.get('error', 'Unknown failure'))[:150]
                 current_step.result_summary = f"Failure: {error_msg}"
                 # Keep failed step at the top, force replan
                 self.state.current_plan = [current_step] + self.state.current_plan[1:] # Keep failed step visible
                 self.state.current_plan.insert(1, PlanStep(description=f"Analyze failure of step '{current_step.description[:30]}...' and replan."))
                 self.state.consecutive_error_count += 1; self.state.needs_replan = True
                 self.state.successful_actions_since_reflection = self.reflection_threshold # Trigger reflection
        elif last_decision.get("decision") == "thought_process":
             # Treat thought as completing an implicit reasoning step
             current_step.status = "completed"
             current_step.result_summary = f"Thought Recorded: {last_decision.get('content','')[:50]}..."
             self.state.current_plan.pop(0)
             if not self.state.current_plan: self.state.current_plan.append(PlanStep(description="Decide next action based on recorded thought."))
             self.state.consecutive_error_count = 0; self.state.needs_replan = False
        elif last_decision.get("decision") == "complete":
             self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing.", status="completed")]
             self.state.consecutive_error_count = 0; self.state.needs_replan = False
        else: # Error or Unknown decision
             current_step.status = "failed"; current_step.result_summary = f"Agent/Tool Error: {self.state.last_action_summary[:100]}..."
             self.state.current_plan = [current_step] + self.state.current_plan[1:]
             self.state.current_plan.insert(1, PlanStep(description="Re-evaluate due to agent error or unclear decision."))
             self.state.consecutive_error_count += 1; self.state.needs_replan = True

        log_plan = f"Plan updated (Heuristic). Steps: {len(self.state.current_plan)}. Next: '{self.state.current_plan[0].description[:60]}...'"
        self.logger.info(log_plan)


    # --- Enhanced Periodic Tasks (#11, #12, #13) ---
    async def _run_periodic_tasks(self):
        if not self.state.workflow_id or not self.state.context_id or self._shutdown_event.is_set(): return

        # Check that required attributes exist
        required_attrs = [
            'reflection_type_sequence', 'consolidation_threshold', 
            'consolidation_memory_level', 'consolidation_max_sources',
            'optimization_interval', 'promotion_interval'
        ]
        
        for attr in required_attrs:
            if not hasattr(self, attr):
                self.logger.error(f"Missing required configuration attribute: {attr}")
                return
            
        tasks_to_run: List[Tuple[str, Dict]] = []
        trigger_reason = [] # Track why meta-cog is running

        # Adaptive Reflection Trigger (#12)
        # Check tool availability first
        reflection_tool_available = self._find_tool_server(TOOL_REFLECTION) is not None
        consolidation_tool_available = self._find_tool_server(TOOL_CONSOLIDATION) is not None
        optimize_wm_tool_available = self._find_tool_server(TOOL_OPTIMIZE_WM) is not None
        auto_focus_tool_available = self._find_tool_server(TOOL_AUTO_FOCUS) is not None

        # Add tasks only if tools are available
        if self.state.needs_replan or self.state.successful_actions_since_reflection >= self.reflection_threshold:
            if reflection_tool_available:
                reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
                tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
                trigger_reason.append(f"ReplanNeeded({self.state.needs_replan})" if self.state.needs_replan else f"SuccessCount({self.state.successful_actions_since_reflection})")
                self.state.successful_actions_since_reflection = 0
                self.state.reflection_cycle_index += 1
            else:
                self.logger.warning(f"Skipping reflection: Tool {TOOL_REFLECTION} not available")

        # Adaptive Consolidation Trigger (#12)
        # Example: Trigger if many episodic action logs exist (requires stats or query)
        # For simplicity, retain success count trigger for now. Could enhance with TOOL_COMPUTE_STATS.
        if self.state.successful_actions_since_consolidation >= self.consolidation_threshold:
             if consolidation_tool_available:
                tasks_to_run.append((TOOL_CONSOLIDATION, {"workflow_id": self.state.workflow_id, "consolidation_type": "summary", "query_filter": {"memory_level": self.consolidation_memory_level}, "max_source_memories": self.consolidation_max_sources}))
                trigger_reason.append(f"ConsolidateThreshold({self.state.successful_actions_since_consolidation})")
                self.state.successful_actions_since_consolidation = 0
             else:
                self.logger.warning(f"Skipping consolidation: Tool {TOOL_CONSOLIDATION} not available")

        # Optimization Trigger (Unchanged - loop based)
        self.state.loops_since_optimization += 1
        if self.state.loops_since_optimization >= self.optimization_interval:
             if optimize_wm_tool_available:
                tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.state.context_id})); trigger_reason.append("OptimizeInterval")
                tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.state.context_id})); trigger_reason.append("FocusUpdate")
                self.state.loops_since_optimization = 0
             else:
                 self.logger.warning(f"Skipping optimization: Tool {TOOL_OPTIMIZE_WM} not available")

        # Memory Promotion Check Trigger (Unchanged - loop based)
        self.state.loops_since_promotion_check += 1
        if self.state.loops_since_promotion_check >= self.promotion_interval:
             if auto_focus_tool_available:
                tasks_to_run.append(("CHECK_PROMOTIONS", {})); trigger_reason.append("PromotionInterval")
                self.state.loops_since_promotion_check = 0
             else:
                self.logger.warning(f"Skipping promotion check: Tool {TOOL_AUTO_FOCUS} not available")

        if tasks_to_run:
            self.logger.info(f"Running {len(tasks_to_run)} periodic tasks (Triggers: {', '.join(trigger_reason)})...", emoji_key="brain")
            for tool_name, args in tasks_to_run:
                 if self._shutdown_event.is_set(): break
                 try:
                     if tool_name == "CHECK_PROMOTIONS": await self._trigger_promotion_checks(); continue
                     self.logger.debug(f"Executing periodic task: {tool_name}")
                     result_content = await self._execute_tool_call_internal(tool_name, args, record_action=False)
                     # --- Meta-Cognition Feedback Loop (#13) ---
                     if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and result_content.get('success'):
                          content_key = "reflection_content" if tool_name == TOOL_REFLECTION else "consolidated_content"
                          feedback = result_content.get(content_key, "") or result_content.get("data", "") # Look in 'data' too
                          if feedback:
                              # Simple extraction: first line or first 150 chars
                              feedback_summary = str(feedback).split('\n')[0][:150]
                              if feedback_summary:
                                   self.state.last_meta_feedback = f"Feedback from {tool_name.split(':')[-1]}: {feedback_summary}..."
                                   self.logger.info(self.state.last_meta_feedback)
                                   self.state.needs_replan = True # Force replan after meta-cognition feedback
                 except Exception as e: self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)
                 await asyncio.sleep(0.1) # Small delay between periodic tasks

    async def _trigger_promotion_checks(self):
        """Checks promotion criteria for recently accessed, eligible memories."""
        self.logger.debug("Running periodic promotion check...")
        # Query for candidate memories (recently accessed, not max level)
        tool_name_query = TOOL_QUERY_MEMORIES
        server_query = self._find_tool_server(tool_name_query)
        if not server_query: return

        try:
            # Make two separate queries - one for each memory level
            # First query for Episodic memories
            episodic_results = await self._execute_tool_call_internal(
                tool_name_query, {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": MemoryLevel.EPISODIC.value,
                    "sort_by": "last_accessed", 
                    "sort_order": "DESC", 
                    "limit": 5, 
                    "include_content": False
                }, record_action=False
            )
            
            # Second query for Semantic memories
            semantic_results = await self._execute_tool_call_internal(
                tool_name_query, {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": MemoryLevel.SEMANTIC.value,
                    "sort_by": "last_accessed", 
                    "sort_order": "DESC", 
                    "limit": 5, 
                    "include_content": False
                }, record_action=False
            )
            
            # Combine and process results
            combined_memories = []
            
            # Add episodic memories
            if episodic_results.get("success"):
                episodic_memories = episodic_results.get("memories", [])
                self.logger.debug(f"Found {len(episodic_memories)} episodic memories for promotion check")
                combined_memories.extend(episodic_memories)
                
            # Add semantic memories
            if semantic_results.get("success"):
                semantic_memories = semantic_results.get("memories", [])
                self.logger.debug(f"Found {len(semantic_memories)} semantic memories for promotion check")
                combined_memories.extend(semantic_memories)

            # Sort combined results by last_accessed (descending)
            combined_memories.sort(key=lambda m: m.get("last_accessed_unix", 0), reverse=True)
            
            # Take the top 10 most recently accessed memories
            candidates = combined_memories[:10]
            
            # Check each candidate memory for promotion
            if candidates: 
                self.logger.debug(f"Checking {len(candidates)} memories for promotion")
                promo_tasks = [self._check_and_trigger_promotion(mem.get('memory_id')) for mem in candidates]
                await asyncio.gather(*promo_tasks, return_exceptions=True)  # Run checks concurrently
            else:
                self.logger.debug("No memories found for promotion check")
                
        except Exception as e: 
            self.logger.error(f"Error during periodic promotion check query: {e}", exc_info=False)

    # --- Run Method (Main Loop) ---
    async def run(self, goal: str, max_loops: int = 50):
        # (Implementation structure unchanged from v2.1, but uses updated helpers/state)
        if not await self.initialize(): self.logger.critical("Agent initialization failed."); return

        self.logger.info(f"Starting main loop. Goal: '{goal}' Max Loops: {max_loops}", emoji_key="arrow_forward")
        self.state.goal_achieved_flag = False # Ensure flag is reset

        while not self.state.goal_achieved_flag and self.state.current_loop < max_loops:
             if self._shutdown_event.is_set(): self.logger.info("Shutdown signal detected, exiting loop."); break
             self.state.current_loop += 1
             self.logger.info(f"--- Agent Loop {self.state.current_loop}/{max_loops} ---", emoji_key="arrows_counterclockwise")

             # Error Check
             if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                  self.logger.error(f"Max consecutive errors ({MAX_CONSECUTIVE_ERRORS}) reached. Aborting.", emoji_key="stop_sign")
                  if self.state.workflow_id: await self._update_workflow_status_internal("failed", "Agent failed due to repeated errors.")
                  break

             # 1. Gather Context
             context = await self._gather_context()
             if "error" in context and context.get("status") != "No Active Workflow":
                  self.logger.error(f"Context gathering failed: {context['error']}. Pausing."); self.state.consecutive_error_count += 1; self.state.needs_replan = True
                  await asyncio.sleep(3 + self.state.consecutive_error_count); continue

             # 2. Decide
             agent_decision = await self._call_agent_llm(goal, context)

             # 3. Act
             decision_type = agent_decision.get("decision")
             last_tool_result_content = None
             # --- Update Plan based on LLM suggestion FIRST ---
             if agent_decision.get("updated_plan_steps"):
                  self.state.current_plan = agent_decision["updated_plan_steps"]
                  self.logger.info(f"Plan updated by LLM with {len(self.state.current_plan)} steps.")
                  self.state.needs_replan = False # Assume LLM's plan is the new direction

             # --- Execute Action ---
             if decision_type == "call_tool":
                 tool_name = agent_decision.get("tool_name"); arguments = agent_decision.get("arguments", {})
                 # Extract dependencies IF LLM provides them in a structured way (e.g. in args)
                 dependencies = arguments.pop("depends_on_actions", None) if isinstance(arguments, dict) else None
                 self.logger.info(f"Agent requests tool: {tool_name}", emoji_key="wrench")
                 last_tool_result_content = await self._execute_tool_call_internal(tool_name, arguments, True, dependencies)
                 if isinstance(last_tool_result_content, dict) and not last_tool_result_content.get("success"): self.state.needs_replan = True # Set replan if tool failed

             elif decision_type == "thought_process":
                  thought_content = agent_decision.get("content", "No thought content provided.")
                  self.logger.info(f"Agent reasoning: '{thought_content[:100]}...'. Recording.", emoji_key="thought_balloon")
                  if self.state.workflow_id:
                      try:
                           thought_result = await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"workflow_id": self.state.workflow_id, "content": thought_content, "thought_type": "inference"}, True)
                           if thought_result.get("success"): self.state.last_action_summary = f"Recorded thought: {thought_content[:100]}..."; self.state.consecutive_error_count = 0
                           else: raise ToolError(f"Record thought failed: {thought_result.get('error')}")
                      except Exception as e: self.logger.error(f"Failed to record thought: {e}", exc_info=False); self.state.consecutive_error_count += 1; self.state.last_action_summary = f"Error recording thought: {str(e)[:100]}"; self.state.needs_replan = True; self.state.last_error_details = {"tool": TOOL_RECORD_THOUGHT, "error": str(e)}
                  else: self.logger.warning("No workflow to record thought."); self.state.last_action_summary = "Agent provided reasoning, but no workflow active."

             elif decision_type == "complete":
                  summary = agent_decision.get("summary", "Goal achieved."); self.logger.info(f"Agent signals completion: {summary}", emoji_key="tada")
                  self.state.goal_achieved_flag = True; self.state.needs_replan = False
                  if self.state.workflow_id: await self._update_workflow_status_internal("completed", summary)
                  break

             elif decision_type == "error":
                  error_msg = agent_decision.get("message", "Unknown agent error"); self.logger.error(f"Agent decision error: {error_msg}", emoji_key="x")
                  self.state.last_action_summary = f"Agent decision error: {error_msg[:100]}"; self.state.last_error_details = {"agent_decision_error": error_msg}
                  self.state.consecutive_error_count += 1; self.state.needs_replan = True
                  if self.state.workflow_id:
                       try: await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"workflow_id": self.state.workflow_id, "content": f"Agent Error: {error_msg}", "thought_type": "critique"}, False)
                       except Exception: pass

             else: self.logger.warning(f"Unhandled decision: {decision_type}"); self.state.last_action_summary = "Unknown agent decision."; self.state.consecutive_error_count += 1; self.state.needs_replan = True; self.state.last_error_details = {"agent_decision_error": f"Unknown type: {decision_type}"}

             # 4. Update Plan (Fallback if LLM didn't provide one)
             if not agent_decision.get("updated_plan_steps"): # Only use fallback if LLM didn't provide a plan
                 await self._update_plan(context, agent_decision, last_tool_result_content)

             # 5. Periodic Tasks
             await self._run_periodic_tasks()

             # 6. Save State Periodically
             if self.state.current_loop % 5 == 0: await self._save_agent_state()

             # 7. Loop Delay
             await asyncio.sleep(random.uniform(0.8, 1.2))

        # --- End of Loop ---
        self.logger.info("--- Agent Loop Finished ---", emoji_key="stopwatch")
        # (Final status reporting unchanged)
        await self._cleanup_background_tasks()
        await self._save_agent_state()
        if self.state.workflow_id and not self._shutdown_event.is_set(): await self._generate_final_report()

    async def _cleanup_background_tasks(self):
        if self.state.background_tasks:
            self.logger.info(f"Cleaning up {len(self.state.background_tasks)} background tasks...")
            # Cancel tasks that haven't completed
            for task in list(self.state.background_tasks): # Iterate copy
                if not task.done():
                    task.cancel()
            # Wait for tasks to finish/cancel
            await asyncio.gather(*self.state.background_tasks, return_exceptions=True)
            self.logger.info("Background tasks cleaned up.")
            self.state.background_tasks.clear()

    async def signal_shutdown(self):
        self.logger.info("Graceful shutdown signal received.", emoji_key="wave")
        self._shutdown_event.set()
        # Cancel active tasks immediately
        await self._cleanup_background_tasks()

    async def shutdown(self):
        self.logger.info("Shutting down agent loop...", emoji_key="power_button")
        self._shutdown_event.set()
        await self._cleanup_background_tasks()
        await self._save_agent_state() # Final state save
        self.logger.info("Agent loop shutdown complete.", emoji_key="checkered_flag")

    # --- Other Internal Helpers (find_tool_server, check_workflow_exists, update_status, generate_report) ---

    async def _update_workflow_status_internal(self, status: str, message: Optional[str] = None):
        """Internal helper to update workflow status via tool call."""
        if not self.state.workflow_id: return
        
        # Validate status against WorkflowStatus enum
        try:
            # Ensure status is a valid WorkflowStatus value
            status_value = status
            if not any(status == s.value for s in WorkflowStatus):
                # Try to convert string to enum value
                try:
                    status_value = WorkflowStatus(status).value
                except ValueError:
                    self.logger.warning(f"Invalid workflow status: {status}. Using 'completed' as fallback.")
                    status_value = WorkflowStatus.COMPLETED.value
        except Exception:
            status_value = WorkflowStatus.COMPLETED.value  # Fallback
        
        tool_name = TOOL_UPDATE_WORKFLOW_STATUS
        try:
            await self._execute_tool_call_internal(
                tool_name,
                {"workflow_id": self.state.workflow_id, "status": status_value, "completion_message": message},
                record_action=False  # Status updates aren't primary actions
            )
        except Exception as e:
            self.logger.error(f"Error marking workflow {self.state.workflow_id} as {status}: {e}", exc_info=False)

    async def _generate_final_report(self):
        """Generates and logs a final report using the memory tool."""
        if not self.state.workflow_id: return
        self.logger.info(f"Generating final report for workflow {self.state.workflow_id}...", emoji_key="scroll")
        tool_name = TOOL_GENERATE_REPORT # Use constant
        try:
            report_result_content = await self._execute_tool_call_internal(
                tool_name,
                {"workflow_id": self.state.workflow_id, "report_format": "markdown", "style": "professional"},
                record_action=False # Report generation is meta
            )
            if isinstance(report_result_content, dict) and report_result_content.get("success"):
                report_text = report_result_content.get("report", "Report content missing.")
                # Use safe_print for potentially long output
                self.mcp_client.safe_print("\n--- FINAL WORKFLOW REPORT ---\n" + report_text + "\n--- END REPORT ---")
            else:
                self.logger.error(f"Failed to generate final report: {report_result_content.get('error', 'Unknown error')}")
        except Exception as e:
            self.logger.error(f"Exception generating final report: {e}", exc_info=True)
                
    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Finds an active server providing the specified tool."""
        if self.mcp_client and self.mcp_client.server_manager:
            if tool_name in self.mcp_client.server_manager.tools:
                 server_name = self.mcp_client.server_manager.tools[tool_name].server_name
                 if server_name in self.mcp_client.server_manager.active_sessions:
                     return server_name
            # Fallback: Check if tool name matches a server name directly (e.g., for core server tools)
            # This part might be removed if all tools are guaranteed to be prefixed
            # elif tool_name in self.mcp_client.server_manager.active_sessions:
            #     return tool_name
        self.logger.debug(f"Tool '{tool_name}' not found on any active server.")
        return None

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """Checks if a workflow ID exists using list_workflows tool."""
        tool_name = TOOL_LIST_WORKFLOWS
        try:
            # Fetching a larger list might be inefficient just to check one ID.
            # If the memory system offers a direct get_workflow tool, use that instead.
            # Assuming list_workflows is the only way for now:
             result = await self._execute_tool_call_internal(tool_name, {"limit": 1000}, False)
             if isinstance(result, dict) and result.get("success"):
                  wf_list = result.get("workflows", []) # Assume list is in content data now
                  return any(wf.get("workflow_id") == workflow_id for wf in wf_list)
             return False
        except Exception as e: self.logger.error(f"Error checking WF {workflow_id}: {e}"); return False

# --- Main Execution Block (Unchanged from v2.1 - relies on AgentMasterLoop) ---
async def run_agent_process(mcp_server_url: str, anthropic_key: str, goal: str, max_loops: int, state_file: str, config_file: Optional[str]):
    if not MCP_CLIENT_AVAILABLE: print("❌ ERROR: MCPClient dependency not met."); sys.exit(1)
    mcp_client_instance = None; agent_loop_instance = None; exit_code = 0
    printer = print # Default printer

    try:
        printer("Instantiating MCP Client...")
        # Pass config file path to MCPClient constructor if provided
        mcp_client_instance = MCPClient(config_path=config_file)
        if hasattr(mcp_client_instance, 'safe_print'): printer = mcp_client_instance.safe_print # Use client's safe printer

        if not mcp_client_instance.config.api_key:
            if anthropic_key: printer("Using provided Anthropic API key."); mcp_client_instance.config.api_key = anthropic_key; mcp_client_instance.anthropic = AsyncAnthropic(api_key=anthropic_key)
            else: raise ValueError("Anthropic API key missing.")

        printer("Setting up MCP Client...")
        await mcp_client_instance.setup(interactive_mode=False)

        printer("Instantiating Agent Master Loop...")
        agent_loop_instance = AgentMasterLoop(mcp_client_instance=mcp_client_instance, agent_state_file=state_file)

        # --- Setup Signal Handlers ---
        loop = asyncio.get_running_loop()
        def signal_handler_wrapper(signum, frame):
            log.warning(f"Signal {signal.Signals(signum).name} received. Initiating graceful shutdown.")
            if agent_loop_instance: asyncio.create_task(agent_loop_instance.signal_shutdown())
            else: loop.stop() # Stop loop if agent not fully initialized

        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(sig, signal_handler_wrapper, sig, None)

        printer("Running Agent Loop...")
        await agent_loop_instance.run(goal=goal, max_loops=max_loops)

    except KeyboardInterrupt: printer("\n[yellow]Agent loop interrupt handled by signal handler.[/yellow]"); exit_code = 130
    except Exception as main_err: printer(f"\n❌ Critical error: {main_err}"); log.critical("Top-level execution error", exc_info=True); exit_code = 1
    finally:
        printer("Initiating final shutdown sequence...")
        if agent_loop_instance: printer("Shutting down agent loop..."); await agent_loop_instance.shutdown()
        if mcp_client_instance: printer("Closing MCP client..."); await mcp_client_instance.close()
        printer("Agent execution finished.")
        # Ensure loop stops fully if shutdown signal was handled within run_agent_process
        # This might not be strictly necessary depending on how asyncio handles shutdown signal context
        current_loop = asyncio.get_running_loop()
        if current_loop.is_running():
            # Optional: Give tasks a brief moment to finish cleanup after signal
            await asyncio.sleep(0.1)
            # current_loop.stop() # Avoid stopping the loop if run_agent_process is called from elsewhere

        if __name__ == "__main__": sys.exit(exit_code)

if __name__ == "__main__":
    # (Configuration loading unchanged)
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8013")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    AGENT_GOAL = os.environ.get("AGENT_GOAL", "Create workflow 'Advanced Analysis'. Decompose goal into planning data load and analysis steps. Record plan thoughts. Execute load (simulated ok). Check dependencies (should pass). Execute analysis (simulated ok). Consolidate results. Reflect on process. Mark complete.")
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30")) # Increased iterations for more complex goal
    AGENT_STATE_FILENAME = os.environ.get("AGENT_STATE_FILE", AGENT_STATE_FILE) # Use constant
    MCP_CLIENT_CONFIG_FILE = os.environ.get("MCP_CLIENT_CONFIG") # Allow overriding client config path

    if not ANTHROPIC_API_KEY: print("❌ ERROR: ANTHROPIC_API_KEY missing."); sys.exit(1)
    if not MCP_CLIENT_AVAILABLE: print("❌ ERROR: MCPClient dependency missing."); sys.exit(1)

    print("--- Supercharged Agent Master Loop v3.0 ---")
    print(f"Memory System URL: {MCP_SERVER_URL}")
    print(f"Agent Goal: {AGENT_GOAL}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"State File: {AGENT_STATE_FILENAME}")
    print(f"Client Config: {MCP_CLIENT_CONFIG_FILE or 'Default'}")
    print(f"Log Level: {log.level}")
    print("Anthropic API Key: Found")
    print("-----------------------------------------")

    asyncio.run(run_agent_process(MCP_SERVER_URL, ANTHROPIC_API_KEY, AGENT_GOAL, MAX_ITERATIONS, AGENT_STATE_FILENAME, MCP_CLIENT_CONFIG_FILE))
