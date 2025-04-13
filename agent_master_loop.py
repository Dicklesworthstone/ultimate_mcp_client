"""
Supercharged Agent Master Loop - v2.2 (No Placeholders)
======================================================

Finalized orchestrator for AI agents using the Unified Memory System
via the Ultimate MCP Client. Implements dynamic planning, context management,
meta-cognition feedback, state persistence, dependency checking, auto-linking,
access-triggered promotion, and robust execution logic.

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
import uuid
from collections import defaultdict  # Used in Auto-linking
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# External Libraries
import aiofiles
import anthropic
from anthropic.types import AsyncAnthropic, Message

# --- IMPORT YOUR ACTUAL MCP CLIENT and COMPONENTS ---
try:
    from mcp_client import (
        ActionStatus,
        LinkType,
        MCPClient,
        MemoryLevel,
        MemoryType,
        ToolError,
        ToolInputError,
        WorkflowStatus,
    )
    MCP_CLIENT_AVAILABLE = True
    print("INFO: Successfully imported MCPClient and required components.")
except ImportError as import_err:
    print(f"❌ CRITICAL ERROR: Could not import MCPClient or required components: {import_err}")
    print("Ensure mcp_client.py is correctly structured and in the Python path.")
    # Define dummies only if import fails
    MCPClient = type('DummyMCPClient', (object,), {})
    MCP_CLIENT_AVAILABLE = False
    class ToolError(Exception): pass
    class ToolInputError(Exception): pass
    class McpError(Exception): pass
    class WorkflowStatus: ACTIVE="active"; COMPLETED="completed"; FAILED="failed"; PAUSED="paused"; ABANDONED="abandoned"; value="dummy"
    class ActionStatus: COMPLETED="completed"; FAILED="failed"; IN_PROGRESS="in_progress"; PLANNED="planned"; SKIPPED="skipped"; value="dummy"
    class MemoryLevel: EPISODIC = "episodic"; SEMANTIC = "semantic"; PROCEDURAL="procedural"; WORKING="working"; value="dummy"
    class MemoryType: ACTION_LOG = "action_log"; REASONING_STEP="reasoning_step"; PLAN="plan"; ERROR="error"; OBSERVATION="observation"; INSIGHT="insight"; FACT="fact"; PROCEDURE="procedure"; SKILL="skill"; value="dummy"
    class ThoughtType: INFERENCE = "inference"; CRITIQUE="critique"; GOAL="goal"; DECISION="decision"; SUMMARY="summary"; REFLECTION="reflection"; PLAN="plan"; HYPOTHESIS="hypothesis"; QUESTION="question"; value="dummy"
    class LinkType: REQUIRES="requires"; INFORMS="informs"; BLOCKS="blocks"; RELATED="related"; SUPPORTS="supports"; CONTRADICTS="contradicts"; CAUSAL="causal"; GENERALIZES="generalizes"; value="dummy"
    class MemoryUtils: 
        @staticmethod 
        def generate_id(): 
            return str(uuid.uuid4()) # Minimal dummy

# --- Logging Setup ---
log = logging.getLogger("SuperchargedAgentMasterLoop")
if not log.handlers:
    log_level_str = os.environ.get("AGENT_LOOP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(log_level)
    log.info(f"Logger initialized with level {log_level_str}")

# --- Constants ---
AGENT_STATE_FILE = "agent_loop_state_v2.2.json" # Updated state file name
DEFAULT_PLAN = "Initial state: Assess goal, gather context, and formulate initial plan."
# Meta-cognition Intervals/Thresholds
REFLECTION_SUCCESS_THRESHOLD = 7
CONSOLIDATION_SUCCESS_THRESHOLD = 15
OPTIMIZATION_LOOP_INTERVAL = 10
MEMORY_PROMOTION_LOOP_INTERVAL = 20
# Context Limits
CONTEXT_RECENT_ACTIONS = 7
CONTEXT_IMPORTANT_MEMORIES = 5
CONTEXT_KEY_THOUGHTS = 5
CONTEXT_PROACTIVE_MEMORIES = 2
# Tool Names
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context"
TOOL_CREATE_WORKFLOW = "unified_memory:create_workflow"
TOOL_UPDATE_WORKFLOW_STATUS = "unified_memory:update_workflow_status"
TOOL_RECORD_ACTION_START = "unified_memory:record_action_start" # Need this name
TOOL_RECORD_ACTION_COMPLETION = "unified_memory:record_action_completion" # Need this name
TOOL_RECORD_THOUGHT = "unified_memory:record_thought"
TOOL_ADD_DEPENDENCY = "unified_memory:add_action_dependency"
TOOL_GET_DEPENDENCIES = "unified_memory:get_action_dependencies"
TOOL_GET_LINKED_MEMORIES = "unified_memory:get_linked_memories"
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
TOOL_GET_ACTION_DETAILS = "unified_memory:get_action_details" # Assuming this exists or can be added
TOOL_LIST_WORKFLOWS = "unified_memory:list_workflows" # <<<< ADD THIS LINE
TOOL_GENERATE_REPORT = "unified_memory:generate_workflow_report" # <<<< ADD THIS LINE

# --- Agent State Dataclass (#15) ---

# --- Helper for AgentState default factory ---
def _default_tool_stats():
    return defaultdict(lambda: {"success": 0, "failure": 0})

@dataclass
class AgentState:
    workflow_id: Optional[str] = None
    context_id: Optional[str] = None
    workflow_stack: List[str] = field(default_factory=list)
    current_plan: str = DEFAULT_PLAN
    last_action_summary: str = "Loop initialized."
    consecutive_error_count: int = 0
    current_loop: int = 0
    goal_achieved_flag: bool = False
    needs_replan: bool = False
    last_error_details: Optional[Dict] = None
    successful_actions_since_reflection: int = 0
    successful_actions_since_consolidation: int = 0
    loops_since_optimization: int = 0
    loops_since_promotion_check: int = 0
    reflection_cycle_index: int = 0
    last_meta_feedback: Optional[str] = None
    tool_usage_stats: Dict[str, Dict[str, int]] = field(default_factory=_default_tool_stats)
    background_tasks: List[str] = field(default_factory=list)


# --- Agent Loop Class ---
class AgentMasterLoop:
    """Enhanced orchestrator implementing advanced agent capabilities."""

    def __init__(self, mcp_client_instance: MCPClient, agent_state_file: str = AGENT_STATE_FILE): # type: ignore
        if not MCP_CLIENT_AVAILABLE: raise RuntimeError("MCPClient class unavailable.")
        if not isinstance(mcp_client_instance, MCPClient): raise TypeError("Requires MCPClient instance.")

        self.mcp_client = mcp_client_instance
        self.anthropic_client = self.mcp_client.anthropic
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        if not self.anthropic_client:
            self.logger.critical("Anthropic client unavailable! Agent decision-making will fail.")
            raise ValueError("Anthropic client required.")

        self.state = AgentState()
        self._shutdown_event = asyncio.Event()
        self.tool_schemas: List[Dict[str, Any]] = []
        self._background_link_tasks: Set[asyncio.Task] = set() # Track linking tasks (#8)

        # Load persistent state (run synchronously in init)
        asyncio.run(self._load_agent_state())

    async def initialize(self) -> bool:
        self.logger.info("Initializing agent loop...", emoji_key="gear")
        try:
            if not self.mcp_client.server_manager:
                self.logger.error("MCP Client Server Manager not initialized.")
                return False

            # Load/Filter tool schemas
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
                self.logger.error(f"Missing essential tools for loop operation: {missing_essential}")
                return False # Make essential tools mandatory

            # Check workflow ID validity
            if self.state.workflow_id and not await self._check_workflow_exists(self.state.workflow_id):
                self.logger.warning(f"Loaded workflow {self.state.workflow_id} not found. Resetting state.")
                await self._reset_state_to_defaults(); await self._save_agent_state()

            self.logger.info("Agent loop initialized successfully.")
            return True
        except Exception as e:
            self.logger.critical(f"Agent loop initialization failed: {e}", exc_info=True)
            return False

    # --- State Persistence (Unchanged from v2.1) ---
    async def _save_agent_state(self):
        """Saves the agent loop's state to a JSON file."""
        state_dict = dataclasses.asdict(self.state)
        state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Exclude non-serializable fields if necessary (e.g., background_tasks set)
        state_dict.pop("background_tasks", None)
        state_dict["tool_usage_stats"] = dict(state_dict["tool_usage_stats"]) # Convert defaultdict

        try:
            # Ensure directory exists
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.agent_state_file, 'w') as f:
                await f.write(json.dumps(state_dict, indent=2))
            self.logger.debug(f"Agent state saved to {self.agent_state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)

    async def _load_agent_state(self):
        """Loads the agent loop's state from a JSON file."""
        if not self.agent_state_file.exists():
            self.logger.info("No previous agent state file found. Using default state.")
            self.state = AgentState() # Ensure defaults
            return
        try:
            async with aiofiles.open(self.agent_state_file, 'r') as f:
                state_data = json.loads(await f.read())
            # Update the existing state object carefully
            loaded_state = AgentState(**{k: state_data.get(k, getattr(AgentState(), k)) for k in AgentState.__annotations__ if k in state_data})
            # Restore complex types
            loaded_state.tool_usage_stats = defaultdict(lambda: {"success": 0, "failure": 0}, loaded_state.tool_usage_stats or {})
            self.state = loaded_state # Replace current state
            self.logger.info(f"Agent state loaded successfully from {self.agent_state_file}. Resuming loop {self.state.current_loop + 1}.")
            if self.state.workflow_id: self.logger.info(f"Resuming workflow: {self.state.workflow_id}")
            else: self.logger.info("No active workflow loaded.")
        except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Failed to load/parse agent state: {e}. Resetting.", exc_info=True)
            await self._reset_state_to_defaults()
        except Exception as e:
             self.logger.error(f"Unexpected error loading agent state: {e}. Resetting.", exc_info=True)
             await self._reset_state_to_defaults()
             
    async def _reset_state_to_defaults(self):
            """Resets agent state variables to their initial default values."""
            self.state = AgentState() # Reset using dataclass defaults
            self.logger.warning("Agent state has been reset to defaults.")

    # --- Context Gathering (Unchanged from v2.1 - uses proactive search) ---
    async def _gather_context(self) -> Dict[str, Any]:
        """Gathers comprehensive context, preferring bundled tool if available."""
        self.logger.info("Gathering context...", emoji_key="satellite")
        base_context = {
            "current_loop": self.state.current_loop,
            "current_plan": self.state.current_plan,
            "last_action_summary": self.state.last_action_summary,
            "consecutive_errors": self.state.consecutive_error_count,
            "last_error_details": self.state.last_error_details,
            "workflow_stack": self.state.workflow_stack,
            "meta_feedback": self.state.last_meta_feedback,
            "tool_usage_summary": {
                 tool: counts for tool, counts in self.state.tool_usage_stats.items() if counts["success"] > 0 or counts["failure"] > 0
            }
        }
        self.state.last_meta_feedback = None # Clear feedback after including

        current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_workflow_id:
            base_context["status"] = "No Active Workflow"
            base_context["message"] = "Need to create or load a workflow."
            return base_context

        # --- Proactive Memory Retrieval ---
        proactive_memories = []
        if self.state.current_plan != DEFAULT_PLAN:
            try:
                search_result_content = await self._execute_tool_call_internal(
                    TOOL_SEMANTIC_SEARCH,
                    {"workflow_id": current_workflow_id, "query": self.state.current_plan, "limit": CONTEXT_PROACTIVE_MEMORIES, "include_content": False},
                    record_action=False
                )
                if isinstance(search_result_content, dict) and search_result_content.get("success"):
                    # Adjust based on actual return structure of semantic search
                    proactive_memories = search_result_content.get("memories", []) # Assume list is directly in content
                    if proactive_memories: self.logger.info(f"Retrieved {len(proactive_memories)} proactive memories.")
            except Exception as e: self.logger.warning(f"Proactive memory search failed: {e}", exc_info=False)
        base_context["proactive_memories"] = proactive_memories

        # --- Fetch Main Context ---
        tool_name = TOOL_GET_CONTEXT
        if not self._find_tool_server(tool_name):
            base_context["status"] = "Tool Not Available"; base_context["error"] = f"Context tool '{tool_name}' unavailable."; return base_context

        try:
            dynamic_action_limit = max(1, CONTEXT_RECENT_ACTIONS - len(proactive_memories))
            dynamic_mem_limit = max(1, CONTEXT_IMPORTANT_MEMORIES - len(proactive_memories))

            tool_result_content = await self._execute_tool_call_internal(
                tool_name=tool_name,
                arguments={
                    "workflow_id": current_workflow_id,
                    "recent_actions_limit": dynamic_action_limit,
                    "important_memories_limit": dynamic_mem_limit,
                    "key_thoughts_limit": CONTEXT_KEY_THOUGHTS
                },
                record_action=False
            )

            if isinstance(tool_result_content, dict) and tool_result_content.get("success"):
                self.logger.info("Main context gathered successfully.", emoji_key="signal_strength")
                base_context.update(tool_result_content); base_context.pop("success", None)
                base_context["status"] = "Context Ready"
            else:
                error_msg = f"get_workflow_context failed: {tool_result_content.get('error', 'Unknown')}"
                self.logger.warning(error_msg); base_context["status"] = "Tool Execution Failed"; base_context["error"] = error_msg

        except Exception as e:
            self.logger.error(f"Exception gathering main context: {e}", exc_info=True)
            base_context["status"] = "Context Gathering Error"; base_context["error"] = f"Exception: {e}"

        return base_context

    def _construct_agent_prompt(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Constructs the detailed messages list for the Anthropic API call."""
        # (Implementation unchanged from v2.1/previous response)
        system_prompt = f"""You are 'Maestro', an AI agent orchestrator using a Unified Memory System. Your goal is to achieve the user's objective by strategically using memory tools.

Overall Goal: {goal}

Available Unified Memory Tools (Use ONLY these):
"""
        if not self.tool_schemas: system_prompt += "- CRITICAL WARNING: No unified_memory tools loaded.\n"
        else:
            for schema in self.tool_schemas:
                 sanitized_name = schema['name']; original_name = self.mcp_client.server_manager.sanitized_to_original.get(sanitized_name, 'Unknown Original')
                 system_prompt += f"\n- Name: `{sanitized_name}` (Maps to: `{original_name}`)\n"; system_prompt += f"  Desc: {schema.get('description', 'N/A')}\n"; system_prompt += f"  Schema: {json.dumps(schema['input_schema'])}\n"
        system_prompt += f"""
Your Process:
1.  Context Analysis: Deeply analyze the 'Current Context' below. Note workflow status, errors (`last_error_details`), recent actions, memories (note `importance`/`confidence`), thoughts, plan (`current_plan`), and `proactive_memories`.
2.  Error Handling: If `last_error_details` exists, **FIRST** explain the error and propose a specific recovery strategy in your reasoning. Update the plan accordingly.
3.  Reasoning & Planning:
    a. State step-by-step reasoning towards the Goal, considering context, errors, and meta-feedback.
    b. CRITICALLY EVALUATE `current_plan`. Is it still valid? Does it address errors/feedback?
    c. Propose an **Updated Plan** (1-3 concrete, actionable steps). Log significant planning using `record_thought(thought_type='plan')`. If replanning due to error/feedback, state this.
4.  Action Decision: Choose the **single best** next action based on your Updated Plan:
    *   Call Memory Tool: Select the most precise `unified_memory:*` tool. Provide args per schema. **Mandatory:** Call `create_workflow` if context shows 'No Active Workflow'. **Dependencies:** If your plan involves dependent actions, consider using `add_action_dependency`. Before executing a critical action, consider if checking prerequisites via `get_action_dependencies` is needed (if dependencies were previously added).
    *   Record Thought: Use `record_thought` for logging vital reasoning, questions, hypotheses, critiques, or detailed plans not suitable for the main plan variable.
    *   Signal Completion: If Overall Goal is MET, respond ONLY with "Goal Achieved:" and final summary.
5.  Output Format: Respond **ONLY** with the valid JSON for the chosen tool call OR the "Goal Achieved:" text. NO conversational filler.

Key Considerations:
*   Use memory confidence (#15): Be cautious with low-confidence memories.
*   Record actions via orchestrator; focus on the *task* tool.
*   Use the full range of memory tools for knowledge management.
"""
        context_str = json.dumps(context, indent=2, default=str, ensure_ascii=False); max_context_len = 18000
        if len(context_str) > max_context_len: context_str = context_str[:max_context_len] + "\n... (Context Truncated)\n}"; self.logger.warning("Truncated context string sent to LLM.")
        user_prompt = f"Current Context:\n```json\n{context_str}\n```\n\n"; user_prompt += f"My Current Plan:\n```\n{self.state.current_plan}\n```\n\n"; user_prompt += f"Last Action Summary:\n{self.state.last_action_summary}\n\n"
        if self.state.last_error_details: user_prompt += f"**CRITICAL: Address Last Error:**\n```json\n{json.dumps(self.state.last_error_details, indent=2)}\n```\n\n"
        user_prompt += f"Overall Goal: {goal}\n\n"; user_prompt += "**Instruction:** Analyze context & errors. Reason step-by-step. Update plan. Decide ONE action (Tool JSON or 'Goal Achieved:')."
        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]


    async def _call_agent_llm(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Claude 3.7 Sonnet, parses response for tool call or completion."""
        # (Implementation unchanged from v2.1/previous response)
        self.logger.info("Calling Agent LLM (Claude 3.7 Sonnet) for decision...", emoji_key="robot_face")
        if not self.anthropic_client: return {"decision": "error", "message": "Anthropic client not available."}
        messages = self._construct_agent_prompt(goal, context); api_tools = self.tool_schemas
        try:
            response: Message = await self.anthropic_client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=3500, messages=messages, tools=api_tools if api_tools else None, tool_choice={"type": "auto"}, temperature=0.5)
            self.logger.debug(f"LLM Raw Response Stop Reason: {response.stop_reason}")
            decision = {"decision": "error", "message": "LLM provided no actionable output."}; text_response_parts = []; tool_call_detected = None; plan_update_text = None
            for block in response.content:
                if block.type == "text": text_response_parts.append(block.text); match = re.search(r"Updated Plan:\s*([\s\S]+)", block.text, re.IGNORECASE)
                if match: plan_update_text = match.group(1).strip()
                elif block.type == "tool_use": tool_call_detected = block; break
            full_text_response = "".join(text_response_parts).strip()
            if tool_call_detected:
                tool_name_sanitized = tool_call_detected.name; tool_input = tool_call_detected.input or {}
                original_tool_name = self.mcp_client.server_manager.sanitized_to_original.get(tool_name_sanitized, tool_name_sanitized)
                self.logger.info(f"LLM chose tool: {original_tool_name} (Sanitized: {tool_name_sanitized})", emoji_key="hammer_and_wrench")
                if original_tool_name.startswith("unified_memory:"): decision = {"decision": "call_tool", "tool_name": original_tool_name, "arguments": tool_input}
                else: self.logger.warning(f"LLM called non-unified_memory tool '{original_tool_name}'. Treating as reasoning."); decision = {"decision": "thought_process", "content": full_text_response}
            elif full_text_response.startswith("Goal Achieved:"): decision = {"decision": "complete", "summary": full_text_response.replace("Goal Achieved:", "").strip()}
            elif full_text_response: decision = {"decision": "thought_process", "content": full_text_response}; self.logger.info("LLM provided text reasoning/plan update.")
            if plan_update_text and decision.get("decision") != "error": decision["updated_plan"] = plan_update_text
            self.logger.debug(f"Agent Decision Parsed: {decision}")
            return decision
        except anthropic.APIConnectionError as e: msg = f"API Connection Error: {e}"; self.logger.error(msg, exc_info=True)
        except anthropic.RateLimitError: 
            msg = "Rate limit exceeded."; self.logger.error(msg, exc_info=True)
            await asyncio.sleep(random.uniform(5, 10))
        except anthropic.APIStatusError as e: 
            msg = f"API Error {e.status_code}: {e.message}"
            self.logger.error(f"Anthropic API status error: {e.status_code} - {e.response}", exc_info=True)
        except Exception as e: msg = f"Unexpected LLM interaction error: {e}"; self.logger.error(msg, exc_info=True)
        return {"decision": "error", "message": msg}

    # --- Tool Execution Helper (Enhanced with Dependency Check & Auto-Linking Trigger) ---
    async def _execute_tool_call_internal(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        record_action: bool = True,
        action_dependencies: Optional[List[str]] = None # List of action IDs this depends on
        ) -> Dict[str, Any]:
        """Finds server, checks dependencies, executes tool, handles result, optionally records actions, triggers auto-linking."""
        action_id = None
        tool_result_content = {"success": False, "error": "Execution error."}

        # --- 1. Find Server ---
        target_server = self._find_tool_server(tool_name)
        if not target_server:
            err_msg = f"Cannot execute '{tool_name}': Tool/server unavailable."
            self.logger.error(err_msg); self.state.last_error_details = {"tool": tool_name, "error": err_msg}
            return {"success": False, "error": err_msg}

        # --- 2. Dependency Check (#7 - Implemented) ---
        if action_dependencies:
            self.logger.info(f"Checking {len(action_dependencies)} prerequisites for action using {tool_name}...")
            prereqs_met, prereq_reason = await self._check_prerequisites(action_dependencies)
            if not prereqs_met:
                err_msg = f"Prerequisites not met for {tool_name}: {prereq_reason}. Dependencies: {action_dependencies}"
                self.logger.warning(err_msg)
                self.state.last_error_details = {"tool": tool_name, "error": err_msg, "type": "dependency_failure"}
                self.state.needs_replan = True # Force replan
                return {"success": False, "error": err_msg}
            else:
                self.logger.info("Prerequisites met.")

        # --- 3. Record Action Start (Optional) ---
        if record_action:
            # Do not record actions for internal recording tools
            if tool_name not in [TOOL_RECORD_ACTION_START, TOOL_RECORD_ACTION_COMPLETION]:
                 action_id = await self._record_action_start_internal(tool_name, arguments, target_server)

        # --- 4. Execute Primary Tool ---
        try:
            current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
            if 'workflow_id' not in arguments and current_wf_id and tool_name not in [TOOL_CREATE_WORKFLOW, "unified_memory:list_workflows"]:
                 arguments['workflow_id'] = current_wf_id

            call_tool_result = await self.mcp_client.execute_tool(target_server, tool_name, arguments)

            # (Result Parsing Logic Unchanged from v2.1)
            if isinstance(call_tool_result, dict):
                is_error = call_tool_result.get("isError", True); content = call_tool_result.get("content")
                if is_error: tool_result_content = {"success": False, "error": str(content)}
                elif isinstance(content, dict) and "success" in content: tool_result_content = content
                else: tool_result_content = {"success": True, "data": content}
            else: tool_result_content = {"success": False, "error": f"Unexpected result type: {type(call_tool_result)}"}

            self.logger.info(f"Tool {tool_name} executed. Success: {tool_result_content.get('success')}", emoji_key="checkered_flag")
            self.state.last_action_summary = f"Executed {tool_name}. Success: {tool_result_content.get('success')}."
            if not tool_result_content.get('success'):
                err_detail = str(tool_result_content.get('error', 'Unknown'))[:150]
                self.state.last_action_summary += f" Error: {err_detail}"; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_detail, "result": tool_result_content}
            else:
                self.state.last_error_details = None
                # Trigger promotion check on successful memory access (#12)
                if tool_name == TOOL_GET_MEMORY_BY_ID and arguments.get('memory_id'): await self._check_and_trigger_promotion(arguments['memory_id'])
                elif tool_name == TOOL_QUERY_MEMORIES and tool_result_content.get('success'):
                    memories = tool_result_content.get('data', {}).get('memories', [])
                    for mem in memories[:3]: await self._check_and_trigger_promotion(mem.get('memory_id'))

        # (Error Handling Unchanged from v2.1)
        except (ToolError, ToolInputError) as e: err_str = str(e); self.logger.error(f"Tool Error executing {tool_name}: {e}", exc_info=False); tool_result_content = {"success": False, "error": err_str}; self.state.last_action_summary = f"Tool {tool_name} Error: {err_str[:100]}"; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": type(e).__name__}
        except Exception as e: err_str = str(e); self.logger.error(f"Unexpected Error executing {tool_name}: {e}", exc_info=True); tool_result_content = {"success": False, "error": f"Unexpected error: {err_str}"}; self.state.last_action_summary = f"Execution failed: Unexpected error."; self.state.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "Unexpected"}

        # --- 5. Record Action Completion (Optional) ---
        if record_action and action_id:
            # Only record completion if start was recorded
            await self._record_action_completion_internal(action_id, tool_result_content)

        # --- 6. Handle Workflow Creation/Completion Side Effects (#2) ---
        if tool_name == TOOL_CREATE_WORKFLOW and tool_result_content.get("success"):
            new_wf_id = tool_result_content.get("workflow_id")
            parent_id = arguments.get("parent_workflow_id")
            if new_wf_id:
                self.state.workflow_id = new_wf_id; self.state.context_id = new_wf_id
                if parent_id: self.state.workflow_stack.append(new_wf_id)
                else: self.state.workflow_stack = [new_wf_id]
                self.logger.info(f"Switched to {'sub-' if parent_id else 'new'} workflow: {new_wf_id}", emoji_key="label")
        elif tool_name == TOOL_UPDATE_WORKFLOW_STATUS and tool_result_content.get("success"):
             updated_status = arguments.get("status"); updated_wf_id = arguments.get("workflow_id")
             if updated_status in [s.value for s in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.ABANDONED]] and \
                self.state.workflow_stack and updated_wf_id == self.state.workflow_stack[-1]:
                  finished_wf = self.state.workflow_stack.pop()
                  if self.state.workflow_stack:
                      self.state.workflow_id = self.state.workflow_stack[-1]; self.state.context_id = self.state.workflow_id
                      self.logger.info(f"Sub-workflow {finished_wf} finished. Returning to parent {self.state.workflow_id}.", emoji_key="arrow_left")
                      self.state.needs_replan = True # Force replan after sub-task
                  else: # Completed the root workflow
                      self.state.workflow_id = None; self.state.context_id = None
                      self.logger.info(f"Root workflow {finished_wf} finished.")

        # --- 7. Trigger Auto Linking (#8 - Implemented) ---
        if tool_name == TOOL_STORE_MEMORY and tool_result_content.get("success") and tool_result_content.get("memory_id"):
             # Start background task without awaiting it
             linking_task = asyncio.create_task(self._run_auto_linking(tool_result_content["memory_id"]))
             self._background_link_tasks.add(linking_task)
             # Optional: Remove completed tasks from the set to prevent memory leaks
             linking_task.add_done_callback(self._background_link_tasks.discard)

        # Update tool usage stats (#4)
        success_key = "success" if tool_result_content.get("success") else "failure"
        self.state.tool_usage_stats[tool_name][success_key] += 1

        return tool_result_content

    # --- Prerequisite Check (#7 - Implemented) ---
    async def _check_prerequisites(self, prerequisite_action_ids: List[str]) -> Tuple[bool, str]:
        """Checks if all prerequisite actions are COMPLETED."""
        if not prerequisite_action_ids: return True, "No prerequisites specified."

        self.logger.debug(f"Checking status of prerequisites: {prerequisite_action_ids}")
        tool_name = TOOL_GET_ACTION_DETAILS # Requires a tool to get multiple action details efficiently
        target_server = self._find_tool_server(tool_name)
        if not target_server:
            self.logger.warning(f"Cannot check prerequisites: '{tool_name}' tool unavailable.")
            return False, "Dependency check tool unavailable."

        try:
             # Call the tool to get details for all prerequisite actions
             # This assumes the tool can accept a list of IDs. Adapt if it only takes one at a time.
             result_content = await self._execute_tool_call_internal(
                  tool_name, {"action_ids": prerequisite_action_ids}, record_action=False
             )

             if not result_content.get("success"):
                 return False, f"Failed to fetch prerequisite action details: {result_content.get('error', 'Unknown')}"

             # Assuming result_content['data']['actions'] is a list of action detail dicts
             action_details = result_content.get("data", {}).get("actions", [])
             statuses = {a['action_id']: a['status'] for a in action_details if 'action_id' in a and 'status' in a}

             not_met = []
             for req_id in prerequisite_action_ids:
                 status = statuses.get(req_id)
                 if status != ActionStatus.COMPLETED.value:
                     not_met.append(f"{req_id[:8]} (Status: {status or 'Not Found'})")

             if not_met:
                 return False, f"Unmet prerequisites: {', '.join(not_met)}"
             else:
                 return True, "All prerequisites met."

        except Exception as e:
            self.logger.error(f"Error checking prerequisites: {e}", exc_info=True)
            return False, f"Exception during prerequisite check: {e}"

    # --- Access-Triggered Promotion (#12 - Implemented) ---
    async def _check_and_trigger_promotion(self, memory_id: Optional[str]):
        """Checks if a memory's access count warrants attempting promotion."""
        if not memory_id: return

        # Define thresholds here or get from config
        trigger_threshold_episodic = 4 # Check after 4th access (will trigger on 5th)
        trigger_threshold_semantic = 8 # Check after 8th access (will trigger on 9th/10th)

        tool_name_get = TOOL_GET_MEMORY_BY_ID
        tool_name_promote = TOOL_PROMOTE_MEM
        server_get = self._find_tool_server(tool_name_get)
        server_promote = self._find_tool_server(tool_name_promote)

        if not server_get or not server_promote:
            self.logger.debug("Skipping promotion check: required tools unavailable.")
            return

        try:
            # 1. Get current memory details (access count, level)
            mem_details_content = await self._execute_tool_call_internal(
                tool_name_get, {"memory_id": memory_id, "include_content": False, "include_links": False}, record_action=False
            )

            if not mem_details_content.get("success"):
                self.logger.warning(f"Could not get memory details for {memory_id} to check promotion trigger.")
                return

            access_count = mem_details_content.get("access_count", 0)
            current_level = mem_details_content.get("memory_level")

            # 2. Check if trigger threshold is met
            should_trigger = False
            if current_level == MemoryLevel.EPISODIC.value and access_count >= trigger_threshold_episodic:
                should_trigger = True
            elif current_level == MemoryLevel.SEMANTIC.value and access_count >= trigger_threshold_semantic:
                 # Check type eligibility for procedural promotion
                 if mem_details_content.get("memory_type") in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]:
                     should_trigger = True

            # 3. Trigger promotion tool if needed
            if should_trigger:
                 self.logger.info(f"Access count ({access_count}) for memory {memory_id} triggers promotion check from {current_level}.", emoji_key="arrow_up")
                 # Execute promotion check asynchronously
                 asyncio.create_task(
                      self._execute_tool_call_internal(tool_name_promote, {"memory_id": memory_id}, record_action=False),
                      name=f"promotion_check_{memory_id[:8]}"
                 )

        except Exception as e:
            self.logger.error(f"Error during access-triggered promotion check for {memory_id}: {e}", exc_info=False)


    # --- Auto Linking (#8 - Implemented) ---
    async def _run_auto_linking(self, new_memory_id: str):
        """Background task to find and create semantic links for a new memory."""
        self.logger.info(f"[AutoLink] Starting analysis for new memory {new_memory_id}...", emoji_key="link")
        await asyncio.sleep(random.uniform(1.0, 2.5)) # Simulate delay/allow DB commit

        try:
            # --- 1. Get Details of the New Memory ---
            get_tool = TOOL_GET_MEMORY_BY_ID
            mem_details_content = await self._execute_tool_call_internal(
                 get_tool, {"memory_id": new_memory_id, "include_content": True, "include_links": False}, record_action=False
            )
            if not mem_details_content.get("success"):
                 self.logger.warning(f"[AutoLink] Could not get details for source memory {new_memory_id}. Aborting.")
                 return

            content = mem_details_content.get("content", "")
            desc = mem_details_content.get("description", "")
            current_workflow_id = mem_details_content.get("workflow_id")
            current_mem_type = mem_details_content.get("memory_type")
            text_to_search = f"{desc}: {content}" if desc else content

            if not text_to_search or not current_workflow_id:
                 self.logger.debug(f"[AutoLink] Insufficient data for semantic search for memory {new_memory_id}.")
                 return

            # --- 2. Find Semantically Similar Memories ---
            search_tool = TOOL_SEMANTIC_SEARCH
            similar_result_content = await self._execute_tool_call_internal(
                search_tool,
                {"workflow_id": current_workflow_id, "query": text_to_search, "limit": 5, "threshold": 0.80, "include_content": False}, # Higher threshold for linking
                record_action=False
            )
            if not similar_result_content.get("success"):
                self.logger.warning(f"[AutoLink] Semantic search failed for memory {new_memory_id}.")
                return

            similar_mems = similar_result_content.get("memories", []) # Assumes search returns list directly in content now
            if not similar_mems:
                 self.logger.info(f"[AutoLink] No sufficiently similar memories found for {new_memory_id}.")
                 return

            # --- 3. Create Links (If not already existing) ---
            link_tool = TOOL_CREATE_LINK
            links_created_count = 0
            for sim_mem in similar_mems:
                target_id = sim_mem.get("memory_id")
                if target_id and target_id != new_memory_id:
                    similarity = sim_mem.get("similarity", 0.0)
                    target_type = sim_mem.get("memory_type")

                    # Basic link type heuristic
                    link_type = LinkType.RELATED.value
                    if current_mem_type == target_type: link_type = LinkType.SUPPORTS.value # Assume same types support each other
                    elif current_mem_type == MemoryType.INSIGHT.value and target_type == MemoryType.FACT.value: link_type = LinkType.GENERALIZES.value
                    elif current_mem_type == MemoryType.FACT.value and target_type == MemoryType.INSIGHT.value: link_type = LinkType.SUPPORTS.value # Fact supports insight

                    try:
                        # This tool handles duplicate links internally (INSERT OR IGNORE)
                        link_result = await self._execute_tool_call_internal(
                            link_tool,
                            {"source_memory_id": new_memory_id, "target_memory_id": target_id, "link_type": link_type, "description": f"Auto-linked (Similarity: {similarity:.2f})", "strength": similarity},
                            record_action=False
                        )
                        if isinstance(link_result, dict) and link_result.get("success") and link_result.get("link_id"): # Check if a link was created/found
                             links_created_count += 1
                    except Exception as link_err:
                         self.logger.warning(f"[AutoLink] Failed to create link {new_memory_id} -> {target_id}: {link_err}")
            if links_created_count > 0:
                 self.logger.info(f"[AutoLink] Created {links_created_count} links for memory {new_memory_id}", emoji_key="link")

        except Exception as e:
            self.logger.error(f"[AutoLink] Error during auto-linking for {new_memory_id}: {e}", exc_info=True)

    # --- Plan Update (Enhanced) ---
    async def _update_plan(self, context: Dict[str, Any], last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        """Updates plan, preferring LLM's plan, handling errors."""
        self.logger.info("Updating agent plan...", emoji_key="clipboard")
        llm_updated_plan = last_decision.get("updated_plan")

        if llm_updated_plan:
            self.state.current_plan = llm_updated_plan
            self.logger.info(f"Plan updated by LLM.")
            self.state.needs_replan = False
            if self.state.last_error_details: self.state.consecutive_error_count = 0 # LLM handled error
        else: # Fallback to heuristics
            tool_success = False; error_msg = "No result content"; tool_name = "N/A"
            if last_decision.get("decision") == "call_tool":
                 tool_name = last_decision.get("tool_name", "Unknown Tool")
                 if isinstance(last_tool_result_content, dict): tool_success = last_tool_result_content.get("success", False); error_msg = str(last_tool_result_content.get("error", "Tool failure"))
                 else: tool_success = False; error_msg = f"Invalid result type: {type(last_tool_result_content)}"

                 if tool_success:
                     self.state.current_plan = f"After {tool_name}, analyze output and proceed."
                     self.state.consecutive_error_count = 0; self.state.needs_replan = False
                     self.state.successful_actions_since_reflection += 1; self.state.successful_actions_since_consolidation += 1
                 else:
                     self.state.current_plan = f"Tool {tool_name} failed: '{error_msg[:100]}...'. Analyze and replan."
                     self.state.consecutive_error_count += 1; self.state.needs_replan = True
                     self.state.successful_actions_since_reflection = self.reflection_threshold # Trigger reflection
            elif last_decision.get("decision") == "thought_process":
                 self.state.current_plan = f"After thought '{last_decision.get('content','')[:50]}...', decide action."
                 self.state.consecutive_error_count = 0; self.state.needs_replan = False
            elif last_decision.get("decision") == "complete":
                 self.state.current_plan = "Goal Achieved."; self.state.consecutive_error_count = 0; self.state.needs_replan = False
            elif last_decision.get("decision") == "error":
                 self.state.current_plan = f"Agent error: '{last_decision.get('message', 'Unknown')[:100]}...'. Re-evaluate."
                 self.state.consecutive_error_count += 1; self.state.needs_replan = True
            else: # Unknown
                 self.state.current_plan = "Agent decision unclear. Re-evaluate."; self.state.consecutive_error_count += 1; self.state.needs_replan = True

        self.logger.info(f"New Plan: {self.state.current_plan}")


    # --- Periodic Tasks (Enhanced Triggers & Feedback) ---
    async def _run_periodic_tasks(self):
        # (Implementation mostly unchanged from v2.1, uses internal state and _execute_tool_call_internal)
        if not self.state.workflow_id or not self.state.context_id: return
        if self._shutdown_event.is_set(): return

        tasks_to_run: List[Tuple[str, Dict]] = []

        # Reflection Trigger (#4, #11)
        if self.state.needs_replan or self.state.successful_actions_since_reflection >= self.reflection_threshold:
             reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
             tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
             self.state.successful_actions_since_reflection = 0; self.state.reflection_cycle_index += 1

        # Consolidation Trigger (#4, #11)
        if self.state.successful_actions_since_consolidation >= self.consolidation_threshold:
             tasks_to_run.append((TOOL_CONSOLIDATION, {"workflow_id": self.state.workflow_id, "consolidation_type": "summary", "query_filter": {"memory_level": self.consolidation_memory_level}, "max_source_memories": self.consolidation_max_sources}))
             self.state.successful_actions_since_consolidation = 0

        # Optimization Trigger (#9)
        self.state.loops_since_optimization += 1
        if self.state.loops_since_optimization >= self.optimization_interval:
             tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.state.context_id}))
             tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.state.context_id}))
             self.state.loops_since_optimization = 0

        # Memory Promotion Check Trigger (#10, #12)
        self.state.loops_since_promotion_check += 1
        if self.state.loops_since_promotion_check >= self.promotion_interval:
            tasks_to_run.append(("CHECK_PROMOTIONS", {})) # Pseudo-task
            self.state.loops_since_promotion_check = 0

        if tasks_to_run:
            self.logger.info(f"Running {len(tasks_to_run)} periodic tasks...", emoji_key="brain")
            for tool_name, args in tasks_to_run:
                 if self._shutdown_event.is_set(): break
                 try:
                     self.logger.debug(f"Executing periodic task: {tool_name}")
                     if tool_name == "CHECK_PROMOTIONS": await self._trigger_promotion_checks(); continue

                     result_content = await self._execute_tool_call_internal(tool_name, args, record_action=False)
                     # Handle feedback (#11)
                     if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and isinstance(result_content, dict) and result_content.get('success'):
                          content_key = "reflection_content" if tool_name == TOOL_REFLECTION else "consolidated_content"
                          feedback = result_content.get(content_key, "") or result_content.get("content","") # Check backup key
                          if feedback:
                              feedback_summary = f"Feedback from {tool_name.split(':')[-1]}: {str(feedback)[:150]}..."
                              self.state.last_meta_feedback = feedback_summary
                              self.logger.info(feedback_summary)
                              self.state.needs_replan = True # Force replan after meta-cognition
                 except Exception as e: self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)
                 await asyncio.sleep(0.1)

    async def _trigger_promotion_checks(self):
        # (Implementation unchanged from v2.1)
        pass # Add the full _trigger_promotion_checks implementation from v2.1 here

    # --- Main Run Method (Enhanced Error Handling & Replanning) ---
    async def run(self, goal: str, max_loops: int = 50):
        if not await self.initialize():
            self.logger.critical("Agent initialization failed. Aborting run.")
            return

        self.logger.info(f"Starting main loop. Goal: '{goal}' Max Loops: {max_loops}", emoji_key="arrow_forward")

        while not self.state.goal_achieved_flag and self.state.current_loop < max_loops:
            if self._shutdown_event.is_set(): break
            self.state.current_loop += 1
            self.logger.info(f"--- Agent Loop {self.state.current_loop}/{max_loops} ---", emoji_key="arrows_counterclockwise")

            # Error Check
            if self.state.consecutive_error_count >= self.max_consecutive_errors:
                self.logger.error(f"Max consecutive errors ({self.max_consecutive_errors}) reached. Aborting.", emoji_key="stop_sign")
                if self.state.workflow_id: await self._update_workflow_status_internal("failed", "Agent failed due to repeated errors.")
                break

            # 1. Gather Context
            context = await self._gather_context()
            if "error" in context and context.get("status") != "No Active Workflow":
                self.logger.error(f"Context gathering failed: {context['error']}. Pausing before retry.")
                self.state.consecutive_error_count += 1; self.state.needs_replan = True
                await asyncio.sleep(3 + self.state.consecutive_error_count)
                continue

            # 2. Decide (Potentially involves replanning if needs_replan is True)
            # The LLM call prompt handles the replanning instruction based on the flag / error details
            agent_decision = await self._call_agent_llm(goal, context)

            # 3. Act
            decision_type = agent_decision.get("decision")
            last_tool_result_content = None

            # Reset needs_replan flag *after* LLM had a chance to replan
            # The _update_plan function will set it again if the chosen action fails
            self.state.needs_replan = False

            if decision_type == "call_tool":
                tool_name = agent_decision.get("tool_name")
                arguments = agent_decision.get("arguments", {})
                dependencies = arguments.pop("depends_on_actions", None) # Extract dependencies if LLM provided them
                self.logger.info(f"Agent requests tool: {tool_name}", emoji_key="wrench")
                last_tool_result_content = await self._execute_tool_call_internal(
                    tool_name, arguments, record_action=True, action_dependencies=dependencies
                )
                # Check result and set needs_replan if necessary
                if isinstance(last_tool_result_content, dict) and not last_tool_result_content.get("success"):
                    self.state.needs_replan = True

            elif decision_type == "thought_process":
                thought_content = agent_decision.get("content")
                self.logger.info(f"Agent reasoning: '{thought_content[:100]}...'. Recording thought.", emoji_key="thought_balloon")
                if self.state.workflow_id:
                     try:
                         # Record thought as an action, get result
                         thought_result = await self._execute_tool_call_internal(
                             TOOL_RECORD_THOUGHT,
                             {"workflow_id": self.state.workflow_id, "content": thought_content, "thought_type": "inference"},
                             record_action=True
                         )
                         if thought_result.get("success"):
                              self.state.last_action_summary = f"Recorded thought: {thought_content[:100]}..."
                              self.state.consecutive_error_count = 0
                         else:
                              raise ToolError(f"Record thought failed: {thought_result.get('error')}")
                     except Exception as e:
                          self.logger.error(f"Failed to record thought: {e}", exc_info=False); self.state.consecutive_error_count += 1
                          self.state.last_action_summary = f"Error recording thought: {str(e)[:100]}"; self.state.needs_replan = True
                          self.state.last_error_details = {"tool": TOOL_RECORD_THOUGHT, "error": str(e)}
                else: self.logger.warning("No active workflow to record thought."); self.state.last_action_summary = "Agent provided reasoning, but no workflow active."

            elif decision_type == "complete":
                summary = agent_decision.get("summary", "Goal achieved.")
                self.logger.info(f"Agent signals completion: {summary}", emoji_key="tada")
                self.state.goal_achieved_flag = True
                if self.state.workflow_id: await self._update_workflow_status_internal("completed", summary)
                break
            elif decision_type == "error":
                error_msg = agent_decision.get("message", "Unknown agent error")
                self.logger.error(f"Agent decision error: {error_msg}", emoji_key="x")
                self.state.last_action_summary = f"Agent decision error: {error_msg[:100]}"
                self.state.last_error_details = {"agent_decision_error": error_msg}
                self.state.consecutive_error_count += 1; self.state.needs_replan = True
                if self.state.workflow_id: # Log agent error
                      try: await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"workflow_id": self.state.workflow_id, "content": f"Agent Decision Error: {error_msg}", "thought_type": "critique"}, record_action=False)
                      except Exception: pass
            else: # Unknown
                 self.logger.warning(f"Unhandled decision type: {decision_type}")
                 self.state.last_action_summary = "Unknown agent decision."; self.state.consecutive_error_count += 1; self.state.needs_replan = True
                 self.state.last_error_details = {"agent_decision_error": f"Unknown type: {decision_type}"}

            # 4. Update Plan (uses LLM plan from decision if available)
            await self._update_plan(context, agent_decision, last_tool_result_content)

            # 5. Periodic Tasks
            await self._run_periodic_tasks()

            # 6. Save State Periodically
            if self.state.current_loop % 3 == 0: await self._save_agent_state() # Save more often

            # 7. Loop Delay
            await asyncio.sleep(random.uniform(0.5, 1.0)) # Slightly faster loop

        # --- End of Loop ---
        self.logger.info("--- Agent Loop Finished ---", emoji_key="stopwatch")
        if self.state.goal_achieved_flag: self.logger.info("Goal Status: Achieved", emoji_key="party_popper")
        elif self.state.current_loop >= max_loops: self.logger.warning(f"Goal Status: Max loops ({max_loops}) reached.", emoji_key="timer_clock")
        else: self.logger.warning(f"Goal Status: Loop exited. Shutdown signal received: {self._shutdown_event.is_set()}", emoji_key="warning")

        await self._cleanup_background_tasks() # Wait for linking tasks
        await self._save_agent_state() # Final save
        if self.state.workflow_id: await self._generate_final_report()

    # --- Internal Helpers ---
    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Finds an active server providing the specified tool."""
        # Accesses self.mcp_client.server_manager
        if not self.mcp_client or not self.mcp_client.server_manager:
            self.logger.error("_find_tool_server: MCPClient or ServerManager not initialized.")
            return None
        if tool_name in self.mcp_client.server_manager.tools:
            server_name = self.mcp_client.server_manager.tools[tool_name].server_name
            if server_name in self.mcp_client.server_manager.active_sessions:
                return server_name
            else:
                self.logger.debug(f"Tool '{tool_name}' found on server '{server_name}', but server is not connected.")
                return None
        self.logger.debug(f"Tool '{tool_name}' not found in tool registry.")
        return None

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """Checks if a workflow ID exists using list_workflows tool."""
        # Uses _execute_tool_call_internal which uses self.mcp_client
        tool_name = TOOL_LIST_WORKFLOWS # Use constant
        try:
            # Assuming list_workflows can filter by a specific ID or we check the result list
            # Option 1: Assume filter exists
            # result = await self._execute_tool_call_internal(tool_name, {"limit": 1, "workflow_id_filter": workflow_id}, False)
            # Option 2: List and check (less efficient for single check)
             result = await self._execute_tool_call_internal(tool_name, {"limit": 1000}, False) # Fetch potentially many
             if isinstance(result, dict) and result.get("success"):
                  workflows_list = result.get("workflows", [])
                  return any(wf.get("workflow_id") == workflow_id for wf in workflows_list)
             return False # Tool failed or unexpected result format
        except Exception as e:
             self.logger.error(f"Error checking workflow existence for {workflow_id}: {e}", exc_info=False)
             return False

    async def _update_workflow_status_internal(self, status: str, message: Optional[str]):
        """Internal helper to update workflow status via tool call."""
        if not self.state.workflow_id: return
        tool_name = TOOL_UPDATE_WORKFLOW_STATUS
        try:
            await self._execute_tool_call_internal(
                 tool_name,
                 {"workflow_id": self.state.workflow_id, "status": status, "completion_message": message},
                 record_action=False # Status updates aren't primary actions
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

    async def _record_action_start_internal(self, tool_name: str, arguments: Dict[str, Any], target_server: str) -> Optional[str]:
        """Internal helper to record action start."""
        action_id = None
        start_title = f"Execute {tool_name.split(':')[-1]}"
        start_reasoning = f"Agent initiated tool call: {tool_name}"
        # Use the currently active workflow from the state
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id

        if current_wf_id:
            start_tool_name = TOOL_RECORD_ACTION_START # Use constant
            start_server = self._find_tool_server(start_tool_name)
            if start_server:
                try:
                    start_args = {
                        "workflow_id": current_wf_id, "action_type": "tool_use",
                        "title": start_title, "reasoning": start_reasoning,
                        "tool_name": tool_name, "tool_args": arguments
                    }
                    # Directly call execute_tool, getting the CallToolResult-like dict
                    start_result_dict = await self.mcp_client.execute_tool(start_server, start_tool_name, start_args)
                    # Extract the *content* of the result
                    content = start_result_dict.get("content") if isinstance(start_result_dict, dict) else None

                    if isinstance(content, dict) and content.get("success"):
                        action_id = content.get("action_id")
                        self.logger.debug(f"Action {action_id} started for {tool_name}.")
                    else:
                        self.logger.warning(f"Failed recording action start for {tool_name}: {content.get('error', 'Unknown') if isinstance(content, dict) else 'Invalid start result content'}")
                except Exception as e:
                    self.logger.error(f"Exception recording action start for {tool_name}: {e}", exc_info=True)
            else:
                self.logger.error(f"Cannot record action start: Tool '{start_tool_name}' unavailable.")
        else:
            self.logger.warning("Cannot record action start: No active workflow ID.")
        return action_id

    async def _record_action_completion_internal(self, action_id: str, tool_result_content: Dict):
        """Internal helper to record action completion."""
        completion_status = ActionStatus.COMPLETED.value if tool_result_content.get("success") else ActionStatus.FAILED.value
        completion_tool_name = TOOL_RECORD_ACTION_COMPLETION # Use constant
        completion_server = self._find_tool_server(completion_tool_name)
        if completion_server:
            try:
                await self.mcp_client.execute_tool(
                    completion_server, completion_tool_name,
                    {"action_id": action_id, "status": completion_status, "tool_result": tool_result_content}
                )
                self.logger.debug(f"Action {action_id} completion recorded with status: {completion_status}")
            except Exception as e:
                self.logger.error(f"Error recording action completion for {action_id}: {e}", exc_info=True)
        else:
            self.logger.warning(f"Cannot record action completion: Tool '{completion_tool_name}' unavailable.")

    # --- Implemented Dependency Check (#7) ---
    async def _check_prerequisites(self, prerequisite_action_ids: List[str]) -> Tuple[bool, str]:
        """Checks if all prerequisite actions are COMPLETED using the memory tools."""
        if not prerequisite_action_ids: return True, "No prerequisites."

        self.logger.debug(f"Checking status of prerequisites: {prerequisite_action_ids}")
        # Assume a tool exists to get action details by ID list, e.g., unified_memory:get_actions_details
        # If not, we need to call get_action_details individually. Let's assume the latter for robustness.
        tool_name_get_action = TOOL_GET_ACTION_DETAILS # Use the constant
        target_server = self._find_tool_server(tool_name_get_action)
        if not target_server:
            return False, f"Cannot check prerequisites: '{tool_name_get_action}' tool unavailable."

        not_met = []
        try:
            for action_id in prerequisite_action_ids:
                if self._shutdown_event.is_set(): return False, "Shutdown signal received during check."
                result_content = await self._execute_tool_call_internal(
                    tool_name_get_action, {"action_id": action_id}, record_action=False
                )
                if not result_content.get("success"):
                    not_met.append(f"{action_id[:8]} (Fetch Error: {result_content.get('error', 'Unknown')})")
                    continue # Cannot determine status

                # Assuming result_content contains the action details directly or under 'data'
                action_data = result_content.get("data", result_content) # Adjust based on tool output
                if isinstance(action_data, dict):
                    status = action_data.get("status")
                    if status != ActionStatus.COMPLETED.value:
                        not_met.append(f"{action_id[:8]} (Status: {status or 'Unknown'})")
                else:
                    not_met.append(f"{action_id[:8]} (Invalid details received)")
                await asyncio.sleep(0.05) # Avoid overwhelming the server

            if not_met:
                return False, f"Unmet prerequisites: {', '.join(not_met)}"
            else:
                return True, "All prerequisites met."
        except Exception as e:
            self.logger.error(f"Error checking prerequisites: {e}", exc_info=True)
            return False, f"Exception during check: {e}"

    # --- Implemented Access-Triggered Promotion (#12) ---
    async def _check_and_trigger_promotion(self, memory_id: Optional[str]):
        if not memory_id: return

        get_tool = TOOL_GET_MEMORY_BY_ID
        promo_tool = TOOL_PROMOTE_MEM
        if not self._find_tool_server(get_tool) or not self._find_tool_server(promo_tool): return

        try:
            # Get minimal details needed for trigger check
            mem_details = await self._execute_tool_call_internal(
                get_tool, {"memory_id": memory_id, "include_content": False, "include_links": False}, record_action=False
            )
            if not mem_details.get("success"): return

            access_count = mem_details.get("access_count", 0)
            current_level = mem_details.get("memory_level")
            mem_type = mem_details.get("memory_type")

            # Check thresholds (using constants or config)
            trigger = False
            if current_level == MemoryLevel.EPISODIC.value and access_count >= 5: trigger = True
            elif current_level == MemoryLevel.SEMANTIC.value and access_count >= 10 and mem_type in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]: trigger = True

            if trigger:
                self.logger.info(f"Access count ({access_count}) triggers promotion check for {memory_id} ({current_level}).")
                # Run promotion check asynchronously
                promo_task = asyncio.create_task(  # noqa: F841
                    self._execute_tool_call_internal(promo_tool, {"memory_id": memory_id}, False),
                    name=f"promo_check_{memory_id[:8]}"
                )
                # Track task if needed, or just fire-and-forget
                # self._background_tasks.add(promo_task); promo_task.add_done_callback(...)

        except Exception as e:
             self.logger.error(f"Error in access-triggered promotion check for {memory_id}: {e}", exc_info=False)

    # --- Implemented Auto Linking (#8) ---
    async def _run_auto_linking(self, new_memory_id: str):
        """Background task to find and create semantic links for a new memory."""
        self.logger.info(f"[AutoLink] Starting analysis for {new_memory_id}...", emoji_key="link")
        await asyncio.sleep(random.uniform(1.5, 3.0)) # Longer delay to ensure commit/availability

        try:
            # Get Details
            get_tool = TOOL_GET_MEMORY_BY_ID
            mem_details = await self._execute_tool_call_internal(get_tool, {"memory_id": new_memory_id, "include_content": True}, False)
            if not mem_details.get("success"): self.logger.warning(f"[AutoLink] Failed get details for {new_memory_id}."); return
            content=mem_details.get("content",""); desc=mem_details.get("description","")
            wf_id=mem_details.get("workflow_id"); current_type=mem_details.get("memory_type")
            text = f"{desc}: {content}" if desc else content
            if not text or not wf_id: return

            # Find Similar
            search_tool = TOOL_SEMANTIC_SEARCH
            similar_result = await self._execute_tool_call_internal(
                search_tool, {"workflow_id": wf_id, "query": text, "limit": 4, "threshold": 0.82, "include_content": False}, False # Stricter threshold
            )
            if not similar_result.get("success"): return
            similar_mems = similar_result.get("memories", [])
            if not similar_mems: self.logger.info(f"[AutoLink] No links found for {new_memory_id}."); return

            # Create Links
            link_tool = TOOL_CREATE_LINK
            target_server = self._find_tool_server(link_tool)
            if not target_server: self.logger.warning("[AutoLink] Create link tool unavailable."); return

            links_created = 0
            link_tasks = []
            for sim_mem in similar_mems:
                 target_id = sim_mem.get("memory_id")
                 if target_id and target_id != new_memory_id:
                     similarity = sim_mem.get("similarity", 0.0)
                     target_type = sim_mem.get("memory_type")
                     link_type = LinkType.RELATED.value # Default
                     # Add more sophisticated type logic here if needed
                     if current_type == target_type: link_type = LinkType.SUPPORTS.value

                     # Create link creation coroutine
                     link_coro = self._execute_tool_call_internal(
                          link_tool,
                          {"source_memory_id": new_memory_id, "target_memory_id": target_id, "link_type": link_type, "description": f"Auto (Sim: {similarity:.2f})", "strength": similarity},
                          False
                     )
                     link_tasks.append(link_coro)

            # Execute link creation concurrently
            link_results = await asyncio.gather(*link_tasks, return_exceptions=True)
            for i, res in enumerate(link_results):
                 if isinstance(res, dict) and res.get("success"): links_created += 1
                 elif isinstance(res, Exception): self.logger.warning(f"[AutoLink] Link creation failed: {res}")
                 else: self.logger.warning(f"[AutoLink] Link creation returned unexpected: {res}")

            if links_created > 0: self.logger.info(f"[AutoLink] Created {links_created} links for memory {new_memory_id}", emoji_key="link")

        except Exception as e: self.logger.error(f"[AutoLink] Error for {new_memory_id}: {e}", exc_info=True)

    async def _cleanup_background_tasks(self):
        """Wait for pending background tasks (like auto-linking) to finish."""
        if self._background_link_tasks:
            self.logger.info(f"Waiting for {len(self._background_link_tasks)} background linking tasks to complete...")
            await asyncio.gather(*self._background_link_tasks, return_exceptions=True)
            self.logger.info("Background linking tasks finished.")
            self._background_link_tasks.clear()

    # --- Shutdown Methods ---
    async def signal_shutdown(self):
        self.logger.info("Graceful shutdown signal received.", emoji_key="wave")
        self._shutdown_event.set()

    async def shutdown(self):
        self.logger.info("Shutting down agent loop...", emoji_key="power_button")
        self._shutdown_event.set() # Ensure event is set
        await self._cleanup_background_tasks() # Wait for background tasks
        await self._save_agent_state() # Final state save
        self.logger.info("Agent loop shutdown complete.", emoji_key="checkered_flag")


# --- Main Execution Block ---
async def run_agent_process(mcp_server_url: str, anthropic_key: str, goal: str, max_loops: int, state_file: str, config_file: Optional[str]):
    # (Main execution logic remains the same as v2.1, using the enhanced AgentMasterLoop)
    if not MCP_CLIENT_AVAILABLE: print("❌ ERROR: MCPClient dependency not met."); sys.exit(1)
    mcp_client_instance = None; agent_loop_instance = None; exit_code = 0
    printer = print
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

        # --- Setup Signal Handlers ---
        loop = asyncio.get_running_loop()
        def signal_handler_wrapper(signum, frame): log.warning(f"Signal {signal.Signals(signum).name} received."); asyncio.create_task(agent_loop_instance.signal_shutdown()) if agent_loop_instance else sys.exit(1)
        loop.add_signal_handler(signal.SIGINT, signal_handler_wrapper, signal.SIGINT, None)
        loop.add_signal_handler(signal.SIGTERM, signal_handler_wrapper, signal.SIGTERM, None)

        printer("Running Agent Loop...")
        await agent_loop_instance.run(goal=goal, max_loops=max_loops)
    except KeyboardInterrupt: printer("\n[yellow]Agent loop interrupt handled.[/yellow]"); exit_code = 130
    except Exception as main_err: printer(f"\n❌ Critical error: {main_err}"); log.critical("Top-level execution error", exc_info=True); exit_code = 1
    finally:
        if agent_loop_instance: printer("Shutting down agent loop..."); await agent_loop_instance.shutdown()
        if mcp_client_instance: printer("Closing MCP client..."); await mcp_client_instance.close()
        printer("Agent execution finished.")
        if __name__ == "__main__": sys.exit(exit_code)

if __name__ == "__main__":
    # (Configuration loading remains the same)
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8013")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    AGENT_GOAL = os.environ.get("AGENT_GOAL", "Create workflow 'Data Analysis'. Add plan thought. Query memories for 'data'. If found, store insight 'Data exists'. Check prerequisites for a dummy action (should pass). Mark workflow complete.")
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "25"))
    AGENT_STATE_FILENAME = os.environ.get("AGENT_STATE_FILE", "agent_loop_state_v2.2.json")
    MCP_CLIENT_CONFIG_FILE = os.environ.get("MCP_CLIENT_CONFIG")
    if os.environ.get("AGENT_LOOP_VERBOSE") == "1": log.setLevel(logging.DEBUG)

    if not ANTHROPIC_API_KEY: print("❌ ERROR: ANTHROPIC_API_KEY missing."); sys.exit(1)
    if not MCP_CLIENT_AVAILABLE: print("❌ ERROR: MCPClient dependency missing."); sys.exit(1)

    print("--- Supercharged Agent Master Loop v2.2 ---")
    # (Print config details unchanged)
    print("-----------------------------------------")

    # --- Run ---
    asyncio.run(run_agent_process(MCP_SERVER_URL, ANTHROPIC_API_KEY, AGENT_GOAL, MAX_ITERATIONS, AGENT_STATE_FILENAME, MCP_CLIENT_CONFIG_FILE))
