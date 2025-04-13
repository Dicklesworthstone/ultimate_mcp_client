"""
Supercharged Agent Master Loop - v2.0
====================================

An advanced orchestrator for AI agents using the Unified Memory System
via the Ultimate MCP Client. Integrates dynamic planning, meta-cognition,
dependency tracking, context management, and robust execution logic.

Designed for Claude 3.7 Sonnet (or comparable models with tool use).
"""

import asyncio
import json
import os
import sys
import logging
import time
import random
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# External Libraries
import anthropic
import aiofiles
from anthropic.types import Message, AsyncAnthropic

# --- IMPORT YOUR ACTUAL MCP CLIENT and COMPONENTS ---
# This assumes your client code is structured accessibly
try:
    from mcp_client import MCPClient # The main client class
    from mcp_client import ( # Enums, Exceptions, Utils needed by the loop
        MemoryUtils,
        ToolError, ToolInputError, McpError, # Base exceptions
        ServerStatus, ServerType, # Enums used if checking server state directly
        WorkflowStatus, ActionStatus, ActionType, ArtifactType, ThoughtType, MemoryLevel, MemoryType, LinkType # Core Enums
    )
    MCP_CLIENT_AVAILABLE = True
    print("INFO: Successfully imported MCPClient and required components.")
except ImportError as import_err:
    print(f"❌ CRITICAL ERROR: Could not import MCPClient or required components: {import_err}")
    print("Ensure mcp_client.py is correctly structured and in the Python path.")
    MCPClient = type('DummyMCPClient', (object,), {}) # Dummy for type hints
    MCP_CLIENT_AVAILABLE = False
    # Define dummy exceptions if import failed
    class ToolError(Exception): pass
    class ToolInputError(Exception): pass
    class McpError(Exception): pass
    # Dummy Enums for type hints if import failed
    class WorkflowStatus: ACTIVE="active"; COMPLETED="completed"; FAILED="failed"; PAUSED="paused"; ABANDONED="abandoned"; value="dummy"
    class ActionStatus: COMPLETED="completed"; FAILED="failed"; IN_PROGRESS="in_progress"; PLANNED="planned"; SKIPPED="skipped"; value="dummy"
    class MemoryLevel: EPISODIC = "episodic"; SEMANTIC = "semantic"; PROCEDURAL="procedural"; WORKING="working"; value="dummy"
    class MemoryType: ACTION_LOG = "action_log"; REASONING_STEP="reasoning_step"; PLAN="plan"; ERROR="error"; value="dummy"
    class ThoughtType: INFERENCE = "inference"; CRITIQUE="critique"; GOAL="goal"; DECISION="decision"; SUMMARY="summary"; REFLECTION="reflection"; PLAN="plan"; HYPOTHESIS="hypothesis"; QUESTION="question"; value="dummy"
    class LinkType: REQUIRES="requires"; INFORMS="informs"; BLOCKS="blocks"; value="dummy"

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
AGENT_STATE_FILE = "agent_loop_state.json"
DEFAULT_PLAN = "Initial state: Assess goal, gather context, and formulate initial plan."
# Meta-cognition Intervals/Thresholds (Configurable maybe later)
REFLECTION_SUCCESS_THRESHOLD = 7
CONSOLIDATION_SUCCESS_THRESHOLD = 15
OPTIMIZATION_LOOP_INTERVAL = 10
MEMORY_PROMOTION_LOOP_INTERVAL = 25 # Check less often
# Tool Names (ensure these match names registered by your memory server)
TOOL_GET_CONTEXT = "unified_memory:get_workflow_context"
TOOL_GET_CONTEXT_BUNDLE = "unified_memory:get_agent_context_bundle" # Preferred, if implemented
TOOL_CREATE_WORKFLOW = "unified_memory:create_workflow"
TOOL_UPDATE_WORKFLOW_STATUS = "unified_memory:update_workflow_status"
TOOL_RECORD_THOUGHT = "unified_memory:record_thought"
TOOL_ADD_DEPENDENCY = "unified_memory:add_action_dependency"
TOOL_GET_DEPENDENCIES = "unified_memory:get_action_dependencies"
TOOL_GET_LINKED_MEMORIES = "unified_memory:get_linked_memories" # Assumes this exists
TOOL_REFLECTION = "unified_memory:generate_reflection"
TOOL_CONSOLIDATION = "unified_memory:consolidate_memories"
TOOL_OPTIMIZE_WM = "unified_memory:optimize_working_memory"
TOOL_AUTO_FOCUS = "unified_memory:auto_update_focus"
TOOL_PROMOTE_MEM = "unified_memory:promote_memory_level"


# --- Agent Loop Class ---
class AgentMasterLoop:
    """Sophisticated orchestrator for the Unified Memory System."""

    def __init__(self, mcp_client_instance: MCPClient, agent_state_file: str = AGENT_STATE_FILE):
        if not MCP_CLIENT_AVAILABLE:
            raise RuntimeError("MCPClient class not found. Cannot initialize AgentMasterLoop.")
        if not isinstance(mcp_client_instance, MCPClient):
            raise TypeError("AgentMasterLoop requires a valid MCPClient instance.")

        self.mcp_client = mcp_client_instance
        self.anthropic_client = self.mcp_client.anthropic
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        if not self.anthropic_client:
            self.logger.critical("Anthropic client is unavailable. Agent decision-making will fail.")
            raise ValueError("Anthropic client required but not initialized in MCPClient.")

        # Core Agent State (initialized via _load_agent_state or defaults)
        self.workflow_id: Optional[str] = None
        self.context_id: Optional[str] = None
        self.workflow_stack: List[str] = [] # For hierarchical tasks (#2)
        self.current_plan: str = DEFAULT_PLAN
        self.last_action_summary: str = "Loop initialized."
        self.consecutive_error_count: int = 0
        self.max_consecutive_errors: int = 3
        self.current_loop: int = 0
        self.goal_achieved_flag: bool = False
        self.needs_replan: bool = False # Flag for dynamic replanning (#1)
        self.last_error_details: Optional[Dict] = None # (#8)

        # Meta-Cognition State
        self.successful_actions_since_reflection: int = 0
        self.successful_actions_since_consolidation: int = 0
        self.loops_since_optimization: int = 0
        self.loops_since_promotion_check: int = 0
        self.reflection_cycle_index: int = 0
        # Configurable thresholds
        self.reflection_threshold: int = REFLECTION_SUCCESS_THRESHOLD
        self.consolidation_threshold: int = CONSOLIDATION_SUCCESS_THRESHOLD
        self.optimization_interval: int = OPTIMIZATION_LOOP_INTERVAL
        self.promotion_interval: int = MEMORY_PROMOTION_LOOP_INTERVAL
        self.reflection_type_sequence: List[str] = ["progress", "gaps", "strengths"]

        self.tool_schemas: List[Dict[str, Any]] = []
        self._shutdown_event = asyncio.Event() # For graceful shutdown signalling

    # --- Initialization and State Persistence (#15) ---
    async def initialize(self) -> bool:
        """Loads agent state, verifies client setup, loads tool schemas."""
        self.logger.info("Initializing agent loop...", emoji_key="gear")
        try:
            await self._load_agent_state() # Load persistent state first

            if not self.mcp_client.server_manager:
                self.logger.error("MCP Client Server Manager not initialized.")
                return False

            # Load/Filter tool schemas
            all_tools_for_api = self.mcp_client.server_manager.format_tools_for_anthropic()
            self.tool_schemas = [
                schema for schema in all_tools_for_api
                if self.mcp_client.server_manager.sanitized_to_original.get(schema['name'], '').startswith("unified_memory:")
            ]
            if not self.tool_schemas:
                self.logger.warning("No 'unified_memory:*' tools loaded. Agent capabilities limited.", emoji_key="warning")
            else:
                 self.logger.info(f"Loaded {len(self.tool_schemas)} unified_memory tool schemas.", emoji_key="clipboard")

            # Check if the currently loaded workflow_id is still valid (optional)
            if self.workflow_id:
                 if not await self._check_workflow_exists(self.workflow_id):
                      self.logger.warning(f"Loaded workflow_id {self.workflow_id} no longer exists. Resetting.")
                      self.workflow_id = None
                      self.context_id = None
                      self.workflow_stack = []
                      self.current_plan = DEFAULT_PLAN
                      await self._save_agent_state() # Save reset state

            self.logger.info("Agent loop initialized successfully.")
            return True
        except Exception as e:
            self.logger.critical(f"Agent loop initialization failed: {e}", exc_info=True)
            return False

    async def _save_agent_state(self):
        """Saves the agent loop's state to a JSON file."""
        state = {
            "workflow_id": self.workflow_id,
            "context_id": self.context_id,
            "workflow_stack": self.workflow_stack,
            "current_plan": self.current_plan,
            "last_action_summary": self.last_action_summary,
            "consecutive_error_count": self.consecutive_error_count,
            "current_loop": self.current_loop,
            "goal_achieved_flag": self.goal_achieved_flag,
            "needs_replan": self.needs_replan,
            "successful_actions_since_reflection": self.successful_actions_since_reflection,
            "successful_actions_since_consolidation": self.successful_actions_since_consolidation,
            "loops_since_optimization": self.loops_since_optimization,
            "loops_since_promotion_check": self.loops_since_promotion_check,
            "reflection_cycle_index": self.reflection_cycle_index,
            "last_error_details": self.last_error_details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            async with aiofiles.open(self.agent_state_file, 'w') as f:
                await f.write(json.dumps(state, indent=2))
            self.logger.debug(f"Agent state saved to {self.agent_state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)

    async def _load_agent_state(self):
        """Loads the agent loop's state from a JSON file."""
        if not self.agent_state_file.exists():
            self.logger.info("No previous agent state file found. Starting fresh.")
            return

        try:
            async with aiofiles.open(self.agent_state_file, 'r') as f:
                state = json.loads(await f.read())

            # Load state variables, using defaults if keys are missing
            self.workflow_id = state.get("workflow_id")
            self.context_id = state.get("context_id", self.workflow_id) # Default context to workflow
            self.workflow_stack = state.get("workflow_stack", [self.workflow_id] if self.workflow_id else [])
            self.current_plan = state.get("current_plan", DEFAULT_PLAN)
            self.last_action_summary = state.get("last_action_summary", "State loaded.")
            self.consecutive_error_count = state.get("consecutive_error_count", 0)
            self.current_loop = state.get("current_loop", 0) # Resume loop count
            self.goal_achieved_flag = state.get("goal_achieved_flag", False)
            self.needs_replan = state.get("needs_replan", False)
            self.last_error_details = state.get("last_error_details")
            self.successful_actions_since_reflection = state.get("successful_actions_since_reflection", 0)
            self.successful_actions_since_consolidation = state.get("successful_actions_since_consolidation", 0)
            self.loops_since_optimization = state.get("loops_since_optimization", 0)
            self.loops_since_promotion_check = state.get("loops_since_promotion_check", 0)
            self.reflection_cycle_index = state.get("reflection_cycle_index", 0)

            self.logger.info(f"Agent state loaded successfully from {self.agent_state_file}. Resuming loop {self.current_loop + 1}.")
            self.logger.info(f"Resuming workflow: {self.workflow_id}" if self.workflow_id else "No workflow loaded.")

        except FileNotFoundError:
             self.logger.info("Agent state file not found.") # Should be caught by exists() check but good practice
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            self.logger.error(f"Failed to load or parse agent state from {self.agent_state_file}: {e}. Starting fresh.", exc_info=True)
            # Reset state to defaults if load fails
            await self._reset_state_to_defaults()
        except Exception as e:
             self.logger.error(f"Unexpected error loading agent state: {e}", exc_info=True)
             await self._reset_state_to_defaults()

    async def _reset_state_to_defaults(self):
        """Resets agent state variables to their initial default values."""
        self.workflow_id = None
        self.context_id = None
        self.workflow_stack = []
        self.current_plan = DEFAULT_PLAN
        self.last_action_summary = "State reset due to load error."
        self.consecutive_error_count = 0
        self.current_loop = 0
        self.goal_achieved_flag = False
        self.needs_replan = False
        self.last_error_details = None
        self.successful_actions_since_reflection = 0
        self.successful_actions_since_consolidation = 0
        self.loops_since_optimization = 0
        self.loops_since_promotion_check = 0
        self.reflection_cycle_index = 0

    # --- Core Agent Methods Implementation ---

    async def _gather_context(self) -> Dict[str, Any]:
        """Gathers comprehensive context, preferring bundled tool if available."""
        self.logger.info("Gathering context...", emoji_key="satellite")
        base_context = {
            "current_loop": self.current_loop,
            "current_plan": self.current_plan,
            "last_action_summary": self.last_action_summary,
            "consecutive_errors": self.consecutive_error_count,
            "last_error_details": self.last_error_details # Include last error (#8)
        }

        if not self.workflow_id:
            base_context["status"] = "No Active Workflow"
            base_context["message"] = "Need to create or load a workflow."
            return base_context

        # Check for preferred bundled context tool (#13)
        bundle_tool_name = TOOL_GET_CONTEXT_BUNDLE
        bundle_server = self._find_tool_server(bundle_tool_name)

        if bundle_server:
            self.logger.debug(f"Attempting context retrieval using bundled tool: {bundle_tool_name}")
            try:
                tool_result_content = await self._execute_tool_call_internal(
                    tool_name=bundle_tool_name,
                    arguments={"workflow_id": self.workflow_id, "context_size": 7}, # Example args
                    record_action=False
                )
                if isinstance(tool_result_content, dict) and tool_result_content.get("success"):
                    self.logger.info("Context gathered successfully via bundled tool.", emoji_key="signal_strength")
                    # Merge fetched context into base context
                    base_context.update(tool_result_content)
                    base_context.pop("success", None) # Remove redundant flag
                    base_context["status"] = "Context Ready (Bundled)"
                    return base_context
                else:
                    self.logger.warning(f"Bundled context tool '{bundle_tool_name}' failed: {tool_result_content.get('error')}. Falling back.")
            except Exception as e:
                self.logger.warning(f"Error calling bundled context tool '{bundle_tool_name}': {e}. Falling back.", exc_info=False)

        # Fallback to original context tool
        tool_name = TOOL_GET_CONTEXT
        target_server = self._find_tool_server(tool_name)
        if not target_server:
            base_context["status"] = "Tool Not Available"
            base_context["error"] = f"Context tool '{tool_name}' unavailable."
            return base_context

        try:
            tool_result_content = await self._execute_tool_call_internal(
                tool_name=tool_name,
                arguments={
                    "workflow_id": self.workflow_id,
                    "recent_actions_limit": 7,
                    "important_memories_limit": 5,
                    "key_thoughts_limit": 5
                },
                record_action=False
            )
            if isinstance(tool_result_content, dict) and tool_result_content.get("success"):
                self.logger.info("Context gathered successfully via standard tool.", emoji_key="signal_strength")
                base_context.update(tool_result_content)
                base_context.pop("success", None)
                base_context["status"] = "Context Ready (Standard)"
                return base_context
            else:
                error_msg = f"Standard context tool failed: {tool_result_content.get('error', 'Unknown')}"
                self.logger.warning(error_msg)
                base_context["status"] = "Tool Execution Failed"
                base_context["error"] = error_msg
                return base_context
        except Exception as e:
            self.logger.error(f"Exception gathering context via standard tool: {e}", exc_info=True)
            base_context["status"] = "Context Gathering Error"
            base_context["error"] = f"Exception: {e}"
            return base_context

    def _construct_agent_prompt(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Constructs the detailed messages list including planning and error handling."""
        # --- System Prompt (More detailed instructions) ---
        system_prompt = f"""You are 'Maestro', an advanced AI agent using a Unified Memory System. Your goal is to achieve the user's objective by strategically using memory tools.

Overall Goal: {goal}

Available Unified Memory Tools (Use ONLY these):
"""
        # Append tool schemas
        if not self.tool_schemas:
            system_prompt += "\n- CRITICAL WARNING: No unified_memory tools loaded.\n"
        else:
            for schema in self.tool_schemas:
                 sanitized_name = schema['name']
                 original_name = self.mcp_client.server_manager.sanitized_to_original.get(sanitized_name, sanitized_name)
                 system_prompt += f"\n- Name: `{sanitized_name}` (Maps to: `{original_name}`)\n"
                 system_prompt += f"  Desc: {schema.get('description', 'N/A')}\n"
                 system_prompt += f"  Schema: {json.dumps(schema['input_schema'])}\n"

        system_prompt += f"""
Your Process:
1.  Context Analysis: Deeply analyze the 'Current Context' below. Note workflow status, errors, recent actions, memories (consider confidence!), thoughts, and the current plan.
2.  Error Handling: If `last_error_details` is present, EXPLICITLY address the error in your reasoning. Explain why it happened and how your new plan avoids repeating it.
3.  Reasoning & Planning:
    a. State your step-by-step reasoning towards the Overall Goal based on the context and *current* plan.
    b. Evaluate the `current_plan`. Is it still valid? Does it address recent events/errors?
    c. Propose an **Updated Plan** (1-3 concise steps). If replanning significantly, state why. Log important planning thoughts with `record_thought`.
4.  Action Decision: Choose the **single best** next action based on your Updated Plan:
    *   Call Memory Tool: Select the most precise `unified_memory:*` tool. Provide arguments matching the schema. **Mandatory:** If context shows 'No Active Workflow', you MUST call `create_workflow`. **Dependency Check:** Before choosing an action, consider if it depends on prior actions using `get_action_dependencies` (if needed and planned).
    *   Record Thought: Use `record_thought` for logging internal reasoning, questions, hypotheses, detailed plans, or critiques.
    *   Signal Completion: If the Overall Goal is MET, respond ONLY with "Goal Achieved:" and a final summary.
5.  Output Format: Respond **ONLY** with the valid JSON for the chosen tool call OR the "Goal Achieved:" text. NO conversational filler.

Tool Usage Guidance:
*   `store_memory`: For facts, observations, results. Include `importance` and `confidence`.
*   `query_memories`: Filter precisely (tags, level, type, importance). Use `search_text` for keywords.
*   `search_semantic_memories`: For conceptual similarity. Use precise `query`.
*   `create_memory_link`: Link related memories (use `supports`, `contradicts`, `causal`, etc.).
*   `record_action_start`/`completion`: Your orchestrator handles these; focus on the *primary* tool call for the task.
*   `add_action_dependency`: Use during planning if Action B requires Action A.
*   `get_linked_memories`: Explore connections of a specific memory.
*   `consolidate_memories`/`generate_reflection`: Use periodically when prompted or if reasoning suggests it's beneficial.
"""

        # --- User Prompt ---
        context_str = json.dumps(context, indent=2, default=str, ensure_ascii=False)
        # Truncate context if extremely long to prevent API errors
        max_context_len = 15000 # Example limit
        if len(context_str) > max_context_len:
            context_str = context_str[:max_context_len] + "\n... (Context Truncated)\n}"
            self.logger.warning("Truncated context string sent to LLM due to length.")

        user_prompt = f"Current Context:\n```json\n{context_str}\n```\n\n"
        user_prompt += f"My Current Plan:\n```\n{self.current_plan}\n```\n\n"
        user_prompt += f"Last Action Summary:\n{self.last_action_summary}\n\n"
        if self.last_error_details: # Include specific error details (#8)
             user_prompt += f"Last Error Details:\n```json\n{json.dumps(self.last_error_details, indent=2)}\n```\n\n"
        user_prompt += f"Overall Goal: {goal}\n\n"
        # Explicit instruction including planning (#1) and error handling (#8)
        user_prompt += "**Instruction:** Analyze context (especially errors). Reason step-by-step. Propose/Confirm Updated Plan. Decide next action (Tool JSON or 'Goal Achieved:')."

        return [{"role": "user", "content": system_prompt + "\n---\n" + user_prompt}]

    async def _call_agent_llm(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Claude 3.7 Sonnet, parses response for tool call or completion."""
        # (Implementation is largely the same as the previous refined version,
        #  using self.anthropic_client, self.tool_schemas, and error handling.
        #  No major changes needed here besides ensuring the prompt construction
        #  is called correctly.)
        self.logger.info("Calling Claude 3.7 Sonnet for decision...", emoji_key="robot_face")
        if not self.anthropic_client:
            return {"decision": "error", "message": "Anthropic client not available."}

        messages = self._construct_agent_prompt(goal, context)
        api_tools = self.tool_schemas # Use the filtered schemas

        try:
            response: Message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=3000, # Increased further for complex planning/reasoning
                messages=messages,
                tools=api_tools if api_tools else None, # Handle case where no tools are loaded
                tool_choice={"type": "auto"},
                temperature=0.6 # Slightly lower temp for more deterministic planning/tool use
            )

            self.logger.debug(f"LLM Raw Response Stop Reason: {response.stop_reason}")

            decision = {"decision": "error", "message": "LLM provided no actionable output."}
            text_response_parts = []
            tool_call_detected = None
            plan_update_text = None # Extract plan if provided

            for block in response.content:
                if block.type == "text":
                    text_response_parts.append(block.text)
                    # Check for explicit plan update in text (heuristic)
                    if "Updated Plan:" in block.text:
                         # Extract text after "Updated Plan:"
                         plan_update_text = block.text.split("Updated Plan:", 1)[-1].strip()
                elif block.type == "tool_use":
                    tool_call_detected = block
                    # Prioritize tool call over text interpretation for plan
                    break

            full_text_response = "".join(text_response_parts).strip()

            if tool_call_detected:
                # (Same tool call processing logic as before...)
                tool_name_sanitized = tool_call_detected.name
                tool_input = tool_call_detected.input or {}
                original_tool_name = self.mcp_client.server_manager.sanitized_to_original.get(tool_name_sanitized, tool_name_sanitized)
                self.logger.info(f"LLM chose tool: {original_tool_name} (Sanitized: {tool_name_sanitized})", emoji_key="hammer_and_wrench")
                if original_tool_name.startswith("unified_memory:"):
                    decision = {"decision": "call_tool", "tool_name": original_tool_name, "arguments": tool_input}
                else:
                     self.logger.warning(f"LLM called non-unified_memory tool '{original_tool_name}'. Treating as reasoning.")
                     decision = {"decision": "thought_process", "content": full_text_response} # Fallback to text

            elif full_text_response.startswith("Goal Achieved:"):
                decision = {"decision": "complete", "summary": full_text_response.replace("Goal Achieved:", "").strip()}
            elif full_text_response:
                # Assume text is reasoning or a plan update if no tool call/completion
                decision = {"decision": "thought_process", "content": full_text_response}
                self.logger.info("LLM provided text reasoning/plan update.")
            # else: decision remains the default error

            # Store extracted plan update if found (#1)
            if plan_update_text:
                 decision["updated_plan"] = plan_update_text

            self.logger.debug(f"Agent Decision Parsed: {decision}")
            return decision

        # (Keep Anthropic API error handling)
        except anthropic.APIConnectionError as e: msg = f"API Connection Error: {e}"; self.logger.error(msg, exc_info=True)
        except anthropic.RateLimitError as e: msg = "Rate limit exceeded."; self.logger.error(msg, exc_info=True); await asyncio.sleep(random.uniform(5, 10))
        except anthropic.APIStatusError as e: msg = f"API Error {e.status_code}: {e.message}"; self.logger.error(f"Anthropic API status error: {e.status_code} - {e.response}", exc_info=True)
        except Exception as e: msg = f"Unexpected LLM interaction error: {e}"; self.logger.error(msg, exc_info=True)
        return {"decision": "error", "message": msg}


    async def _execute_tool_call_internal(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        record_action: bool = True
        ) -> Dict[str, Any]:
        """Internal helper to find server, execute tool, handle result, optionally record actions."""
        action_id = None
        tool_result_content = {"success": False, "error": "Execution error."} # Default

        # --- 1. Find Target Server ---
        target_server = self._find_tool_server(tool_name)
        if not target_server:
             err_msg = f"Cannot execute '{tool_name}': Tool not found or server not connected."
             self.logger.error(err_msg)
             self.last_error_details = {"tool": tool_name, "error": err_msg} # (#8)
             return {"success": False, "error": err_msg}

        # --- 2. Dependency Check (#7) ---
        # Optional: Add check here if the LLM planned this action and specified prerequisites
        # Example (needs agent plan structure):
        # if planned_action_info and planned_action_info.get('prerequisites'):
        #     prereq_check_result = await self._check_prerequisites(planned_action_info['prerequisites'])
        #     if not prereq_check_result['met']:
        #         err_msg = f"Prerequisites not met for {tool_name}: {prereq_check_result['reason']}"
        #         self.logger.warning(err_msg)
        #         self.last_error_details = {"tool": tool_name, "error": err_msg, "type": "dependency_failure"}
        #         return {"success": False, "error": err_msg}

        # --- 3. Record Action Start (Optional) ---
        if record_action:
             action_id = await self._record_action_start_internal(tool_name, arguments, target_server)

        # --- 4. Execute Primary Tool ---
        try:
             # Ensure workflow_id is passed correctly
             if 'workflow_id' not in arguments and self.workflow_id and tool_name not in [TOOL_CREATE_WORKFLOW, "unified_memory:list_workflows"]:
                  arguments['workflow_id'] = self.workflow_id

             # Use the robust execute_tool from MCPClient
             call_tool_result = await self.mcp_client.execute_tool(target_server, tool_name, arguments)

             # Parse result (assuming dict like CallToolResult)
             if isinstance(call_tool_result, dict):
                 tool_result_content = call_tool_result.get("content", {"error": "Tool returned no content"})
                 is_error = call_tool_result.get("isError", True)
                 if is_error:
                     tool_result_content = {"success": False, "error": str(tool_result_content)}
                 elif isinstance(tool_result_content, dict) and "success" not in tool_result_content:
                     # Wrap successful non-standard results
                     tool_result_content = {"success": True, "data": tool_result_content}
                 elif not isinstance(tool_result_content, dict):
                     # Wrap other successful non-dict results
                     tool_result_content = {"success": True, "data": tool_result_content}
             else:
                 tool_result_content = {"success": False, "error": f"Unexpected result type: {type(call_tool_result)}"}

             self.logger.info(f"Tool {tool_name} executed. Success: {tool_result_content.get('success')}", emoji_key="checkered_flag")
             # Update last summary *before* recording completion
             self.last_action_summary = f"Executed {tool_name}. Success: {tool_result_content.get('success')}."
             if not tool_result_content.get('success'):
                  err_detail = str(tool_result_content.get('error', 'Unknown'))[:150]
                  self.last_action_summary += f" Error: {err_detail}"
                  self.last_error_details = {"tool": tool_name, "args": arguments, "error": err_detail, "result": tool_result_content} # (#8)
             else:
                  self.last_error_details = None # Clear last error on success

        except (ToolError, ToolInputError) as e:
             self.logger.error(f"Tool Error executing {tool_name}: {e}", exc_info=False)
             err_str = str(e)
             tool_result_content = {"success": False, "error": err_str}
             self.last_action_summary = f"Execution failed: Tool {tool_name} error: {err_str[:100]}"
             self.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": type(e).__name__} # (#8)
        except Exception as e:
             self.logger.error(f"Unexpected Error executing tool {tool_name}: {e}", exc_info=True)
             err_str = str(e)
             tool_result_content = {"success": False, "error": f"Unexpected error: {err_str}"}
             self.last_action_summary = f"Execution failed: Unexpected error in {tool_name}."
             self.last_error_details = {"tool": tool_name, "args": arguments, "error": err_str, "type": "Unexpected"} # (#8)

        # --- 5. Record Action Completion (Optional) ---
        if record_action and action_id:
             await self._record_action_completion_internal(action_id, tool_result_content)

        # --- 6. Handle Workflow Creation Side Effect ---
        if tool_name == TOOL_CREATE_WORKFLOW and tool_result_content.get("success"):
            new_wf_id = tool_result_content.get("workflow_id")
            if new_wf_id:
                self.workflow_id = new_wf_id
                self.context_id = new_wf_id # Set context ID
                self.workflow_stack = [new_wf_id] # Reset stack with new root (#2)
                self.logger.info(f"Switched to newly created workflow: {self.workflow_id}", emoji_key="label")

        # --- 7. Result Summarization (Placeholder) ---
        # (#6) If tool_result_content is large and success=True, trigger summarization here.
        # summarized_content = await self._summarize_if_needed(tool_result_content)
        # return summarized_content # Return summary instead of full result

        return tool_result_content # Return the actual content dict

    async def _record_action_start_internal(self, tool_name: str, arguments: Dict[str, Any], target_server: str) -> Optional[str]:
        """Internal helper to record action start."""
        action_id = None
        start_title = f"Execute {tool_name.split(':')[-1]}"
        start_reasoning = f"Agent initiated tool call: {tool_name}"
        current_wf_id = self.workflow_id

        if current_wf_id:
            start_tool_name = "unified_memory:record_action_start"
            start_server = self._find_tool_server(start_tool_name)
            if start_server:
                try:
                    start_args = {
                        "workflow_id": current_wf_id, "action_type": "tool_use",
                        "title": start_title, "reasoning": start_reasoning,
                        "tool_name": tool_name, "tool_args": arguments
                    }
                    start_result = await self.mcp_client.execute_tool(start_server, start_tool_name, start_args)
                    content = start_result.get("content") if isinstance(start_result, dict) else None
                    if isinstance(content, dict) and content.get("success"):
                        action_id = content.get("action_id")
                        self.logger.debug(f"Action {action_id} started for {tool_name}.")
                    else:
                        self.logger.warning(f"Failed recording action start for {tool_name}: {content.get('error', 'Unknown')}")
                except Exception as e:
                    self.logger.error(f"Exception recording action start for {tool_name}: {e}", exc_info=True)
        return action_id

    async def _record_action_completion_internal(self, action_id: str, tool_result_content: Dict):
        """Internal helper to record action completion."""
        completion_status = ActionStatus.COMPLETED.value if tool_result_content.get("success") else ActionStatus.FAILED.value
        completion_tool_name = "unified_memory:record_action_completion"
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

    # --- Plan Update (incorporates LLM plan output) ---
    async def _update_plan(self, context: Dict[str, Any], last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        """Updates the agent's internal plan, potentially using LLM output."""
        self.logger.info("Updating agent plan...", emoji_key="clipboard")
        tool_success = False
        error_msg = None
        llm_updated_plan = last_decision.get("updated_plan") # (#1) Check if LLM provided a plan

        if last_decision.get("decision") == "call_tool":
             # (Error checking logic remains the same as previous)
             tool_name = last_decision.get("tool_name", "Unknown Tool")
             if isinstance(last_tool_result_content, dict):
                 tool_success = last_tool_result_content.get("success", False)
                 if not tool_success: error_msg = str(last_tool_result_content.get("error", "Tool failure"))
             else:
                 tool_success = False
                 error_msg = f"Tool execution failed or returned invalid result type: {type(last_tool_result_content)}"

             if tool_success:
                 self.current_plan = llm_updated_plan or f"After running {tool_name}, analyze output ({str(last_tool_result_content)[:50]}...) and proceed."
                 self.consecutive_error_count = 0
                 self.successful_actions_since_reflection += 1
                 self.successful_actions_since_consolidation += 1
                 self.needs_replan = False # Reset replan flag on success
             else:
                 self.current_plan = llm_updated_plan or f"Tool {tool_name} failed. Error: '{str(error_msg)[:100]}...'. Analyze failure and replan."
                 self.consecutive_error_count += 1
                 self.needs_replan = True # Set replan flag on error (#1)
                 self.successful_actions_since_reflection = self.reflection_threshold # Trigger reflection

        elif last_decision.get("decision") == "thought_process":
             thought = last_decision.get('content','')[:100]
             self.current_plan = llm_updated_plan or f"After thought '{thought}...', decide next concrete action."
             self.consecutive_error_count = 0
             self.needs_replan = False
        elif last_decision.get("decision") == "complete":
             self.current_plan = "Goal Achieved. Finalizing."
             self.consecutive_error_count = 0
             self.needs_replan = False
        elif last_decision.get("decision") == "error":
             error_msg = last_decision.get("message", "Unknown agent error")
             self.current_plan = llm_updated_plan or f"Agent error: '{error_msg[:100]}...'. Attempt recovery/re-evaluation."
             self.consecutive_error_count += 1
             self.needs_replan = True
             self.successful_actions_since_reflection = self.reflection_threshold
        else:
             self.logger.warning(f"Cannot update plan for unknown decision: {last_decision.get('decision')}")
             self.current_plan = llm_updated_plan or "Agent decision unclear. Re-evaluate context and plan."
             self.consecutive_error_count += 1
             self.needs_replan = True

        self.logger.info(f"New Plan: {self.current_plan}")

    # --- Periodic Tasks (smarter triggers) ---
    async def _run_periodic_tasks(self):
        """Runs maintenance and meta-cognitive tasks based on state."""
        if not self.workflow_id or not self.context_id: return
        if self._shutdown_event.is_set(): return # Don't run if shutting down

        tasks_to_run: List[Tuple[str, Dict]] = []
        now = time.time()

        # --- Reflection Trigger (#4 - Feedback) ---
        if self.needs_replan or self.successful_actions_since_reflection >= self.reflection_threshold:
             reflection_type = self.reflection_type_sequence[self.reflection_cycle_index % len(self.reflection_type_sequence)]
             tasks_to_run.append((TOOL_REFLECTION, {"workflow_id": self.workflow_id, "reflection_type": reflection_type}))
             self.successful_actions_since_reflection = 0
             self.reflection_cycle_index += 1

        # --- Consolidation Trigger (#4 - Feedback) ---
        if self.successful_actions_since_consolidation >= self.consolidation_threshold:
             tasks_to_run.append((TOOL_CONSOLIDATION, {
                 "workflow_id": self.workflow_id, "consolidation_type": "summary",
                 "query_filter": {"memory_level": self.consolidation_memory_level},
                 "max_source_memories": self.consolidation_max_sources
             }))
             self.successful_actions_since_consolidation = 0

        # --- Optimization & Focus Trigger (#9 - Dynamic WM) ---
        self.loops_since_optimization += 1
        # Also check if working memory size exceeds threshold? Needs context fetch.
        # Simplified for now: use loop interval.
        if self.loops_since_optimization >= self.optimization_interval:
             tasks_to_run.append((TOOL_OPTIMIZE_WM, {"context_id": self.context_id}))
             tasks_to_run.append((TOOL_AUTO_FOCUS, {"context_id": self.context_id}))
             self.loops_since_optimization = 0

        # --- Memory Promotion Check (#10 - Auto Promotion) ---
        # --- Memory Promotion Check (#10 - Auto Promotion) ---
        self.loops_since_promotion_check += 1
        if self.loops_since_promotion_check >= self.promotion_interval:
            self.loops_since_promotion_check = 0 # Reset counter immediately
            self.logger.info("Checking for promotable memories using unified_memory:query_memories...", emoji_key="level_slider")

            # --- Step 1: Query for Episodic -> Semantic Candidates ---
            query_tool_name = "unified_memory:query_memories"
            promote_tool_name = TOOL_PROMOTE_MEM # Use constant TOOL_PROMOTE_MEM

            # Define criteria for Episodic -> Semantic promotion
            episodic_promo_criteria = {
                "workflow_id": self.workflow_id,
                "memory_level": MemoryLevel.EPISODIC.value,
                "min_access_count": 5, # Configurable? For now, hardcoded example
                "min_confidence": 0.8,
                "sort_by": "importance", # Prioritize important ones first
                "sort_order": "DESC",
                "limit": 10, # Check up to 10 candidates per cycle
                "include_content": False # Don't need full content
            }

            promo_candidates_ep_to_sem = []
            try:
                 # Use the internal helper to execute the query
                 query_result = await self._execute_tool_call_internal(
                     tool_name=query_tool_name,
                     arguments=episodic_promo_criteria,
                     record_action=False # Don't record this internal check as a main action
                 )

                 # Check if the query was successful and extract memory IDs
                 if isinstance(query_result, dict) and query_result.get("success"):
                     # Assuming query_result['data']['memories'] contains the list
                     memories_found = query_result.get("data", {}).get("memories", [])
                     promo_candidates_ep_to_sem = [mem.get("memory_id") for mem in memories_found if mem.get("memory_id")]
                     if promo_candidates_ep_to_sem:
                         self.logger.info(f"Found {len(promo_candidates_ep_to_sem)} Episodic candidates for promotion check.")

                 elif isinstance(query_result, dict): # Query tool reported failure
                      self.logger.warning(f"Query for Episodic promotion candidates failed: {query_result.get('error')}")
                 else: # Unexpected result
                      self.logger.warning(f"Unexpected result querying Episodic promotion candidates: {query_result}")

            except Exception as e:
                 self.logger.warning(f"Error querying for Episodic promotion candidates: {e}", exc_info=False) # Log less verbose

            # --- Step 2: Query for Semantic -> Procedural Candidates ---
            # Define criteria for Semantic -> Procedural promotion
            # Requires specific memory types (procedure, skill) + high usage/confidence
            semantic_promo_criteria = {
                "workflow_id": self.workflow_id,
                "memory_level": MemoryLevel.SEMANTIC.value,
                # Filter by type *within* the query if supported, otherwise filter after fetch
                "memory_type_in": [MemoryType.PROCEDURE.value, MemoryType.SKILL.value], # Assumes query_memories supports 'in' filter
                "min_access_count": 10, # Higher threshold for procedural
                "min_confidence": 0.9,
                "sort_by": "importance",
                "sort_order": "DESC",
                "limit": 5, # Check fewer candidates for procedural
                "include_content": False
            }
            # NOTE: If query_memories doesn't support 'memory_type_in', fetch candidates
            # based on level/access/confidence and then filter the results in the loop below.
            # Assuming for now it supports it (or can be adapted to).

            promo_candidates_sem_to_proc = []
            try:
                 # Use the internal helper to execute the query
                 query_result = await self._execute_tool_call_internal(
                     tool_name=query_tool_name,
                     arguments=semantic_promo_criteria,
                     record_action=False
                 )

                 if isinstance(query_result, dict) and query_result.get("success"):
                     memories_found = query_result.get("data", {}).get("memories", [])
                     # If type filtering happened in DB, this is fine.
                     # If not, add filtering here:
                     # promo_candidates_sem_to_proc = [
                     #    mem.get("memory_id") for mem in memories_found
                     #    if mem.get("memory_id") and mem.get("memory_type") in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]
                     # ]
                     promo_candidates_sem_to_proc = [mem.get("memory_id") for mem in memories_found if mem.get("memory_id")]
                     if promo_candidates_sem_to_proc:
                          self.logger.info(f"Found {len(promo_candidates_sem_to_proc)} Semantic candidates (procedure/skill) for promotion check.")

                 elif isinstance(query_result, dict):
                      self.logger.warning(f"Query for Semantic promotion candidates failed: {query_result.get('error')}")
                 else:
                      self.logger.warning(f"Unexpected result querying Semantic promotion candidates: {query_result}")

            except Exception as e:
                 self.logger.warning(f"Error querying for Semantic promotion candidates: {e}", exc_info=False)


            # --- Step 3: Add Promotion Tasks for Candidates ---
            # Combine candidates, avoiding duplicates if a memory somehow qualified for both
            all_candidates = list(set(promo_candidates_ep_to_sem + promo_candidates_sem_to_proc))

            if all_candidates:
                self.logger.info(f"Queueing promotion checks for {len(all_candidates)} candidate memories.")
                for mem_id in all_candidates:
                    # Add the task to potentially run `promote_memory_level`
                    # The promote tool itself contains the logic to check thresholds again before actually promoting.
                    tasks_to_run.append((promote_tool_name, {"memory_id": mem_id}))
                    # We could pass the target_level explicitly if needed, but letting the tool decide is fine.
                    # tasks_to_run.append((promote_tool_name, {"memory_id": mem_id, "target_level": "semantic"})) # Example explicit target

        if tasks_to_run:
            self.logger.info(f"Running {len(tasks_to_run)} periodic maintenance/meta-cognition tasks...", emoji_key="brain")
            # Execute tasks sequentially for simplicity, could use asyncio.gather
            for tool_name, args in tasks_to_run:
                 if self._shutdown_event.is_set(): break # Check shutdown flag
                 try:
                     self.logger.debug(f"Executing periodic task: {tool_name}")
                     # Execute without recording separate actions
                     result_content = await self._execute_tool_call_internal(tool_name, args, record_action=False)
                     # (#4) Feed back results? Check if reflection/consolidation content exists
                     if tool_name in [TOOL_REFLECTION, TOOL_CONSOLIDATION] and isinstance(result_content, dict) and result_content.get('success'):
                          content_key = "reflection_content" if tool_name == TOOL_REFLECTION else "consolidated_content"
                          feedback_content = result_content.get(content_key, "")
                          if feedback_content:
                               # Add this result summary to the next context?
                               # For simplicity, just log it now. A better system might store it.
                               self.logger.info(f"Feedback from {tool_name}: {feedback_content[:150]}...")
                               # Potentially set self.needs_replan = True after reflection?
                 except Exception as e:
                     self.logger.warning(f"Periodic task {tool_name} failed: {e}", exc_info=False)
                     await asyncio.sleep(0.1) # Small delay after failure


    # --- Main Run Method ---
    async def run(self, goal: str, max_loops: int = 50):
        """Main execution loop for the agent."""
        if not await self.initialize():
            self.logger.critical("Agent initialization failed. Aborting run.")
            return

        self.logger.info(f"Starting main loop. Goal: '{goal}' Max Loops: {max_loops}", emoji_key="arrow_forward")
        # Loop counter is loaded from state, don't reset here unless intended
        # self.current_loop = 0
        self.goal_achieved_flag = self.goal_achieved_flag # Loaded from state

        while not self.goal_achieved_flag and self.current_loop < max_loops:
             if self._shutdown_event.is_set():
                 self.logger.info("Shutdown signal received, exiting loop.")
                 break
             self.current_loop += 1
             self.logger.info(f"--- Agent Loop {self.current_loop}/{max_loops} ---", emoji_key="arrows_counterclockwise")

             # Error Check
             if self.consecutive_error_count >= self.max_consecutive_errors:
                 self.logger.error(f"Max consecutive errors ({self.max_consecutive_errors}) reached. Aborting.", emoji_key="stop_sign")
                 if self.workflow_id: await self._update_workflow_status_internal("failed", "Agent failed due to repeated errors.")
                 break

             # 1. Context
             context = await self._gather_context()
             if "error" in context and context.get("status") != "No Active Workflow":
                 self.logger.error(f"Context gathering failed: {context['error']}. Pausing and retrying.")
                 self.consecutive_error_count += 1
                 await asyncio.sleep(3) # Longer pause on context failure
                 continue

             # 2. Decide (using LLM)
             agent_decision = await self._call_agent_llm(goal, context)

             # 3. Act
             decision_type = agent_decision.get("decision")
             last_tool_result_content = None # Store result *content* dict

             if decision_type == "call_tool":
                 tool_name = agent_decision.get("tool_name")
                 arguments = agent_decision.get("arguments", {})
                 self.logger.info(f"Agent requests tool: {tool_name}", emoji_key="wrench")
                 # Use internal execute which handles action recording and error details
                 last_tool_result_content = await self._execute_tool_call_internal(
                     tool_name, arguments, record_action=True
                 )

             elif decision_type == "thought_process":
                 thought_content = agent_decision.get("content")
                 self.logger.info(f"Agent reasoning: '{thought_content[:100]}...'. Recording thought.", emoji_key="thought_balloon")
                 if self.workflow_id:
                      try:
                          # Record thought as an action
                          await self._execute_tool_call_internal(
                              TOOL_RECORD_THOUGHT,
                              {"workflow_id": self.workflow_id, "content": thought_content, "thought_type": "inference"},
                              record_action=True # Record this reasoning step
                          )
                          self.last_action_summary = f"Recorded thought: {thought_content[:100]}..."
                          self.consecutive_error_count = 0 # Reset error count
                      except Exception as e:
                           self.logger.error(f"Failed to record thought via tool: {e}")
                           self.last_action_summary = f"Error recording thought: {str(e)[:100]}"
                           self.consecutive_error_count += 1
                 else:
                      self.logger.warning("Cannot record thought: No active workflow ID.")
                      self.last_action_summary = "Agent provided reasoning, but no workflow active to record it."

             elif decision_type == "complete":
                 summary = agent_decision.get("summary", "Goal achieved.")
                 self.logger.info(f"Agent signals completion: {summary}", emoji_key="tada")
                 self.goal_achieved_flag = True
                 if self.workflow_id: await self._update_workflow_status_internal("completed", summary)
                 break # Exit loop

             elif decision_type == "error":
                 error_msg = agent_decision.get("message", "Unknown agent error")
                 self.logger.error(f"Agent decision error: {error_msg}", emoji_key="x")
                 self.last_action_summary = f"Agent decision error: {error_msg[:100]}"
                 self.last_error_details = {"agent_decision_error": error_msg} # (#8)
                 self.consecutive_error_count += 1
                 if self.workflow_id: # Log agent error as critique thought
                      try: await self._execute_tool_call_internal(TOOL_RECORD_THOUGHT, {"workflow_id": self.workflow_id, "content": f"Agent Decision Error: {error_msg}", "thought_type": "critique"}, record_action=False)
                      except Exception: pass

             else: # Unknown decision
                 self.logger.warning(f"Unhandled agent decision type: {decision_type}")
                 self.last_action_summary = "Received unknown decision from agent."
                 self.consecutive_error_count += 1
                 self.last_error_details = {"agent_decision_error": f"Unknown type: {decision_type}"} # (#8)

             # 4. Update Plan (using LLM's plan if provided in decision)
             await self._update_plan(context, agent_decision, last_tool_result_content)

             # 5. Periodic Tasks
             await self._run_periodic_tasks()

             # 6. Save Agent State Periodically (#15)
             if self.current_loop % 5 == 0: # Save every 5 loops
                 await self._save_agent_state()

             # 7. Loop Delay
             await asyncio.sleep(random.uniform(0.5, 1.2)) # Shorter random delay


        # --- End of Loop ---
        self.logger.info("--- Agent Loop Finished ---", emoji_key="stopwatch")
        if self.goal_achieved_flag: self.logger.info("Goal Status: Achieved", emoji_key="party_popper")
        elif self.current_loop >= max_loops: self.logger.warning(f"Goal Status: Max loops ({max_loops}) reached.", emoji_key="timer_clock")
        else: self.logger.warning("Goal Status: Loop exited prematurely.", emoji_key="warning")

        # Final state save
        await self._save_agent_state()

        # Final Report
        if self.workflow_id: await self._generate_final_report()

    # --- Internal Helpers ---
    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """Checks if a workflow ID exists using list_workflows tool."""
        tool_name = "unified_memory:list_workflows"
        server = self._find_tool_server(tool_name)
        if not server: return False
        try:
             result = await self._execute_tool_call_internal(tool_name, {"limit": 1, "workflow_id_filter": workflow_id}, record_action=False) # Assuming list_workflows accepts a filter
             # Adjust check based on actual return format of list_workflows
             return isinstance(result, dict) and result.get("success") and len(result.get("workflows", [])) > 0
        except Exception:
             return False

    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Finds an active server providing the specified tool."""
        if tool_name in self.mcp_client.server_manager.tools:
             server_name = self.mcp_client.server_manager.tools[tool_name].server_name
             if server_name in self.mcp_client.server_manager.active_sessions:
                 return server_name
        # self.logger.warning(f"Tool '{tool_name}' not found on any active server.")
        return None # Return None if not found or server inactive

    async def _update_workflow_status_internal(self, status: str, message: Optional[str]):
        """Internal helper to update workflow status via tool call."""
        if not self.workflow_id: return
        tool_name = TOOL_UPDATE_WORKFLOW_STATUS
        try:
             await self._execute_tool_call_internal(
                  tool_name,
                  {"workflow_id": self.workflow_id, "status": status, "completion_message": message},
                  record_action=False # Status updates aren't primary actions
             )
        except Exception as e:
             self.logger.error(f"Error marking workflow {self.workflow_id} as {status}: {e}", exc_info=False)


    async def _generate_final_report(self):
        """Generates and logs a final report using the memory tool."""
        if not self.workflow_id: return
        self.logger.info(f"Generating final report for workflow {self.workflow_id}...", emoji_key="scroll")
        tool_name = "unified_memory:generate_workflow_report"
        try:
            report_result_content = await self._execute_tool_call_internal(
                 tool_name,
                 {"workflow_id": self.workflow_id, "report_format": "markdown", "style": "professional"},
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

    async def signal_shutdown(self):
        """Signals the main loop to stop gracefully."""
        self.logger.info("Shutdown signal received.")
        self._shutdown_event.set()

    async def shutdown(self):
        """Cleans up agent loop resources."""
        self.logger.info("Shutting down agent loop...", emoji_key="power_button")
        # Signal loop to stop if running
        self._shutdown_event.set()
        # Save final state
        await self._save_agent_state()
        # MCP Client closure is handled externally by the script that created it
        self.logger.info("Agent loop shutdown complete.", emoji_key="wave")


# --- Main Execution Block (Using Agent Loop Class) ---
async def run_agent_process(mcp_server_url: str, anthropic_key: str, goal: str, max_loops: int, state_file: str):
    """Sets up the MCP Client and runs the Agent Master Loop."""
    mcp_client_instance = None
    agent_loop_instance = None
    exit_code = 0

    # Use safe_print if available early, otherwise standard print
    printer = print
    if MCPClient and hasattr(MCPClient, 'safe_print'):
         printer = MCPClient.safe_print

    try:
        if not MCP_CLIENT_AVAILABLE: raise RuntimeError("MCPClient class not found.")
        printer("Instantiating MCP Client...")
        # Pass API key directly if provided, client constructor should handle precedence
        mcp_client_instance = MCPClient()
        # Ensure API key is set correctly after config load
        if not mcp_client_instance.config.api_key:
            if anthropic_key:
                 printer("Using provided Anthropic API key.")
                 mcp_client_instance.config.api_key = anthropic_key
                 mcp_client_instance.anthropic = AsyncAnthropic(api_key=anthropic_key)
            else:
                 raise ValueError("Anthropic API key missing and not provided.")

        printer("Setting up MCP Client (connecting to servers)...")
        await mcp_client_instance.setup(interactive_mode=False)

        printer("Instantiating Agent Master Loop...")
        agent_loop_instance = AgentMasterLoop(
            mcp_client_instance=mcp_client_instance,
            agent_state_file=state_file
        )

        printer("Running Agent Loop...")
        await agent_loop_instance.run(goal=goal, max_loops=max_loops)

    except KeyboardInterrupt:
        printer("\n[yellow]Agent loop interrupted by user.[/yellow]")
        if agent_loop_instance: await agent_loop_instance.signal_shutdown() # Signal graceful stop
        exit_code = 130 # Standard exit code for Ctrl+C
    except Exception as main_err:
        printer(f"\n❌ Critical error during agent execution: {main_err}")
        log.critical("Top-level execution error", exc_info=True) # Log full traceback
        exit_code = 1
    finally:
        if agent_loop_instance:
            printer("Shutting down agent loop...")
            await agent_loop_instance.shutdown() # Saves state
        if mcp_client_instance:
            printer("Closing MCP client...")
            await mcp_client_instance.close() # Closes connections, etc.
        printer("Agent execution finished.")
        # Exit with appropriate code only if __main__
        if __name__ == "__main__":
            sys.exit(exit_code)


# --- Signal Handling for Graceful Shutdown ---
def _handle_signal(signum, frame, loop_instance: Optional[AgentMasterLoop]):
    """Signal handler to trigger graceful shutdown."""
    signal_name = signal.Signals(signum).name
    log.warning(f"Received signal {signal_name} ({signum}). Initiating graceful shutdown...")
    if loop_instance:
        # Signal the running loop to stop. Don't await here.
        asyncio.create_task(loop_instance.signal_shutdown())
    else:
        # If loop isn't running yet, just exit
        log.warning("Loop instance not available, exiting immediately.")
        sys.exit(1)

if __name__ == "__main__":
    # --- Configuration from Environment ---
    MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8013") # Memory system URL
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    AGENT_GOAL = os.environ.get("AGENT_GOAL", "Create a simple workflow for 'Test Goal'. Add a planning thought. Query memories containing 'test'. If results found, record insight. Mark workflow complete.")
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "20")) # Increased default
    AGENT_STATE_FILENAME = os.environ.get("AGENT_STATE_FILE", "agent_loop_state.json")
    if os.environ.get("AGENT_LOOP_VERBOSE") == "1": log.setLevel(logging.DEBUG)

    if not ANTHROPIC_API_KEY:
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)
    if not MCP_CLIENT_AVAILABLE:
         print("❌ ERROR: MCPClient dependency not met. Cannot run.")
         sys.exit(1)

    print("--- Supercharged Agent Master Loop Configuration ---")
    print(f"Unified Memory Server URL: {MCP_SERVER_URL}")
    print(f"Agent Goal: {AGENT_GOAL}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Agent State File: {AGENT_STATE_FILENAME}")
    print(f"Log Level: {logging.getLevelName(log.level)}")
    print("Anthropic API Key: Found")
    print("--------------------------------------------------")

    # --- Setup Signal Handlers & Run ---
    loop = asyncio.get_event_loop()
    # Pass the loop instance later once created
    loop.add_signal_handler(signal.SIGINT, lambda: _handle_signal(signal.SIGINT, None, agent_loop_instance_ref.get()))
    loop.add_signal_handler(signal.SIGTERM, lambda: _handle_signal(signal.SIGTERM, None, agent_loop_instance_ref.get()))

    # Use a simple container to pass the loop instance to the signal handler
    agent_loop_instance_ref = {}
    def update_ref(loop_instance):
         agent_loop_instance_ref.clear()
         agent_loop_instance_ref[0] = loop_instance

    # Run the main process, capture the loop instance reference
    main_task = loop.create_task(run_agent_process(MCP_SERVER_URL, ANTHROPIC_API_KEY, AGENT_GOAL, MAX_ITERATIONS, AGENT_STATE_FILENAME))

    # Need to get the loop instance *after* it's created inside run_agent_process
    # This is tricky. A better way might be to pass a queue or event.
    # For simplicity, we rely on the global `agent_loop_instance` potentially being set
    # (if run_agent_process were modified to set a global ref).
    # The current signal handler approach relies on the loop instance being set externally.
    # A cleaner pattern: run_agent_process should *return* the loop instance if needed.
    # For now, the signal handler might not have the loop instance ref immediately.

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
         log.info("Main task cancelled.")
    finally:
         loop.close()
         log.info("Event loop closed.")
