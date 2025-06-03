from __future__ import annotations

"""Robust Agent Loop implementation using MCPClient infrastructure."""

###############################################################################
# SECTION 0. Imports & typing helpers
###############################################################################

import asyncio
import dataclasses
import datetime as _dt
import enum
import json
import logging
import math
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Set, Tuple

###############################################################################
# SECTION 1. Agent configuration constants
###############################################################################

# These constants provide default values for agent behavior when MCPClient config is unavailable.

# Metacognition & reflection timing
REFLECTION_INTERVAL = 15  # generate reflection memories every N turns
STALL_THRESHOLD = 3  # consecutive non‑progress turns → forced reflection

# Default model names (fallback only - use mcp_client.config.default_model instead)
DEFAULT_FAST_MODEL = "gpt-4.1-nano"

###############################################################################
# SECTION 2. Enumerations & simple data classes
###############################################################################


class LinkKind(str, enum.Enum):
    """Canonical, machine-friendly edge types the UMS already recognises."""
    RELATED = "related"
    CAUSAL = "causal"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    SEQUENTIAL = "sequential"
    REFERENCES = "references"
    CONSEQUENCE_OF = "consequence_of"

class Phase(str, enum.Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    REVIEW = "review"
    COMPLETE = "complete"




@dataclasses.dataclass
class AMLState:
    workflow_id: Optional[str] = None
    root_goal_id: Optional[str] = None
    current_leaf_goal_id: Optional[str] = None
    phase: Phase = Phase.UNDERSTAND
    loop_count: int = 0
    cost_usd: float = 0.0
    stuck_counter: int = 0
    last_reflection_turn: int = 0
    
    # New fields for enhanced agent state management
    current_plan: Optional[List[Dict[str, Any]]] = dataclasses.field(default_factory=list)
    goal_stack: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    current_thought_chain_id: Optional[str] = None
    last_action_summary: Optional[str] = None
    last_error_details: Optional[Dict[str, Any]] = None
    consecutive_error_count: int = 0
    needs_replan: bool = True
    goal_achieved_flag: bool = False
    context_id: Optional[str] = None

UMS_SERVER_NAME = "Ultimate MCP Server"

###############################################################################
# SECTION 3. Graph / Memory management helpers
###############################################################################

# ---------------------------------------------------------------------------
# Graph-level buffered writer  ✧  REMOVED (no longer needed)
# ---------------------------------------------------------------------------

# The GraphWriteBuffer class has been removed. All UMS operations are now
# handled directly through MCPClient tool calls which have their own optimizations.

class MemoryGraphManager:
    """Rich graph operations on the Unified Memory Store (UMS).

    *Only this class is fully implemented in this revision; the rest of the file
    remains an outline so that downstream work can continue incrementally.*

    The implementation speaks *directly* to the UMS server via MCPClient tool calls.
    All public APIs are **async‑safe** and execute via the MCP infrastructure.
    """

    #############################################################
    # Construction / low‑level helpers
    #############################################################

    def __init__(self, mcp_client, state: AMLState):
        self.mcp_client = mcp_client
        self.state = state
        # UMS server name - matches what's used in mcp_client_multi.py
        self.ums_server_name = UMS_SERVER_NAME
        self.agent = None  # Will be set by AgentMasterLoop

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Construct the full MCP tool name for UMS tools."""
        return f"{self.ums_server_name}:{base_tool_name}"

    # ----------------------------------------------------------- public API

    async def auto_link(
        self,
        src_id: str,
        tgt_id: str,
        context_snip: str = "",
        *,                           # <- forces keyword for the override
        kind_hint: LinkKind | None = None,
    ) -> None:
        """
        Create (or upsert) a semantic link between two memories.

        Parameters
        ----------
        src_id, tgt_id : str
            Memory IDs.
        context_snip   : str
            Short free-text description stored in `description`.
        kind_hint      : LinkKind | None, optional
            If provided we *trust* the caller's classification instead of running
            the `_infer_link_type` heuristic.  Keeps the cheap-LLM budget down when
            the semantics are already known (e.g. reasoning-trace construction).
        """
        link_type = (kind_hint or LinkKind(
            await self._infer_link_type(src_id, tgt_id, context_snip)
        )).value

        await self.mcp_client._execute_tool_and_parse_for_agent(
            self.ums_server_name,
            self._get_ums_tool_name("create_memory_link"),
            {
                "workflow_id": self.state.workflow_id,
                "source_memory_id": src_id,
                "target_memory_id": tgt_id,
                "link_type": link_type,
                "strength": 1.0,
                "description": context_snip[:180],
                "extra_json": json.dumps({"loop": self.state.loop_count,
                                          "phase": self.state.phase.value})[:300],
            },
        )

    async def register_reasoning_trace(
        self,
        thought_mem_id: str,
        evidence_ids: list[str] | None = None,
        derived_fact_ids: list[str] | None = None,
    ) -> None:
        """
        Capture the *why-tree* of a reasoning step:

            • THOUGHT         (REASONING_STEP memory)
                ╠═ evidence  → SUPPORTS
                ╚═ produces  → CONSEQUENCE_OF

        This becomes machine-query-able («show me all evidence contradicting
        THOUGHT X»).
        """
        evidence_ids = evidence_ids or []
        derived_fact_ids = derived_fact_ids or []
        for ev in evidence_ids:
            await self.auto_link(ev, thought_mem_id, "thought uses evidence", kind_hint=LinkKind.SUPPORTS)
        for fact in derived_fact_ids:
            await self.auto_link(thought_mem_id, fact, "thought leads to fact", kind_hint=LinkKind.CONSEQUENCE_OF)


            
    async def detect_inconsistencies(self) -> List[Tuple[str, str]]:
        """
        Detect contradictions using UMS get_contradictions tool.
        Returns list of (memory_id_a, memory_id_b) tuples representing contradictory pairs.
        """
        try:
            contradictions_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_contradictions"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 50,
                    "include_resolved": False  # Exclude already resolved contradictions
                }
            )
            
            # Note: MemoryGraphManager doesn't have direct access to the helper method,
            # but we can add a simple unwrapping here since it follows the same pattern
            if not contradictions_envelope.get("success"):
                self.mcp_client.logger.warning(f"MCPClient: UMS get_contradictions failed: {contradictions_envelope.get('error_message', 'Unknown error')}")
                return []
                
            ums_tool_envelope = contradictions_envelope.get("data", {})
            if not isinstance(ums_tool_envelope, dict) or not ums_tool_envelope.get("success"):
                ums_error = ums_tool_envelope.get("error_message", "UMS tool reported failure")
                self.mcp_client.logger.warning(f"UMS get_contradictions tool failed: {ums_error}")
                return []
                
            contradictions_data = ums_tool_envelope.get("data", {})
            contradictions_found = contradictions_data.get("contradictions_found", [])
            
            # Convert to the expected format: List[Tuple[str, str]]
            pairs = []
            for contradiction in contradictions_found:
                mem_a = contradiction.get("memory_id_a")
                mem_b = contradiction.get("memory_id_b")
                if mem_a and mem_b:
                    pairs.append((mem_a, mem_b))
            
            self.mcp_client.logger.debug(f"Found {len(pairs)} contradictions via UMS")
            return pairs
                
        except Exception as e:
            self.mcp_client.logger.warning(f"Error detecting contradictions via UMS: {e}")
            return []



    async def promote_hot_memories(self, importance_cutoff: float = 7.5, access_cutoff: int = 5) -> None:
        """Promote memories worth keeping long‑term from working → semantic."""
        try:
            # Query for promotion candidates via UMS
            candidates_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("query_memories"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": "working",
                    "min_importance": importance_cutoff,
                    "limit": 100
                }
            )
            
            if not candidates_res.get("success"):
                self.mcp_client.logger.warning(f"MCPClient: UMS query_memories failed: {candidates_res.get('error_message', 'Unknown error')}")
                return
                
            ums_tool_envelope = candidates_res.get("data", {})
            if not isinstance(ums_tool_envelope, dict) or not ums_tool_envelope.get("success"):
                ums_error = ums_tool_envelope.get("error_message", "UMS tool reported failure")
                self.mcp_client.logger.warning(f"UMS query_memories tool failed: {ums_error}")
                return
                
            candidates_data = ums_tool_envelope.get("data", {})
            candidates = candidates_data.get("memories", [])
            
            # Promote memories that meet criteria
            for memory in candidates:
                importance = memory.get("importance", 0.0)
                access_count = memory.get("access_count", 0)
                
                if importance >= importance_cutoff or access_count >= access_cutoff:
                    await self.mcp_client._execute_tool_and_parse_for_agent(
                        self.ums_server_name,
                        self._get_ums_tool_name("update_memory"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "memory_id": memory["memory_id"],
                            "memory_level": "semantic",
                        }
                    )
                    
        except Exception as e:
            self.mcp_client.logger.warning(f"Error during memory promotion: {e}")



    #############################################################
    # Internal helpers
    #############################################################

    async def _infer_link_type(self, src_id: str, tgt_id: str, context: str) -> str:
        """Fast heuristic and cheap‑LLM back‑off for deciding link type."""
        
        # ------------------------------------------------------------------
        # 0) **CACHE SHORT-CIRCUIT** – if both memories already carry a
        #    reciprocal cached type we trust that and return it instantly.
        # ------------------------------------------------------------------
        try:
            meta_src_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_memory_metadata"),
                {"workflow_id": self.state.workflow_id, "memory_id": src_id}
            )
            try:
                meta_src_data = await self.agent._get_valid_ums_data_payload(meta_src_res, "get_memory_metadata_src")
                meta_src = meta_src_data.get("metadata", {}).get("link_type_cache", {})
                if tgt_id in meta_src:
                    return meta_src[tgt_id]
            except RuntimeError:
                pass  # Fall through to tgt check
                    
            meta_tgt_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_memory_metadata"),
                {"workflow_id": self.state.workflow_id, "memory_id": tgt_id}
            )
            try:
                meta_tgt_data = await self.agent._get_valid_ums_data_payload(meta_tgt_res, "get_memory_metadata_tgt")
                meta_tgt = meta_tgt_data.get("metadata", {}).get("link_type_cache", {})
                if src_id in meta_tgt:
                    return meta_tgt[src_id]
            except RuntimeError:
                pass  # Fall through to normal path
        except Exception:
            # Any metadata hiccup → fall back to normal path
            pass

        # 1. Heuristic quick rules -------------------------------------------
        ctx_lower = context.lower()
        causal_cues = ("because", "therefore", "so that", "as a result", "leads to")
        if any(c in ctx_lower for c in (" not ", " no ", " contradict")):
            return "CONTRADICTS"
        if any(c in ctx_lower for c in causal_cues):
            return "CAUSAL"
            
        # 2. Embedding cosine shortcut  (saves ~40 % cheap-LLM calls)
        src_vec = await self._get_cached_embedding(src_id)
        tgt_vec = await self._get_cached_embedding(tgt_id)
        if src_vec is not None and tgt_vec is not None:
            sim = await self._cosine_similarity(src_vec, tgt_vec)
            if sim is not None and sim >= 0.80:
                return "RELATED"

        # 3. Check tag overlap -------------------------------------------------
        tags_src = await self._get_tags_for_memory(src_id)
        tags_tgt = await self._get_tags_for_memory(tgt_id)
        if tags_src & tags_tgt:
            return "RELATED"
            
        # 4. Fallback to cheap LLM structured call ----------------------------
        schema = {"type": "object", "properties": {"link_type": {"type": "string"}}, "required": ["link_type"]}
        prompt = (
            "You are an expert knowledge‑graph assistant. Given two text snippets, "
            "classify the best relationship type among: RELATED, CAUSAL, SEQUENTIAL, "
            "CONTRADICTS, SUPPORTS, GENERALIZES, SPECIALIZES.\n"  # limited to those the DB knows
            f"SRC: {(await self._get_memory_content(src_id))[:300]}\n"  # limit snippet
            f"TGT: {(await self._get_memory_content(tgt_id))[:300]}\n"
            'Answer with JSON: {"link_type": '
            "<TYPE>"
            "}"
        )
        try:
            # Use MCPClient's LLM infrastructure for cheap/fast calls
            resp = await self.mcp_client.query_llm_structured(prompt, schema, use_cheap_model=True)
            t = resp.get("link_type", "RELATED").upper()
            inferred = t if t in {"RELATED", "CAUSAL", "SEQUENTIAL", "CONTRADICTS", "SUPPORTS", "GENERALIZES", "SPECIALIZES"} else "RELATED"
        except Exception:
            inferred = "RELATED"

        # ------------------------------------------------------------------
        # 5)  **CACHE RESULT** for both memories so future calls are O(1)
        # ------------------------------------------------------------------
        try:
            await self._update_link_type_cache(src_id, tgt_id, inferred)
        except Exception:
            pass
        return inferred

    # ------------------------ misc small helpers ---------------------------

    async def _get_memory_content(self, memory_id: str) -> str:
        try:
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_memory_by_id"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_id": memory_id
                }
            )
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(res, "get_memory_by_id")
                content = ums_data.get("content", "")
                # TODO: Update access count via UMS tool if available
                return content
            except RuntimeError:
                return ""
        except Exception:
            return ""

    async def _get_tags_for_memory(self, memory_id: str) -> Set[str]:
        try:
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_memory_tags"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_id": memory_id
                }
            )
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(res, "get_memory_tags")
                tags = ums_data.get("tags", [])
                return set(tags)
            except RuntimeError:
                return set()
        except Exception:
            return set()

    # ----------------------- embedding helpers ------------------------------

    async def _get_cached_embedding(self, memory_id: str) -> Optional[list[float]]:
        """
        Get embedding vector for a memory using UMS tools.
        """
        try:
            # Try to get embedding from UMS
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_embedding"),
                {"workflow_id": self.state.workflow_id, "memory_id": memory_id}
            )
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(res, "get_embedding")
                return ums_data.get("vector")
            except RuntimeError:
                pass  # Fall through to create embedding
            
            # If no embedding exists, try to create one
            create_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("create_embedding"),
                {"workflow_id": self.state.workflow_id, "memory_id": memory_id}
            )
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(create_res, "create_embedding")
                return ums_data.get("vector")
            except RuntimeError:
                return None
                
        except Exception:
            return None

    async def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> Optional[float]:
        """Calculate cosine similarity, with fallback to local computation."""
        try:
            # Try server-side computation first
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("vector_similarity"),
                {"vec_a": vec_a, "vec_b": vec_b}
            )
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(res, "vector_similarity")
                return ums_data.get("cosine")
            except RuntimeError:
                pass  # Fall through to local computation
        except Exception:
            pass
            
        # Fallback to local computation
        try:
            dot = sum(x * y for x, y in zip(vec_a, vec_b, strict=False))
            na = math.sqrt(sum(x * x for x in vec_a))
            nb = math.sqrt(sum(y * y for y in vec_b))
            return dot / (na * nb + 1e-9)
        except Exception:
            return None

    # ----------------------- link-strength decay ----------------------------

    async def decay_link_strengths(self, half_life_days: int = 14) -> None:
        """Halve link strength for edges older than *half_life_days*."""
        try:
            # Use UMS tool if available, otherwise skip
            await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("decay_link_strengths"),
                {
                    "workflow_id": self.state.workflow_id,
                    "half_life_days": half_life_days
                }
            )
        except Exception:
            # If decay tool is not available, skip silently
            pass

    # ----------------------- metadata helpers -----------------------------

    async def _get_metadata(self, memory_id: str) -> Dict[str, Any]:
        """Fetch metadata JSON for *memory_id* (empty dict if none)."""
        try:
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_memory_metadata"),
                {"workflow_id": self.state.workflow_id,
                 "memory_id": memory_id}
            )
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(res, "get_memory_metadata")
                return ums_data.get("metadata", {}) or {}
            except RuntimeError:
                return {}
        except Exception:
            return {}

    async def _update_link_type_cache(self, src_id: str, tgt_id: str, link_type: str) -> None:
        """Persist reciprocal cache entries `src→tgt` and `tgt→src`."""
        for a, b in ((src_id, tgt_id), (tgt_id, src_id)):
            try:
                meta = await self._get_metadata(a)
                cache = meta.get("link_type_cache", {})
                cache[b] = link_type
                meta["link_type_cache"] = cache
                    
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    self.ums_server_name,
                    self._get_ums_tool_name("update_memory_metadata"),
                    {"workflow_id": self.state.workflow_id,
                    "memory_id": a,
                    "metadata": meta}
                )
            except Exception:
                continue

    # --- tiny convenience -----------------------------------------------
    async def mark_contradiction_resolved(self, mem_a: str, mem_b: str) -> None:
        """
        Tag the CONTRADICTS edge A↔B as resolved so Metacognition skips it later.
        """
        meta_flag = {"resolved_at": int(time.time())}
        for a, b in ((mem_a, mem_b), (mem_b, mem_a)):
            try:
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    self.ums_server_name, 
                    self._get_ums_tool_name("update_memory_link_metadata"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "source_memory_id": a,
                        "target_memory_id": b,
                        "link_type": "CONTRADICTS",
                        "metadata": meta_flag,
                    }
                )
            except Exception:
                pass

class AsyncTask:
    """Represents a *single* asynchronous micro-call delegated to the fast LLM.

    The task life-cycle is fully tracked so the orchestrator can make informed
    scheduling decisions (e.g. cost accounting, detecting starved tasks).
    """

    __slots__ = ("name", "coro", "callback", "_task")

    def __init__(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        callback: Optional[Callable[[Any], Awaitable[None] | None]] = None,
    ) -> None:
        self.name: str = name
        self.coro: Coroutine[Any, Any, Any] = coro
        self.callback = callback

        self._task: Optional[asyncio.Task[Any]] = None

    # ------------------------------------------------------------------ api

    def start(self) -> None:
        """Schedule the coroutine on the current loop and attach bookkeeping."""
        if self._task is not None:
            # Idempotent: start() can be called only once.
            return
        self._task = asyncio.create_task(self.coro, name=self.name)
        self._task.add_done_callback(self._on_done)  # type: ignore[arg-type]

    # Properties -------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()



    # Internal ---------------------------------------------------------------

    def _on_done(self, task: asyncio.Task[Any]) -> None:  # pragma: no cover
        if self.callback is not None:
            # Get result for callback, but don't store it
            try:
                result = task.result()
            except BaseException:  # noqa: BLE001 – capture any exception
                result = None  # Pass None to callback on exception
            
            # Fire user callback
            if asyncio.iscoroutinefunction(self.callback):  # type: ignore[arg-type]
                asyncio.create_task(self.callback(result))
            else:
                try:
                    self.callback(result)  # type: ignore[misc]
                except Exception:  # noqa: BLE001 – user callbacks shouldn't crash loop
                    pass


###############################################################################


class AsyncTaskQueue:
    """Light-weight FIFO scheduler for `AsyncTask` objects.

    * Concurrency-bounded: you can limit how many tasks run simultaneously.
    * Guarantees that `spawn()` never blocks; tasks are either started
      immediately or queued.
    * `drain()` returns only when **all** tasks (running *and* queued) have
      completed, making it perfect to call right before the expensive smart-LLM
      turn to ensure cheap tasks have flushed their results into memory.
    """

    def __init__(self, max_concurrency: int = 8) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._concurrency = max_concurrency
        self._running: Set[AsyncTask] = set()
        self._pending: asyncio.Queue[AsyncTask] = asyncio.Queue()
        self._completed: List[AsyncTask] = []
        self._loop = asyncio.get_event_loop()

    # ------------------------------------------------------------------ API

    def spawn(self, task: AsyncTask) -> None:
        """Submit a task for execution respecting the concurrency limit."""
        if task.done:
            # Don't queue completed tasks.
            self._completed.append(task)
            return
        task._task = None  # ensure not started yet
        self._loop.call_soon(self._maybe_start, task)

    async def drain(self) -> None:
        """Wait until *all* tasks (running & queued) finish."""
        # Continuously wait for running tasks to finish; when running is empty
        # check if queue is empty too.
        while self._running or not self._pending.empty():
            # Wait for *any* running task to finish.
            if self._running:
                done_tasks, _ = await asyncio.wait(
                    {t._task for t in self._running if t._task is not None},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Mark them done so _maybe_start can fire queued tasks.
                for d in done_tasks:
                    # Find corresponding AsyncTask instance.
                    for at in list(self._running):
                        if at._task is d:
                            self._running.remove(at)
                            self._completed.append(at)
                            break
            # Start queued tasks if we have slots.
            await self._fill_slots()



    def cancel_all(self) -> None:
        """Best-effort cancellation of running and pending tasks."""
        # Cancel running tasks ------------------------------------------------
        for task in list(self._running):
            if task._task is not None and not task._task.done():
                task._task.cancel()
        self._running.clear()
        # Clear pending queue -------------------------------------------------
        while not self._pending.empty():
            try:
                self._pending.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ----------------------------------------------------------------- stats

    @property
    def running(self) -> int:
        return len(self._running)



    # ------------------------------------------------------------ internal

    def _maybe_start(self, task: AsyncTask) -> None:
        if self.running < self._concurrency:
            self._start_task(task)
        else:
            self._pending.put_nowait(task)

    def _start_task(self, task: AsyncTask) -> None:
        task.start()
        self._running.add(task)
        # When this asyncio.Task completes we need to release slot.
        task._task.add_done_callback(lambda _: self._loop.call_soon(self._task_finished, task))  # type: ignore[arg-type]

    def _task_finished(self, task: AsyncTask) -> None:
        # Remove from running, append to completed.
        self._running.discard(task)
        self._completed.append(task)
        # Kick the queue to start next task if available.
        self._loop.call_soon_threadsafe(self._loop.create_task, self._fill_slots())

    async def _fill_slots(self) -> None:
        """Consume the pending queue until concurrency limit is reached."""
        while self.running < self._concurrency and not self._pending.empty():
            task = await self._pending.get()
            self._start_task(task)


###############################################################################
# SECTION 5. Metacognition Engine (NEW – fully implemented)
###############################################################################


class MetacognitionEngine:
    """Centralised self-monitoring, reflection, and phase-transition logic.

    Responsibilities
    ----------------
    • *Progress Assessment*  – decide whether the agent is making headway.
    • *Reflection Triggering* – periodically or when stuck, generate reflection
      thoughts and store them via UMS.
    • *Phase Transitions* – recommend when to move UNDERSTAND→PLAN→EXECUTE etc.
    • *Stuck Recovery* – orchestrate remedial actions (e.g., auto-reflection,
      cheap model critique, calling MemoryGraphManager.detect_inconsistencies).

    It relies on:
      * **MemoryGraphManager** for working-memory updates.
      * **LLMOrchestrator** for cheap/expensive model calls.
    """

    def __init__(
        self,
        mcp_client,
        state: AMLState,
        mem_graph: "MemoryGraphManager",
        llm_orch: "LLMOrchestrator",
        async_queue: "AsyncTaskQueue",
    ) -> None:
        self.mcp_client = mcp_client
        self.state = state
        self.mem_graph = mem_graph
        self.llm = llm_orch
        self.async_queue = async_queue
        self.logger = mcp_client.logger  # Add logger reference

    def set_agent(self, agent: "AgentMasterLoop") -> None:
        """Link the main agent for auto-linking capabilities."""
        self.agent = agent

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Construct the full MCP tool name for UMS tools."""
        return f"{UMS_SERVER_NAME}:{base_tool_name}"

    # ---------------------------------------------------------------- public

    async def maybe_reflect(self, turn_ctx: Dict[str, Any]) -> None:
        """Generate a reflection memory when cadence or stuckness criteria hit."""
        
        conditions = [
            self.state.loop_count - self.state.last_reflection_turn >= REFLECTION_INTERVAL,
            self.state.stuck_counter >= STALL_THRESHOLD,
        ]
        
        if not any(conditions):
            return

        # Use contradictions from context, or fetch fresh ones if needed
        contradictions = turn_ctx.get('contradictions', [])

        # Only fetch contradictions if not already provided by context
        if not contradictions:
            contradictions = await self.mem_graph.detect_inconsistencies()
            # Update context with the contradictions we found
            turn_ctx["contradictions"] = contradictions
            
        if contradictions:
            turn_ctx["has_contradictions"] = True
            
            # Try to actively resolve contradictions
            for pair in contradictions[:2]:  # Resolve up to 2 per turn to avoid overwhelming
                await self._resolve_contradiction(pair[0], pair[1])
            
            # Escalate persistent contradictions to BLOCKER goals
            await self._escalate_persistent_contradictions(contradictions)
                    
        # Determine reflection type based on context
        if turn_ctx.get("has_contradictions"):
            reflection_type = "strengths"  # Focus on what's working vs what conflicts
        elif self.state.stuck_counter >= STALL_THRESHOLD:
            reflection_type = "plan"  # Focus on next steps when stuck

        # Use UMS generate_reflection tool
        try:
            reflection_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "generate_reflection",
                {
                    "workflow_id": self.state.workflow_id,
                    "reflection_type": reflection_type,
                    "recent_ops_limit": 30,
                    "max_tokens": 800
                }
            )
            
            try:
                reflection_data = await self.agent._get_valid_ums_data_payload(reflection_envelope, "generate_reflection")
                reflection_id = reflection_data.get("reflection_id")
                self.mcp_client.logger.debug(f"Generated {reflection_type} reflection: {reflection_id}")
            except RuntimeError as e:
                self.mcp_client.logger.warning(f"Reflection generation failed: {e}")
                
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to generate reflection: {e}")

        self.state.last_reflection_turn = self.state.loop_count
        # reset stuck counter after reflection
        self.state.stuck_counter = 0

        # Optimize working memory periodically to keep it focused
        if self.state.loop_count % 5 == 0:  # Every 5 turns
            try:
                optimize_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    "optimize_working_memory",
                    {
                        "context_id": self.state.workflow_id,
                        "strategy": "balanced"
                    }
                )
                try:
                    await self.agent._get_valid_ums_data_payload(optimize_res, "optimize_working_memory")
                    self.logger.debug("Optimized working memory")
                except RuntimeError as e:
                    self.logger.debug(f"Working memory optimization failed: {e}")
            except Exception as e:
                self.logger.debug(f"Working memory optimization failed (non-critical): {e}")

        # Schedule lightweight graph upkeep (doesn't need smart model)
        if self.state.loop_count % 5 == 0:
            async def _maint():
                await self.mem_graph.decay_link_strengths()
                await self.mem_graph.promote_hot_memories()
            self.async_queue.spawn(AsyncTask("graph_maint", _maint()))

    async def assess_and_transition(self, progress_made: bool) -> None:
        """Update stuck counter, maybe switch phase."""
        if progress_made:
            self.state.stuck_counter = 0
        else:
            self.state.stuck_counter += 1

        # Automatic phase promotion logic ----------------------------------
        if self.state.phase == Phase.UNDERSTAND and progress_made:
            self.state.phase = Phase.PLAN
        elif self.state.phase == Phase.PLAN and progress_made:
            self.state.phase = Phase.EXECUTE
        # Graph maintenance now happens in background via maybe_reflect

        # Enter REVIEW if goal reached (placeholder detection)
        if await self._goal_completed():
            self.state.phase = Phase.REVIEW

    # ---------------------------------------------------------------- util


        
    async def _goal_completed(self) -> bool:
        """Check if the current goal is completed with proper evidence and artifact validation."""
        try:
            # Get goal details via UMS tool
            goal_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_goal_details"),
                {"workflow_id": self.state.workflow_id, "goal_id": self.state.current_leaf_goal_id}
            )
            
            try:
                goal_data_payload = await self.agent._get_valid_ums_data_payload(goal_envelope, "get_goal_details_completion_check")
                goal_data = goal_data_payload.get("goal", {})
                if not goal_data:
                    self.logger.warning(f"Goal completion check: no goal data in payload: {goal_data_payload}")
                    return False
            except RuntimeError as e:
                self.logger.warning(f"Goal completion check failed: {e}")
                return False
                
            status = goal_data.get("status", "")
            if status != "completed":
                return False
            
            # Use the enhanced validation method
            is_valid, reason = await self._validate_goal_outputs(self.state.current_leaf_goal_id)
            
            if not is_valid:
                self.logger.info(f"Goal completion validation failed: {reason}")
                
                # Store detailed validation failure
                try:
                    validation_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        "store_memory",
                        {
                            "workflow_id": self.state.workflow_id,
                            "content": f"Goal completion validation failed: {reason}. Goal: {goal_data.get('title', 'Unknown')}",
                            "memory_type": "validation_failure",
                            "memory_level": "working",
                            "importance": 7.0,
                            "description": f"Validation failure: {reason}"
                        }
                    )
                    await self.agent._get_valid_ums_data_payload(validation_res, "store_memory_validation_failure")
                except RuntimeError as e:
                    self.logger.warning(f"Failed to store validation failure memory: {e}")
                except Exception as e:
                    self.logger.warning(f"Error storing validation failure memory: {e}")
                
                return False
            
            # Additional edge-aware progress check: verify goal has evidence links
            links_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "get_linked_memories",
                {
                    "workflow_id": self.state.workflow_id, 
                    "memory_id": self.state.current_leaf_goal_id,
                    "direction": "outgoing"
                }
            )
            
            try:
                links_data = await self.agent._get_valid_ums_data_payload(links_envelope, "get_linked_memories_goal_completion")
                # Look for evidence that this goal contributed to parent goal
                evidence_links = [
                    link for link in links_data.get("links", [])
                    if link.get("link_type") in ["CONSEQUENCE_OF", "SUPPORTS"] 
                    and link.get("target_memory_id") != self.state.current_leaf_goal_id  # avoid self-loops
                ]
                
                if not evidence_links:
                    self.logger.info("Goal lacks evidence links - completion may be premature")
                    return False
            except RuntimeError as e:
                self.logger.warning(f"Could not verify goal evidence links: {e}")
                return False
            
            # NEW: If goal validation passes but status is not 'completed', update it.
            if is_valid and status != "completed":
                self.logger.info(f"Goal '{self.state.current_leaf_goal_id[:8]}' validation passed but status is '{status}'. Updating to 'completed'.")
                try:
                    update_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("update_goal_status"),
                        {
                            "goal_id": self.state.current_leaf_goal_id,
                            "status": "completed",
                            "reason": f"Goal validated by agent at loop {self.state.loop_count}"
                        }
                    )
                    await self.agent._get_valid_ums_data_payload(update_envelope, "update_goal_status")
                    return True  # It's valid AND we've updated its status
                except RuntimeError as e:
                    self.logger.warning(f"Failed to update goal status to completed: {e}")
                    return True  # Still valid even if status update failed
            
            self.logger.info(f"Goal completion validated successfully: {reason}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating goal completion: {e}")
            return False

    async def _escalate_persistent_contradictions(self, contradictions: List[Tuple[str, str]]) -> None:
        """Track contradiction pairs and escalate to BLOCKER goals when they persist ≥3 times."""
        # Simple persistent tracking using UMS metadata
        for pair in contradictions:
            pair_key = f"contradiction_{min(pair)}_{max(pair)}"  # normalized key
            try:
                # Try to get existing count
                meta_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    "get_workflow_metadata",
                    {"workflow_id": self.state.workflow_id}
                )
                try:
                    meta_data = await self.agent._get_valid_ums_data_payload(meta_res, "get_workflow_metadata")
                    current_meta = meta_data.get("metadata", {})
                except RuntimeError:
                    current_meta = {}
                count = current_meta.get(pair_key, 0) + 1
                
                # Update count
                current_meta[pair_key] = count
                update_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME, 
                    "update_workflow_metadata",
                    {"workflow_id": self.state.workflow_id, "metadata": current_meta}
                )
                try:
                    await self.agent._get_valid_ums_data_payload(update_res, "update_workflow_metadata")
                except RuntimeError as e:
                    self.logger.warning(f"Failed to update workflow metadata: {e}")
                
                # Escalate if threshold reached
                if count >= 3:
                    blocker_title = f"RESOLVE: Contradiction {pair[0][:8]}↔{pair[1][:8]}"
                    blocker_desc = f"Persistent contradiction detected {count} times. Requires explicit resolution."
                    await self._create_blocker_goal(blocker_title, blocker_desc, priority=1)  # highest priority

                    # Tag both memories so other components can suppress them
                    for mem in pair:
                        try:
                            tag_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                "add_tag_to_memory",
                                {"workflow_id": self.state.workflow_id,
                                 "memory_id": mem,
                                 "tag": "BLOCKER"},
                            )
                            try:
                                await self.agent._get_valid_ums_data_payload(tag_res, "add_tag_to_memory")
                            except RuntimeError as e:
                                self.logger.warning(f"Failed to tag memory {mem[:8]}: {e}")
                        except Exception as e:
                            self.logger.warning(f"Error tagging memory {mem[:8]}: {e}")
            except Exception:
                pass  # fail gracefully if metadata storage isn't available

    async def _validate_goal_outputs(self, goal_id: str) -> Tuple[bool, str]:
        """
        Validate that a goal has produced expected outputs.
        
        Returns
        -------
        Tuple[bool, str]
            (is_valid, reason) - True if goal outputs are valid, False with reason if not
        """
        try:
            # Get goal details
            goal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_goal_details"),
                {"workflow_id": self.state.workflow_id, "goal_id": goal_id}
            )
            
            try:
                goal_data_envelope = await self.agent._get_valid_ums_data_payload(goal_res, "get_goal_details")
                if not goal_data_envelope.get("goal"):
                    return False, f"Goal validation: no goal data in response: {goal_data_envelope}"
                goal_data = goal_data_envelope["goal"]
            except RuntimeError as e:
                return False, f"Could not retrieve goal details: {e}"
            goal_desc = goal_data.get("description", "").lower()
            goal_title = goal_data.get("title", "").lower()
            
            # Check if goal expects artifact creation
            creation_verbs = ["create", "write", "generate", "produce", "build", "develop", "implement", "design", "make", "output", "save", "export"]
            expects_artifact = any(verb in goal_desc or verb in goal_title for verb in creation_verbs)
            
            if expects_artifact:
                # Check for artifacts
                artifacts_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    "get_artifacts",
                    {
                        "workflow_id": self.state.workflow_id,
                        "limit": 10,
                        "is_output": True
                    }
                )
                
                try:
                    artifacts_data = await self.agent._get_valid_ums_data_payload(artifacts_res, "get_artifacts")
                    artifacts = artifacts_data.get("artifacts", [])
                    goal_created = goal_data.get("created_at", 0)
                    recent_artifacts = [
                        a for a in artifacts 
                        if a.get("created_at", 0) >= goal_created
                    ]
                    
                    if not recent_artifacts:
                        return False, f"Goal expects artifact creation but none found since goal creation"
                except RuntimeError as e:
                    return False, f"Could not verify artifacts for creation goal: {e}"
            
            # Check for substantive memory traces
            memories_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "get_linked_memories",
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_id": goal_id,
                    "limit": 20
                }
            )
            
            try:
                memories_data = await self.agent._get_valid_ums_data_payload(memories_res, "get_linked_memories")
                memories = memories_data.get("links", [])
                if len(memories) < 2:  # At least goal creation + some evidence
                    return False, "Insufficient memory trace for goal completion"
            except RuntimeError as e:
                return False, f"Could not verify memory traces for goal: {e}"
            
            return True, "Goal outputs validated successfully"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def _resolve_contradiction(self, mem_a: str, mem_b: str) -> None:
        """
        Actively resolve a contradiction by analyzing it with the LLM and creating a resolution.
        
        Parameters
        ----------
        mem_a, mem_b : str
            Memory IDs of the contradicting memories
        """
        try:
            # Get both memories
            mem_a_content = await self.mem_graph._get_memory_content(mem_a)
            mem_b_content = await self.mem_graph._get_memory_content(mem_b)
            
            if not mem_a_content or not mem_b_content:
                self.logger.warning(f"Could not retrieve content for contradiction resolution: {mem_a}, {mem_b}")
                return
            
            # Use fast LLM to analyze and resolve
            resolution_prompt = f"""
            Two memories contradict each other. Analyze the contradiction and provide a resolution.
            
            Memory A: {mem_a_content[:500]}
            Memory B: {mem_b_content[:500]}
            
            Provide:
            1. Why they contradict (brief explanation)
            2. Which is likely correct (A, B, both partially, or neither)
            3. A resolution statement that reconciles the information
            4. Confidence in the resolution (0.0-1.0)
            """
            
            schema = {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "assessment": {"type": "string"},
                    "resolution": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["reason", "assessment", "resolution", "confidence"]
            }
            
            result = await self.llm.fast_structured_call(resolution_prompt, schema)
            
            # Store resolution as a new memory
            resolution_content = (
                f"CONTRADICTION RESOLUTION:\n"
                f"Reason: {result['reason']}\n"
                f"Assessment: {result['assessment']}\n"
                f"Resolution: {result['resolution']}\n"
                f"Confidence: {result['confidence']:.2f}"
            )
            
            resolution_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "store_memory",
                {
                    "workflow_id": self.state.workflow_id,
                    "content": resolution_content,
                    "memory_type": "contradiction_resolution",
                    "memory_level": "working",
                    "importance": 8.0,  # High importance for resolutions
                    "description": f"Resolution of contradiction between {mem_a[:8]} and {mem_b[:8]}"
                }
            )
            
            try:
                resolution_data = await self.agent._get_valid_ums_data_payload(resolution_res, "store_memory_resolution")
                resolution_mem_id = resolution_data.get("memory_id")
                
                if resolution_mem_id:
                    # Link both original memories to the resolution
                    await self.mem_graph.auto_link(
                        src_id=mem_a,
                        tgt_id=resolution_mem_id,
                        context_snip="resolved contradiction",
                        kind_hint=LinkKind.REFERENCES
                    )
                    await self.mem_graph.auto_link(
                        src_id=mem_b,
                        tgt_id=resolution_mem_id,
                        context_snip="resolved contradiction",
                        kind_hint=LinkKind.REFERENCES
                    )
                    
                    # Mark the original contradiction as resolved
                    await self.mem_graph.mark_contradiction_resolved(mem_a, mem_b)
                    
                    self.logger.info(f"Resolved contradiction between {mem_a[:8]} and {mem_b[:8]} with confidence {result['confidence']:.2f}")
            except RuntimeError as e:
                self.logger.warning(f"Failed to store contradiction resolution memory: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to resolve contradiction between {mem_a} and {mem_b}: {e}")

    async def _create_blocker_goal(
        self,
        title: str,
        description: str,
        priority: int = 1,
        parent_goal_id: Optional[str] = None,
    ) -> str:
        """Create a blocker goal using UMS create_goal tool."""
        try:
            goal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "create_goal",
                {
                    "workflow_id": self.state.workflow_id,
                    "description": description,
                    "title": title,
                    "priority": priority,
                    "parent_goal_id": parent_goal_id,
                    "initial_status": "active",
                    "reasoning": f"Created blocker goal at loop {self.state.loop_count}"
                }
            )
            
            try:
                goal_data = await self.agent._get_valid_ums_data_payload(goal_res, "create_goal_blocker")
                if goal_data.get("goal"):
                    goal_id = goal_data["goal"]["goal_id"]
                    self.logger.debug(f"Created blocker goal {goal_id}: {title}")
                    return goal_id
                else:
                    self.logger.error(f"Failed to create blocker goal: No goal data in response: {goal_data}")
                    return str(uuid.uuid4())  # Fallback to prevent crashes
            except RuntimeError as e:
                self.logger.error(f"Failed to create blocker goal: {e}")
                return str(uuid.uuid4())  # Fallback to prevent crashes
                
        except Exception as e:
            self.logger.error(f"Error creating blocker goal: {e}")
            return str(uuid.uuid4())  # Fallback to prevent crashes


###############################################################################
# SECTION 6. Orchestrators & ToolExecutor (outline)
###############################################################################


class ToolExecutor:
    def __init__(self, mcp_client, state: AMLState, mem_graph):
        self.mcp_client = mcp_client
        self.state = state
        self.mem_graph = mem_graph
        self.agent = None  # Will be set by AgentMasterLoop
        
        # Build a proper tool-to-server mapping
        self._tool_to_server_map = {}
        self._ums_tool_names = set()
        
        # Define which tools are specifically UMS-related (these are suffixes)
        self._ums_tool_suffixes = {
            "create_workflow",
            "get_workflow_details",
            "record_action_start",
            "record_action_completion",
            "get_recent_actions",
            "get_thought_chain",
            "store_memory",
            "get_memory_by_id",
            "get_memory_metadata",
            "get_memory_tags",
            "update_memory_metadata",
            "update_memory_link_metadata",
            "create_memory_link",
            "get_workflow_metadata",
            "get_contradictions",
            "query_memories",
            "update_memory",
            "get_linked_memories",
            "add_tag_to_memory",
            "create_embedding",
            "get_embedding",
            "get_working_memory",
            "focus_memory",
            "optimize_working_memory",
            "save_cognitive_state",
            "load_cognitive_state",
            "decay_link_strengths",
            "generate_reflection",
            "get_rich_context_package",
            "get_goal_details",
            "create_goal",
            "vector_similarity",
            "record_artifact",
            "get_artifact_by_id",
            "get_similar_memories",
            "query_goals",
            "consolidate_memories",
            "diagnose_file_access_issues",
            "generate_workflow_report",
            "update_goal_status",
        }
        
        self._build_tool_server_mapping()

    def _build_tool_server_mapping(self):
        """Build mapping from tool names to their actual server names."""
        if not hasattr(self.mcp_client, 'server_manager'):
            return
        
        sm = self.mcp_client.server_manager
        
        # Map each tool to its server
        for tool_name in sm.tools.keys():
            # Find which server this tool belongs to
            for server_name, session in sm.active_sessions.items():
                # Check if this tool is from this server (handle colon format)
                if tool_name.startswith(f"{server_name}:"):
                    self._tool_to_server_map[tool_name] = server_name
                    break
                # Check if tool name contains the server name (sanitized format)
                elif server_name.replace(" ", "_") in tool_name:
                    self._tool_to_server_map[tool_name] = server_name
                    break
        
        # Identify UMS tools specifically by checking suffixes
        for tool_name in sm.tools.keys():
            if any(tool_name.endswith(ums_suffix) for ums_suffix in self._ums_tool_suffixes):
                self._ums_tool_names.add(tool_name)
        
        self.mcp_client.logger.debug(f"Built tool-to-server mapping for {len(self._tool_to_server_map)} tools")
        self.mcp_client.logger.debug(f"Identified {len(self._ums_tool_names)} UMS tools")
        
        # Debug: show some examples
        ums_examples = list(self._ums_tool_names)[:5]
        non_ums_examples = [name for name in list(sm.tools.keys())[:10] if name not in self._ums_tool_names][:5]
        self.mcp_client.logger.debug(f"UMS tool examples: {ums_examples}")
        self.mcp_client.logger.debug(f"Non-UMS tool examples: {non_ums_examples}")

    def _determine_server_for_tool(self, tool_name: str) -> str:
        """Determine which server a tool belongs to."""
        
        # First check our mapping
        if tool_name in self._tool_to_server_map:
            return self._tool_to_server_map[tool_name]
        
        # Try to rebuild mapping in case it's stale
        self._build_tool_server_mapping()
        if tool_name in self._tool_to_server_map:
            return self._tool_to_server_map[tool_name]
        
        # Handle base tool names by checking if they exist with UMS server prefix
        if hasattr(self.mcp_client, 'server_manager'):
            sm = self.mcp_client.server_manager
            
            # Try with UMS server prefix (the most common case)
            ums_tool_name = f"Ultimate MCP Server:{tool_name}"
            if ums_tool_name in sm.tools:
                self._tool_to_server_map[tool_name] = UMS_SERVER_NAME
                return UMS_SERVER_NAME
            
            # Check if it's already a full tool name
            if tool_name in sm.tools:
                # Extract server name from the tool name
                if ':' in tool_name:
                    server_name = tool_name.split(':', 1)[0]
                    if server_name in sm.active_sessions:
                        self._tool_to_server_map[tool_name] = server_name
                        return server_name
            
            # Try reverse lookup in sanitized mappings
            if hasattr(sm, 'sanitized_to_original'):
                if tool_name in sm.sanitized_to_original:
                    original_name = sm.sanitized_to_original[tool_name]
                    if ':' in original_name:
                        server_name = original_name.split(':', 1)[0]
                        self._tool_to_server_map[tool_name] = server_name
                        return server_name
                
                # Check if any sanitized name ends with our tool name
                for sanitized, original in sm.sanitized_to_original.items():
                    if sanitized.endswith(f"_{tool_name}") or sanitized == f"Ultimate_MCP_Server_{tool_name}":
                        if ':' in original:
                            server_name = original.split(':', 1)[0]
                            self._tool_to_server_map[tool_name] = server_name
                            return server_name
        
        # If all else fails, log detailed debugging info
        self.mcp_client.logger.error(f"Could not determine server for tool: {tool_name}")
        if hasattr(self.mcp_client, 'server_manager'):
            sm = self.mcp_client.server_manager
            self.mcp_client.logger.error(f"Available sessions: {list(sm.active_sessions.keys())}")
            self.mcp_client.logger.error(f"Tools in server manager: {len(sm.tools)}")
            self.mcp_client.logger.error(f"Sample tools: {list(sm.tools.keys())[:10]}")
            if hasattr(sm, 'sanitized_to_original'):
                self.mcp_client.logger.error(f"Sanitized mappings: {dict(list(sm.sanitized_to_original.items())[:5])}")
        
        raise RuntimeError(f"Could not determine server for tool: {tool_name}")

    async def run(self, llm_seen_tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Execute a tool with proper server routing."""
        
        # Map LLM-seen name back to original MCP name
        original_mcp_tool_name = None
        if hasattr(self.mcp_client, 'server_manager') and hasattr(self.mcp_client.server_manager, 'sanitized_to_original'):
            original_mcp_tool_name = self.mcp_client.server_manager.sanitized_to_original.get(llm_seen_tool_name)
        
        if not original_mcp_tool_name:
            # If not found, it might be that the LLM generated a tool name that wasn't in the provided list,
            # or the mapping wasn't updated correctly. Fallback to using llm_seen_tool_name directly
            # but log a significant warning.
            self.mcp_client.logger.warning(
                f"Could not map LLM tool name '{llm_seen_tool_name}' to an original MCP tool name using sanitized_to_original map. "
                f"Attempting to use '{llm_seen_tool_name}' directly. This may fail if it's not a valid original MCP name."
            )
            original_mcp_tool_name = llm_seen_tool_name  # Fallback

        # Determine the actual server for this original tool name
        server_name = self._determine_server_for_tool(original_mcp_tool_name)  # Pass original name
        is_ums_tool = original_mcp_tool_name in self._ums_tool_names  # Check original name
        
        # Start UMS action tracking only for actual actions (not internal UMS operations)
        action_id = None
        if not is_ums_tool:  # Only track non-UMS tools as "actions"
            try:
                action_start_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    f"{UMS_SERVER_NAME}:record_action_start",  # Use f-string for clarity
                    {
                        "workflow_id": self.state.workflow_id,
                        "action_type": "tool_use",
                        "reasoning": f"Executing tool {original_mcp_tool_name} to advance current goal",
                        "tool_name": original_mcp_tool_name,  # Log original name
                        "tool_args": tool_args,
                        "title": f"Execute {original_mcp_tool_name}",
                        # Ensure idempotency_key is robust, e.g., hash of args + loop count
                        "idempotency_key": f"{original_mcp_tool_name}_{hash(json.dumps(tool_args, sort_keys=True))}_{self.state.loop_count}"
                    }
                )
                
                try:
                    ums_data = await self.agent._get_valid_ums_data_payload(action_start_res, "record_action_start")
                    action_id = ums_data.get("action_id")
                    self.mcp_client.logger.debug(f"Started action tracking for {original_mcp_tool_name}: {action_id}")
                except RuntimeError as e:
                    self.mcp_client.logger.debug(f"Action tracking failed (non-critical): {e}")
                    action_id = None
            except Exception as e:
                self.mcp_client.logger.warning(f"Failed to start action tracking for {original_mcp_tool_name}: {e}")
        
        # Execute the actual tool using original_mcp_tool_name
        try:
            result_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                server_name, original_mcp_tool_name, tool_args  # Pass original name
            )
            
            success = result_envelope.get("success", False)
            
            # Complete action tracking for non-UMS tools
            if action_id:  # Only if action tracking was started
                try:
                    completion_status = "completed" if success else "failed"
                    summary = f"Tool {original_mcp_tool_name} {'succeeded' if success else 'failed'}"
                    if not success:
                        error_msg = result_envelope.get("error_message", "Unknown error")
                        summary += f": {error_msg}"
                    
                    await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        f"{UMS_SERVER_NAME}:record_action_completion",  # Use f-string
                        {
                            "action_id": action_id,
                            "status": completion_status,
                            "tool_result": result_envelope.get("data") if success else result_envelope,  # Store full envelope on error, data on success
                            "summary": summary,
                            "conclusion_thought": None  # Agent can add this later if needed
                        }
                    )
                    self.mcp_client.logger.debug(f"Completed action tracking for {original_mcp_tool_name}: {completion_status}")
                except Exception as e:
                    self.mcp_client.logger.warning(f"Failed to complete action tracking for {original_mcp_tool_name}: {e}")
            
            # Handle tool execution results
            if not success:
                error_msg = result_envelope.get("error_message", "Unknown tool execution error")
                self.mcp_client.logger.error(f"Tool {original_mcp_tool_name} failed: {error_msg}")
            
            return result_envelope  # Return the full envelope
            
        except Exception as e:
            # Complete action tracking with error status if action_id exists
            if action_id:
                try:
                    await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        f"{UMS_SERVER_NAME}:record_action_completion",  # Use f-string
                        {
                            "action_id": action_id,
                            "status": "failed",
                            "tool_result": {"error": str(e), "tool_name": original_mcp_tool_name},
                            "summary": f"Tool {original_mcp_tool_name} failed with exception: {str(e)}",
                            "conclusion_thought": None
                        }
                    )
                except Exception:
                    pass  # Don't let action tracking errors mask the original error
            
            self.mcp_client.logger.error(f"Tool {original_mcp_tool_name} execution failed with exception: {e}", exc_info=True)
            return {"success": False, "error_message": f"Exception during tool execution: {str(e)}", "error_type": "ToolExecutionException"}

    async def run_parallel(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple independent tools in parallel with proper server routing."""
        if not tool_calls:
            return {"success": True, "results": [], "timing": {}, "batch_memory_id": None}

        start_time = time.time()
        
        # Start batch action tracking
        batch_action_id = None
        try:
            batch_action_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                f"{UMS_SERVER_NAME}:record_action_start",
                {
                    "workflow_id": self.state.workflow_id,
                    "action_type": "parallel_tools",
                    "reasoning": f"Executing {len(tool_calls)} tools in parallel",
                    "tool_name": "parallel_execution",
                    "tool_args": {"tool_count": len(tool_calls), "tools": [tc.get("tool_name") for tc in tool_calls]},
                    "title": f"Parallel execution of {len(tool_calls)} tools"
                }
            )
            
            try:
                ums_data = await self.agent._get_valid_ums_data_payload(batch_action_res, "record_action_start_batch")
                batch_action_id = ums_data.get("action_id")
            except RuntimeError as e:
                self.mcp_client.logger.debug(f"Batch action tracking failed (non-critical): {e}")
                batch_action_id = None
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to start batch action tracking: {e}")

        # Create tasks with proper error handling and server routing
        tasks = []
        tool_identifiers = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call['tool_name']
            tool_args = call.get('tool_args', {})
            tool_id = call.get('tool_id', f"{tool_name}_{i}")
            
            # Create coroutine for this tool with proper server determination
            async def execute_with_timing(name=tool_name, args=tool_args, tid=tool_id):
                t_start = time.time()
                try:
                    # Determine server for this specific tool
                    server_name = self._determine_server_for_tool(name)
                    is_ums_tool = name in self._ums_tool_names
                    
                    # Start individual action tracking for non-UMS tools
                    individual_action_id = None
                    if not is_ums_tool:
                        try:
                            action_start_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                f"{UMS_SERVER_NAME}:record_action_start",
                                {
                                    "workflow_id": self.state.workflow_id,
                                    "action_type": "tool_use",
                                    "reasoning": f"Executing tool {name} to advance current goal",
                                    "tool_name": name,
                                    "tool_args": args,
                                    "title": f"Execute {name}",
                                    "idempotency_key": f"{name}_{hash(str(args))}_{self.state.loop_count}"
                                }
                            )
                            
                            try:
                                ums_data = await self.agent._get_valid_ums_data_payload(action_start_res, "record_action_start_individual")
                                individual_action_id = ums_data.get("action_id")
                            except RuntimeError as e:
                                self.mcp_client.logger.debug(f"Individual action tracking failed (non-critical): {e}")
                                individual_action_id = None
                        except Exception as e:
                            self.mcp_client.logger.warning(f"Failed to start individual action tracking for {name}: {e}")
                    
                    # Execute the tool
                    result = await self.mcp_client._execute_tool_and_parse_for_agent(
                        server_name, name, args
                    )
                    
                    success = result.get("success", False)
                    
                    # Complete individual action tracking for non-UMS tools
                    if individual_action_id:
                        try:
                            completion_status = "completed" if success else "failed"
                            summary = f"Tool {name} {'succeeded' if success else 'failed'}"
                            if not success:
                                error_msg = result.get("error_message", "Unknown error")
                                summary += f": {error_msg}"
                            
                            await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                f"{UMS_SERVER_NAME}:record_action_completion",
                                {
                                    "action_id": individual_action_id,
                                    "status": completion_status,
                                    "tool_result": result if success else None,
                                    "summary": summary
                                }
                            )
                        except Exception as e:
                            self.mcp_client.logger.warning(f"Failed to complete individual action tracking for {name}: {e}")
                    
                    return {
                        "tool_id": tid,
                        "tool_name": name,
                        "success": success,
                        "result": result,
                        "execution_time": time.time() - t_start,
                        "error": None
                    }
                except Exception as e:
                    self.mcp_client.logger.error(f"Parallel execution error for {name}: {e}")
                    
                    # Complete individual action tracking with error for non-UMS tools
                    if 'individual_action_id' in locals() and individual_action_id:
                        try:
                            await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                f"{UMS_SERVER_NAME}:record_action_completion",
                                {
                                    "action_id": individual_action_id,
                                    "status": "failed",
                                    "summary": f"Tool {name} failed with exception: {str(e)}"
                                }
                            )
                        except Exception:
                            pass
                    
                    return {
                        "tool_id": tid,
                        "tool_name": name,
                        "success": False,
                        "result": {"success": False, "error_message": str(e)},
                        "execution_time": time.time() - t_start,
                        "error": str(e)
                    }
            
            tasks.append(execute_with_timing())
            tool_identifiers.append(tool_id)

        # Execute all tasks concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Complete batch action tracking
        total_time = time.time() - start_time
        successful_count = sum(1 for r in task_results if r["success"])
        total_count = len(tool_calls)
        
        if batch_action_id:
            try:
                batch_summary = f"Parallel execution completed: {successful_count}/{total_count} tools succeeded in {total_time:.2f}s"
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    f"{UMS_SERVER_NAME}:record_action_completion",
                    {
                        "action_id": batch_action_id,
                        "status": "completed" if successful_count > 0 else "failed",
                        "tool_result": {
                            "successful_count": successful_count,
                            "total_count": total_count,
                            "execution_time": total_time,
                            "tool_results": [r["result"] for r in task_results]
                        },
                        "summary": batch_summary
                    }
                )
            except Exception as e:
                self.mcp_client.logger.warning(f"Failed to complete batch action tracking: {e}")

        # Build ordered results matching input order
        ordered_results = []
        timing_info = {}
        
        for i, tool_id in enumerate(tool_identifiers):
            for result in task_results:
                if result["tool_id"] == tool_id:
                    ordered_results.append(result["result"])
                    timing_info[tool_id] = result["execution_time"]
                    break

        return {
            "success": successful_count > 0,
            "results": ordered_results,
            "timing": timing_info,
            "batch_memory_id": batch_action_id  # Return action ID instead of memory ID
        }

class LLMOrchestrator:
    def __init__(self, mcp_client, state: AMLState):
        self.mcp_client = mcp_client
        self.state = state
        # REMOVED: No more StructuredCall instance creation
        # self._fast_query = StructuredCall(mcp_client, model_name=fast_model)

    async def fast_structured_call(self, prompt: str, schema: dict[str, Any]):
        """Direct call to MCPClient without StructuredCall wrapper."""
        try:
            fast_model = getattr(self.mcp_client.config, 'default_cheap_and_fast_model', DEFAULT_FAST_MODEL)
            return await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": prompt}],
                response_schema=schema,
                model_override=fast_model,
                use_cheap_model=True
            )
        except Exception as e:
            self.mcp_client.logger.error(f"Fast structured call failed: {e}")
            raise

    async def big_reasoning_call(self, messages: list[dict], tool_schemas=None, model_name: str = None):
        """
        Single entry for SMART model calls with ENFORCED structured output.
        
        Parameters
        ----------
        messages : list[dict]
            Chat completion messages in the standard format
        tool_schemas : optional
            Tool schemas for the agent to use
        model_name : str, optional
            Model name to use. If not provided, uses the MCPClient's current model.
        """
        # Use the passed model_name or fall back to MCPClient's current model
        target_model = model_name or self.mcp_client.current_model
        
        self.mcp_client.logger.info(f"[LLMOrch] Starting big_reasoning_call with model: {target_model}")
        
        # Extract valid tool names from schemas - these are already properly formatted
        valid_tool_names = []
        if tool_schemas:
            for schema in tool_schemas:
                if "function" in schema:
                    tool_name = schema["function"].get("name")
                    if tool_name:
                        valid_tool_names.append(tool_name)
        
        # Safety check: ensure we have valid tools
        if not valid_tool_names:
            self.mcp_client.logger.error(f"[LLMOrch] No valid tool names extracted from {len(tool_schemas)} schemas")
            return {
                "decision_type": "THOUGHT_PROCESS",
                "content": "No valid tools available. Cannot proceed with tool-based actions."
            }
        
        self.mcp_client.logger.info(f"[LLMOrch] Extracted {len(valid_tool_names)} valid tool names from schemas")
        self.mcp_client.logger.debug(f"[LLMOrch] Valid tool names: {valid_tool_names[:5]}...")  # Show first 5
        
        # Build response schema that matches what _enact expects
        response_schema = {
            "type": "object",
            "properties": {
                "decision_type": {
                    "type": "string",
                    "enum": ["TOOL_SINGLE", "TOOL_MULTIPLE", "THOUGHT_PROCESS", "DONE"],
                    "description": "Type of decision: TOOL_SINGLE for single tool call, TOOL_MULTIPLE for multiple tools, THOUGHT_PROCESS for reasoning only, DONE when task complete"
                },
                "tool_name": {
                    "type": "string",
                    "enum": valid_tool_names if valid_tool_names else ["no_tools_available"],
                    "description": "CRITICAL: Must use EXACT tool name from the enum list. Tool names include full server prefix like 'Ultimate_MCP_Server_browse'."
                },
                "tool_args": {
                    "type": "object",
                    "description": "Arguments for the tool call (required for TOOL_SINGLE). Should be empty if no arguments are needed for the selected tool, or contain only the arguments defined by that tool's specific schema.",
                    "properties": {},
                    "additionalProperties": False
                },
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "enum": valid_tool_names if valid_tool_names else ["no_tools_available"],
                                "description": "CRITICAL: Must use EXACT tool name from the enum list."
                            },
                            "tool_args": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False
                            },
                            "tool_id": {"type": "string", "description": "Unique identifier for this tool call"}
                        },
                        "required": ["tool_name", "tool_args", "tool_id"],
                        "additionalProperties": False
                    },
                    "description": "List of tools for TOOL_MULTIPLE execution"
                },
                "content": {
                    "type": "string", 
                    "description": "Content for THOUGHT_PROCESS or DONE decisions"
                }
            },
            "required": ["decision_type"],
            "additionalProperties": False
        }
        
        self.mcp_client.logger.info(f"[LLMOrch] Calling query_llm_structured with {len(valid_tool_names)} constrained tools")
        self.mcp_client.logger.info(f"[LLMOrch] Model: {target_model}, Schema: agent_decision")
        self.mcp_client.logger.info(f"[LLMOrch] Tool name enum constraint has {len(valid_tool_names)} valid options")
        
        try:
            # Convert response_schema to the proper response_format for structured output
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_decision",
                    "description": "Agent reasoning decision schema",
                    "strict": True,
                    "schema": response_schema
                }
            }
            
            # Use MCPClient's process_agent_llm_turn for structured LLM interaction
            # This method handles the full flow including tool schemas and structured output
            resp = await self.mcp_client.process_agent_llm_turn(
                prompt_messages=messages,
                tool_schemas=tool_schemas or [],  # Pass tool schemas
                model_name=target_model,
                response_format=response_format,  # Pass the response format for structured output
                ums_tool_schemas=await self._get_ums_tool_schemas_for_agent_use() if hasattr(self, '_get_ums_tool_schemas_for_agent_use') else None
            )
            
            self.mcp_client.logger.info(f"[LLMOrch] Agent LLM turn response received: {type(resp)}")
            
            # process_agent_llm_turn returns a different format than query_llm_structured
            # Extract the actual structured content from the response
            if isinstance(resp, dict):
                # Check if it's already in the expected agent format (detected by process_agent_llm_turn)
                if "decision_type" in resp:
                    self.mcp_client.logger.info(f"[LLMOrch] Decision type: {resp.get('decision_type')}")
                    return resp
                
                # If it's a tool call response from process_agent_llm_turn, convert it
                if resp.get("decision") == "TOOL_SINGLE" and resp.get("tool_name"):
                    return {
                        "decision_type": "TOOL_SINGLE",
                        "tool_name": resp["tool_name"],
                        "tool_args": resp.get("tool_args", {})
                    }
                elif resp.get("decision") == "PARALLEL_TOOLS" and resp.get("tool_calls"):
                    return {
                        "decision_type": "TOOL_MULTIPLE", 
                        "tool_calls": resp["tool_calls"]
                    }
                elif resp.get("decision") == "plan" and resp.get("updated_plan_steps"):
                    # Convert plan response to thought process
                    return {
                        "decision_type": "THOUGHT_PROCESS",
                        "content": f"Planning response with {len(resp['updated_plan_steps'])} steps"
                    }
            
            # Try to repair common response format issues
            if isinstance(resp, dict):
                # Check if it's in the old tool format and convert it
                if "tool" in resp and "tool_input" in resp:
                    tool_name = resp.get("tool", "")
                    tool_args = resp.get("tool_input", {})
                    
                    # Validate tool name exists in our valid tools
                    if tool_name not in valid_tool_names:
                        self.mcp_client.logger.error(f"[LLMOrch] LLM returned invalid tool name: {tool_name}. Valid tools: {valid_tool_names[:5]}...")
                        return {
                            "decision_type": "THOUGHT_PROCESS",
                            "content": f"LLM attempted to use non-existent tool '{tool_name}'. Available tools: {', '.join(valid_tool_names[:5])}... Need to select from these valid options."
                        }
                    
                    # Convert to expected format
                    self.mcp_client.logger.warning(f"[LLMOrch] Converting old tool format to new format")
                    return {
                        "decision_type": "TOOL_SINGLE",
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    }
                
                # Check if it's a planning response
                if "plan" in resp:
                    self.mcp_client.logger.warning(f"[LLMOrch] LLM returned planning response, converting to thought")
                    return {
                        "decision_type": "THOUGHT_PROCESS",
                        "content": f"Planning response received: {resp}"
                    }
            
            # If we can't repair it, return a thought process instead of ERROR
            self.mcp_client.logger.error(f"[LLMOrch] Invalid response format: {resp}")
            return {
                "decision_type": "THOUGHT_PROCESS",
                "content": f"LLM returned unexpected format. Need to analyze the goal and determine next steps. Response was: {str(resp)[:200]}"
            }
                
        except Exception as e:
            self.mcp_client.logger.error(f"[LLMOrch] Error in structured LLM call: {e}")
            return {
                "decision_type": "THOUGHT_PROCESS", 
                "content": f"LLM call failed with error: {str(e)}. Need to proceed with basic reasoning to determine next steps."
            }

###############################################################################
# SECTION 7. Procedural Agenda & Planner helpers (outline)
###############################################################################



















###############################################################################
# SECTION 8. AgentMasterLoop (outline)
###############################################################################


class AgentMasterLoop:
    """
    Top-level orchestrator that coordinates:

        • Phase progression & loop counting
        • Dual-LLM reasoning via `LLMOrchestrator`
        • Tool execution via `ToolExecutor`
        • Asynchronous cheap-LLM micro-tasks
        • Metacognition & graph maintenance

    The class purposefully stays *thin*: it delegates heavy lifting to the
    specialised helpers already implemented in previous sections.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, mcp_client, default_llm_model_string: str) -> None:
        """
        Initialize the AgentMasterLoop with MCP client integration.
        
        Parameters
        ----------
        mcp_client : MCPClient
            The MCP client instance for tool execution and LLM calls
        default_llm_model_string : str
            Default model name to use for reasoning calls
        """
        self.mcp_client = mcp_client
        self.default_llm_model = default_llm_model_string
        self.logger = logging.getLogger("AML")
        
        # Agent state - will be loaded/initialized in initialize()
        self.state = AMLState()
        
        # Component instances - initialized in _initialize_components()
        self.mem_graph: Optional[MemoryGraphManager] = None
        self.tool_exec: Optional[ToolExecutor] = None
        self.llms: Optional[LLMOrchestrator] = None
        self.async_queue: Optional[AsyncTaskQueue] = None
        self.metacog: Optional[MetacognitionEngine] = None

        
        # Tool schemas cache
        self.tool_schemas: List[Dict[str, Any]] = []
        
        # Tool effectiveness tracking
        self._tool_effectiveness_cache: Dict[str, Dict[str, int]] = {}

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

    @property
    def agent_llm_model(self) -> str:
        """Compatibility property for MCPClient access."""
        return self.default_llm_model

    @agent_llm_model.setter  
    def agent_llm_model(self, value: str) -> None:
        """Compatibility setter for MCPClient access."""
        self.default_llm_model = value
        
    # ---------------------------------------------------------------- initialization

    async def initialize(self) -> bool:
        """
        Load persistent state, validate against UMS, and initialize all components.
        
        This method:
        1. Loads agent state from the persistent state file
        2. Validates workflow_id and goal_id against UMS
        3. Resets state if validation fails
        4. Initializes all agent components
        5. Refreshes tool schemas
        6. Loads existing goals from UMS if workflow exists
        7. Saves validated state back to file
        
        Returns
        -------
        bool
            True if initialization succeeded, False if it failed
        """
        try:
            # Validate loaded state against UMS
            await self._validate_and_reset_state()
            
            # Initialize components with validated state
            self._initialize_components()
            
            # Refresh tool schemas
            await self._refresh_tool_schemas()
            
            # Note: Goal creation is handled directly by MetacognitionEngine when needed
            
            # Save validated/updated state
            if self.state.workflow_id:
                await self._save_cognitive_state()
            else:
                self.logger.debug("No workflow_id available, skipping state save during initialization")

            self.logger.info(f"AgentMasterLoop initialized with workflow_id: {self.state.workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"AgentMasterLoop initialization failed: {e}", exc_info=True)
            return False

    async def _load_cognitive_state(self) -> None:
        """Load agent state using UMS cognitive state tools."""
        # Early return if no workflow_id
        if not self.state.workflow_id:
            self.logger.info("No workflow_id available, skipping cognitive state loading")
            return
            
        try:
            # Call UMS load_cognitive_state tool with correct parameters
            load_res_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("load_cognitive_state"),
                {
                    "workflow_id": self.state.workflow_id,
                    "state_id": None  # Get the latest state
                }
            )
            # Check mcp_client_response.get("success") first, if it's False, tool call itself failed at MCPC level.
            if not load_res_envelope.get("success"):
                self.logger.info(f"MCPClient: UMS load_cognitive_state tool call failed: {load_res_envelope.get('error_message', 'Unknown error')}")
                return  # Use default state
            
            # If MCPClient call was successful, then try to get the UMS tool's actual data payload
            try:
                cognitive_state_data = await self._get_valid_ums_data_payload(load_res_envelope, "load_cognitive_state")
                
                if cognitive_state_data.get("state_id"):  # Check for state_id within the UMS payload
                    # We have actual state data - update our state from it
                    self.state.workflow_id = cognitive_state_data.get("workflow_id", self.state.workflow_id)
                    self.state.context_id = cognitive_state_data.get("state_id", self.state.context_id)  # state_id from UMS IS the context_id

                    # Load current_goals (should be a list of goal IDs)
                    persisted_current_goals_list_json = cognitive_state_data.get("current_goals")  # This is a JSON string from DB
                    persisted_current_goals_list = []
                    if persisted_current_goals_list_json:
                        try:
                            import json
                            persisted_current_goals_list = json.loads(persisted_current_goals_list_json) if isinstance(persisted_current_goals_list_json, str) else persisted_current_goals_list_json
                        except (json.JSONDecodeError, TypeError):
                            persisted_current_goals_list = []

                    if isinstance(persisted_current_goals_list, list) and persisted_current_goals_list:
                        # Assuming the first one is the current leaf/operational goal for now
                        self.state.current_leaf_goal_id = persisted_current_goals_list[0]
                        self.logger.info(f"Loaded current_leaf_goal_id from cognitive state: {self.state.current_leaf_goal_id[:8] if self.state.current_leaf_goal_id else 'None'}")
                    elif self.state.root_goal_id:  # Fallback if current_goals is empty but root_goal_id is known
                        self.state.current_leaf_goal_id = self.state.root_goal_id
                        self.logger.info(f"No current_goals in loaded state, falling back to root_goal_id: {self.state.root_goal_id[:8] if self.state.root_goal_id else 'None'}")
                    else:
                        self.state.current_leaf_goal_id = None
                        self.logger.warning("No current_goals in loaded state and no root_goal_id available to fall back on for current_leaf_goal_id.")
                    
                    # focus_areas should contain memory IDs for contextual focus, not goal IDs.
                    # Example: self.state.focus_area_memory_ids = cognitive_state_data.get("focus_areas", [])

                    self.logger.info(f"Loaded cognitive state from UMS: {cognitive_state_data.get('title', 'Unknown')}")
                else:
                    self.logger.info("No cognitive state data found in UMS response payload, using default state")
            except RuntimeError as e:  # Catch error from _get_valid_ums_data_payload
                self.logger.info(f"UMS load_cognitive_state tool reported an issue or returned invalid structure: {e}. Using default state.")
                # Potentially reset parts of the state or use defaults
                
        except Exception as e:
            self.logger.warning(f"Failed to load cognitive state from UMS: {e}")
            self.logger.info("Using default state")

    async def _validate_and_reset_state(self) -> None:
        """Validate loaded workflow_id and goal_id against UMS, reset if invalid."""
        if not self.state.workflow_id:
            self.logger.info("No workflow_id in state, will create new workflow when needed")
            return
            
        try:
            # Validate workflow exists in UMS
            workflow_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_workflow_details"), 
                {"workflow_id": self.state.workflow_id}
            )
            
            try:
                await self._get_valid_ums_data_payload(workflow_res, "get_workflow_details_validation")
            except RuntimeError as e:
                self.logger.warning(f"Workflow {self.state.workflow_id} not found in UMS: {e}, resetting state")
                self._reset_workflow_state()
                return
                
            # Validate root goal exists if specified
            if self.state.root_goal_id:
                goal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_goal_details"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "goal_id": self.state.root_goal_id
                    }
                )
                
                try:
                    await self._get_valid_ums_data_payload(goal_res, "get_goal_details_validation")
                except RuntimeError as e:
                    self.logger.warning(f"Root goal {self.state.root_goal_id} not found in UMS: {e}, resetting goals")
                    self.state.root_goal_id = None
                    self.state.current_leaf_goal_id = None
                    self.state.needs_replan = True
                    
        except Exception as e:
            self.logger.warning(f"Error validating state against UMS: {e}, resetting workflow state")
            self._reset_workflow_state()

    def _reset_workflow_state(self) -> None:
        """Reset workflow-related state fields."""
        self.state.workflow_id = None
        self.state.root_goal_id = None
        self.state.current_leaf_goal_id = None
        self.state.current_plan = []
        self.state.goal_stack = []
        self.state.needs_replan = True
        self.state.goal_achieved_flag = False
        self.state.context_id = None 

    def _initialize_components(self) -> None:
        """Initialize all agent components with the validated state."""
        # Initialize core components
        self.mem_graph = MemoryGraphManager(self.mcp_client, self.state)
        self.async_queue = AsyncTaskQueue(max_concurrency=6)

        # Create orchestrators / engines
        self.llms = LLMOrchestrator(self.mcp_client, self.state)
        self.tool_exec = ToolExecutor(self.mcp_client, self.state, self.mem_graph)
        self.metacog = MetacognitionEngine(self.mcp_client, self.state, self.mem_graph, self.llms, self.async_queue)

        # Link components after initialization
        self.metacog.set_agent(self)
        self.mem_graph.agent = self  # Give MemoryGraphManager access to helper methods
        self.tool_exec.agent = self  # Give ToolExecutor access to helper methods



    async def _add_to_working_memory(self, memory_id: str, make_focal: bool = False) -> bool:
        """
        Properly add a memory to working memory using UMS tools.
        
        Parameters
        ----------
        memory_id : str
            The memory ID to add
        make_focal : bool
            Whether to make this the focal memory
            
        Returns
        -------
        bool
            True if successfully added
        """
        if not self.state.context_id:
            self.logger.warning("No context_id available, cannot add to working memory")
            return False
            
        try:
            if make_focal:
                # Use focus_memory which both adds to working and sets as focal
                res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("focus_memory"),
                    {
                        "memory_id": memory_id,
                        "context_id": self.state.context_id,
                        "add_to_working": True
                    }
                )
                
                try:
                    await self._get_valid_ums_data_payload(res, "focus_memory")
                    self.logger.debug(f"Added {memory_id} to working memory as focal")
                    return True
                except RuntimeError as e:
                    self.logger.warning(f"Failed to focus memory {memory_id}: {e}")
                    return False
            else:
                # Get current working memory
                wm_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_working_memory"),
                    {
                        "context_id": self.state.context_id,
                        "include_content": False,
                        "update_access": False
                    }
                )
                
                current_ids = []
                try:
                    wm_data = await self._get_valid_ums_data_payload(wm_res, "get_working_memory")
                    memories = wm_data.get("working_memories", [])
                    current_ids = [m["memory_id"] for m in memories if m.get("memory_id")]
                except RuntimeError as e:
                    self.logger.warning(f"Failed to get working memory: {e}")
                    return False
                
                if memory_id not in current_ids:
                    current_ids.append(memory_id)
                    
                    # Limit size
                    if len(current_ids) > 20:
                        # Use optimize_working_memory instead of manual truncation
                        try:
                            optimize_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                self._get_ums_tool_name("optimize_working_memory"),
                                {
                                    "context_id": self.state.context_id,
                                    "target_size": 15,
                                    "strategy": "balanced"
                                }
                            )
                            await self._get_valid_ums_data_payload(optimize_res, "optimize_working_memory")
                            return True
                        except RuntimeError as e:
                            self.logger.warning(f"Failed to optimize working memory: {e}")
                            return False
                    
                    # Update cognitive state
                    res = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("save_cognitive_state"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "title": f"Updated working memory at loop {self.state.loop_count}",
                            "working_memory_ids": current_ids,
                            "focus_area_ids": [],
                            "context_action_ids": [],
                            "current_goals": [self.state.current_leaf_goal_id] if self.state.current_leaf_goal_id else []
                        }
                    )
                    
                    try:
                        await self._get_valid_ums_data_payload(res, "save_cognitive_state")
                        self.logger.debug(f"Added {memory_id} to working memory")
                        return True
                    except RuntimeError as e:
                        self.logger.warning(f"Failed to save cognitive state with new memory: {e}")
                        return False
                        
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to add memory to working memory: {e}")
            return False
            
    async def _refresh_tool_schemas(self) -> None:
        """Refresh tool schemas from MCPClient for LLM calls."""
        try:
            # Check if MCPClient and ServerManager are available
            if not getattr(self.mcp_client, "server_manager", None):
                self.logger.error("MCPClient or ServerManager not available")
                self.tool_schemas = []
                return

            # Get the provider for the current model
            provider = self.mcp_client.get_provider_from_model(self.agent_llm_model)
            if not provider:
                self.logger.error(f"Cannot determine provider for model {self.agent_llm_model}")
                self.tool_schemas = []
                return

            self.logger.debug(f"Provider detected for model {self.agent_llm_model}: {provider}")

            # Check how many tools are available before formatting
            sm = self.mcp_client.server_manager
            self.logger.debug(f"ServerManager has {len(sm.tools)} tools before formatting")
            
            # Debug: Show first few tool names
            if sm.tools:
                tool_sample = list(sm.tools.keys())[:5]
                self.logger.debug(f"Sample tool names: {tool_sample}")
            
            # Use the MCPClient's method to get properly formatted tools
            all_llm_formatted = self.mcp_client._format_tools_for_provider(provider)
            
            # Debug what we got back
            if all_llm_formatted is None:
                self.logger.error("_format_tools_for_provider returned None")
                all_llm_formatted = []
            elif not isinstance(all_llm_formatted, list):
                self.logger.error(f"_format_tools_for_provider returned unexpected type: {type(all_llm_formatted)}")
                all_llm_formatted = []
            else:
                self.logger.debug(f"_format_tools_for_provider returned {len(all_llm_formatted)} formatted tools")
            
            # If we got nothing from the formatting method, try a comprehensive fallback approach
            if not all_llm_formatted and sm.tools:
                self.logger.warning(f"Formatting method returned empty for provider '{provider}', trying comprehensive fallback approach")
                
                # Try to format the tools ourselves as a fallback
                fallback_tools = []
                for tool_name, tool_schema in list(sm.tools.items()):
                    try:
                        # Get basic tool info
                        description = tool_schema.get("description", f"Tool: {tool_name}")
                        input_schema = tool_schema.get("inputSchema", {})
                        
                        # Ensure input_schema is valid
                        if not isinstance(input_schema, dict):
                            input_schema = {"type": "object", "properties": {}, "required": []}
                        elif input_schema.get("type") != "object":
                            # Wrap non-object schemas
                            input_schema = {"type": "object", "properties": {"input": input_schema}, "required": []}
                        
                        # Sanitize tool name for LLM compatibility
                        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)[:64]
                        if not sanitized_name:
                            sanitized_name = f"tool_{len(fallback_tools)}"
                        
                        if provider == "anthropic":
                            # Anthropic format
                            formatted_tool = {
                                "name": sanitized_name,
                                "description": description,
                                "input_schema": input_schema
                            }
                        else:
                            # OpenAI format (works for most providers)
                            formatted_tool = {
                                "type": "function",
                                "function": {
                                    "name": sanitized_name,
                                    "description": description,
                                    "parameters": input_schema
                                }
                            }
                        
                        fallback_tools.append(formatted_tool)
                        
                        # Store the mapping for tool execution
                        sm.sanitized_to_original[sanitized_name] = tool_name
                        
                        # Limit fallback to prevent overwhelming the LLM
                        if len(fallback_tools) >= 50:  # Increased from 20 to 50
                            break
                            
                    except Exception as e:
                        self.logger.debug(f"Failed to fallback format tool {tool_name}: {e}")
                        continue
                
                if fallback_tools:
                    self.logger.info(f"Fallback formatting produced {len(fallback_tools)} tools")
                    all_llm_formatted = fallback_tools
                else:
                    self.logger.error("Even fallback formatting failed - no tools available")
            
            self.logger.info(f"Received {len(all_llm_formatted)} tool schemas from MCPClient")
            
            # Store the formatted schemas
            self.tool_schemas = all_llm_formatted

            self.logger.info(f"Loaded {len(self.tool_schemas)} tool schemas")
        except Exception as e:
            self.logger.error(f"Failed to refresh tool schemas: {e}", exc_info=True)
            self.tool_schemas = []

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Construct the full MCP tool name for UMS tools."""
        return f"{UMS_SERVER_NAME}:{base_tool_name}"

    async def _get_valid_ums_data_payload(self, mcp_client_response: Dict[str, Any], tool_name_for_error_context: str) -> Dict[str, Any]:
        """
        Safely unwraps the nested 'data' payload from a UMS tool's response envelope.
        
        'mcp_client_response' is the direct output from mcp_client._execute_tool_and_parse_for_agent.
        
        Returns the actual data payload from the UMS tool if successful.
        Raises RuntimeError if any part of the expected structure is missing or indicates failure.
        """
        self.logger.debug(f"Unwrapping UMS response for {tool_name_for_error_context}. MCPClient envelope: {str(mcp_client_response)[:500]}...")

        if not isinstance(mcp_client_response, dict):
            self.logger.error(f"MCPClient response for {tool_name_for_error_context} is not a dict: {type(mcp_client_response)}")
            raise RuntimeError(f"MCPClient response for {tool_name_for_error_context} is not a dict.")

        if not mcp_client_response.get("success"):
            error_msg = mcp_client_response.get("error_message", f"MCPClient communication layer reported failure for {tool_name_for_error_context}")
            error_type = mcp_client_response.get("error_type", "MCPClientError")
            self.logger.error(f"MCPClient communication layer failure for {tool_name_for_error_context}: {error_type} - {error_msg}. Full response: {mcp_client_response}")
            raise RuntimeError(f"MCPClient layer failure for {tool_name_for_error_context}: {error_type} - {error_msg}")

        # 'mcp_client_response["data"]' should contain the UMS tool's own envelope
        ums_tool_envelope = mcp_client_response.get("data")
        if not isinstance(ums_tool_envelope, dict):
            self.logger.error(f"UMS tool {tool_name_for_error_context} response's outer 'data' field (expected UMS tool's envelope) is not a dict: {type(ums_tool_envelope)}. Full MCPClient data: {ums_tool_envelope}")
            raise RuntimeError(f"Unexpected response structure from {tool_name_for_error_context}: UMS tool envelope is not a dictionary.")

        if not ums_tool_envelope.get("success"):
            ums_error_msg = ums_tool_envelope.get("error_message", f"UMS tool {tool_name_for_error_context} reported an internal error.")
            ums_error_type = ums_tool_envelope.get("error_type", "UMSToolError")
            ums_details = ums_tool_envelope.get("details", "No details provided.")
            self.logger.error(f"UMS tool {tool_name_for_error_context} reported internal failure: {ums_error_type} - {ums_error_msg}. Details: {ums_details}. Full UMS envelope: {ums_tool_envelope}")
            raise RuntimeError(f"UMS tool {tool_name_for_error_context} failure: {ums_error_type} - {ums_error_msg}")
        
        # 'ums_tool_envelope["data"]' is the actual data payload from the UMS tool
        actual_ums_data_payload = ums_tool_envelope.get("data")
        if not isinstance(actual_ums_data_payload, dict):
            self.logger.error(f"UMS tool {tool_name_for_error_context} successful response's inner 'data' field (actual payload) is not a dict: {type(actual_ums_data_payload)}. Full UMS envelope: {ums_tool_envelope}")
            raise RuntimeError(f"Unexpected payload structure from {tool_name_for_error_context}: UMS tool's actual data payload is not a dictionary.")
            
        self.logger.debug(f"Successfully unwrapped UMS data payload for {tool_name_for_error_context}: {str(actual_ums_data_payload)[:300]}...")
        return actual_ums_data_payload

    async def _get_ums_tool_schemas_for_agent_use(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Build the specific dictionary of UMS tool schemas that MCPClient requires 
        to build the structured output JSON for ums_tool_schemas.
        """
        try:
            if not hasattr(self.mcp_client, 'server_manager') or not self.mcp_client.server_manager:
                self.logger.warning("MCPClient.server_manager not available for UMS schema retrieval.")
                return None
                
            server_manager = self.mcp_client.server_manager
            ums_tools_schemas: Dict[str, Dict[str, Any]] = {}  # Stores llm_seen_name -> schema
            
            # Iterate over all tools known to the ServerManager
            for original_mcp_name, tool_obj in server_manager.tools.items():
                # Check if this tool is from the UMS server
                if original_mcp_name.startswith(f"{UMS_SERVER_NAME}:") and hasattr(tool_obj, 'input_schema'):
                    # Find the LLM-seen (sanitized) name for this original UMS tool name
                    llm_seen_name_for_this_ums_tool = None
                    if hasattr(server_manager, 'sanitized_to_original'):
                        for sanitized_name_in_map, original_name_in_map in server_manager.sanitized_to_original.items():
                            if original_name_in_map == original_mcp_name:
                                llm_seen_name_for_this_ums_tool = sanitized_name_in_map
                                break
                    
                    if llm_seen_name_for_this_ums_tool:
                        ums_tools_schemas[llm_seen_name_for_this_ums_tool] = tool_obj.input_schema
                        self.logger.debug(f"Added UMS tool schema for LLM name: '{llm_seen_name_for_this_ums_tool}' (Original: '{original_mcp_name}')")
                    else:
                        # This case should be rare if _format_tools_for_provider ran correctly
                        self.logger.warning(f"Could not find LLM-seen name for UMS tool '{original_mcp_name}'. Schema not added for UMS structured output.")
                        
            if ums_tools_schemas:
                self.logger.info(f"Collected {len(ums_tools_schemas)} UMS tool schemas for structured output prompting.")
                return ums_tools_schemas
            else:
                self.logger.info("No UMS tool schemas found/matched for structured output prompting.")
                return None
            
        except Exception as e:
            self.logger.warning(f"Failed to build UMS tool schemas for structured output: {e}", exc_info=True)
            return None

    async def _rank_tools_for_goal(self, goal_desc: str, phase: Phase, limit: int = 15) -> List[Dict[str, Any]]:
        """Intelligently rank tools using MCPClient's built-in ranking."""
        try:
            return await self.mcp_client.rank_tools_for_goal(goal_desc, phase.value, limit)
        except Exception as e:
            self.logger.error(f"Error in tool ranking: {e}")
            # Fallback to basic tools
            return self.tool_schemas if self.tool_schemas else []
        
    async def _save_cognitive_state(self) -> None:
        """Save current agent state using UMS cognitive state tools."""
        if not self.state.workflow_id:
            self.logger.debug("No workflow_id available, skipping cognitive state saving")
            return
        
        # Don't try to save state if we haven't done any work yet
        if self.state.loop_count == 0:
            self.logger.debug("First loop, skipping cognitive state save until we have memories")
            return
            
        try:
            # First, check if we have any working memories to save
            mem_query_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("query_memories"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": "working",
                    "limit": 20,
                    "sort_by": "created_at",
                    "sort_order": "DESC"
                }
            )
            
            try:
                mem_query_data = await self._get_valid_ums_data_payload(mem_query_envelope, "query_memories_for_save")
                memories = mem_query_data.get("memories", [])
            except RuntimeError as e:
                self.logger.warning(f"Cannot query working memories for state save: {e}")
                return
                
            if not memories:
                self.logger.debug("No working memories yet, skipping state save")
                return
                
            working_memory_ids = [mem["memory_id"] for mem in memories if mem.get("memory_id")]
            
            # Get focus area IDs - memories with high importance
            focus_memory_ids = [
                mem["memory_id"] 
                for mem in memories 
                if mem.get("memory_id") and mem.get("importance", 0) >= 7.0
            ][:3]  # Limit to top 3
            
            # Get recent action IDs
            context_action_ids = []
            try:
                actions_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_recent_actions"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "limit": 10
                    }
                )
                
                actions_data = await self._get_valid_ums_data_payload(actions_envelope, "get_recent_actions_for_save")
                actions = actions_data.get("actions", [])
                context_action_ids = [
                    action.get("action_id") 
                    for action in actions 
                    if action.get("action_id")
                ]
            except Exception as e:
                self.logger.debug(f"Could not get recent action IDs: {e}")
            
            # Get current goal IDs for cognitive state
            current_goal_ids = []
            if self.state.current_leaf_goal_id:
                current_goal_ids.append(self.state.current_leaf_goal_id)
            # Also include root goal if it's different from current leaf goal
            if (self.state.root_goal_id and 
                self.state.root_goal_id != self.state.current_leaf_goal_id):
                current_goal_ids.append(self.state.root_goal_id)
            
            # Create title for the cognitive state
            title = f"Agent state at loop {self.state.loop_count} - Phase: {self.state.phase.value}"
            
            # Now save the cognitive state
            save_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("save_cognitive_state"),
                {
                    "workflow_id": self.state.workflow_id,
                    "title": title,
                    "working_memory_ids": working_memory_ids,
                    "focus_area_ids": focus_memory_ids,
                    "context_action_ids": context_action_ids,
                    "current_goals": current_goal_ids
                }
            )
            
            try:
                save_data = await self._get_valid_ums_data_payload(save_envelope, "save_cognitive_state")
                self.state.context_id = save_data.get("state_id")
                self.logger.debug(f"Saved cognitive state {self.state.context_id} for loop {self.state.loop_count}")
            except RuntimeError as e:
                self.logger.warning(f"Failed to save cognitive state: {e}")
                
        except Exception as e:
            self.logger.warning(f"Error during cognitive state save: {e}")
            # Don't raise - cognitive state saving is not critical for operation
        
        
    async def _create_checkpoint(self, reason: str) -> Optional[str]:
        """
        Create a checkpoint of current cognitive state for recovery.
        
        Parameters
        ----------
        reason : str
            Reason for creating the checkpoint
            
        Returns
        -------
        Optional[str]
            Checkpoint ID if successful, None otherwise
        """
        try:
            # Get current working memory
            working_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_working_memory"),
                {
                    "context_id": self.state.context_id,
                    "limit": 20
                }
            )
            
            working_memory_ids = []
            try:
                wm_data = await self._get_valid_ums_data_payload(working_memory_res, "get_working_memory_checkpoint")
                memories = wm_data.get("memories", [])
                working_memory_ids = [mem.get("memory_id") for mem in memories if mem.get("memory_id")]
            except RuntimeError as e:
                self.logger.warning(f"Failed to get working memory for checkpoint: {e}")
                # Continue with empty list
            
            # Create checkpoint memory
            checkpoint_content = {
                "checkpoint_reason": reason,
                "workflow_id": self.state.workflow_id,
                "current_goal_id": self.state.current_leaf_goal_id,
                "root_goal_id": self.state.root_goal_id,
                "phase": self.state.phase.value,
                "loop_count": self.state.loop_count,
                "working_memory_ids": working_memory_ids[:15],  # Limit to avoid huge checkpoints
                "created_at": int(time.time())
            }
            
            checkpoint_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "store_memory",
                {
                    "workflow_id": self.state.workflow_id,
                    "content": json.dumps(checkpoint_content, indent=2),
                    "memory_type": "checkpoint",
                    "memory_level": "episodic",  # Long-term storage
                    "importance": 9.0,  # High importance for recovery
                    "description": f"Cognitive checkpoint: {reason}"
                }
            )
            
            try:
                checkpoint_data = await self._get_valid_ums_data_payload(checkpoint_res, "store_memory_checkpoint")
                checkpoint_id = checkpoint_data.get("memory_id")
                if checkpoint_id:
                    self.logger.info(f"Created checkpoint {checkpoint_id[:8]} for reason: {reason}")
                    return checkpoint_id
            except RuntimeError as e:
                self.logger.warning(f"Failed to create checkpoint memory: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {e}")
        
        return None

    # ---------------------------------------------------------------- run-loop

    async def run_main_loop(self, overall_goal: str, max_mcp_loops: int) -> bool:
        """
        Execute one complete agent reasoning turn.
        
        Parameters
        ----------
        overall_goal : str
            The natural-language goal for this specific task activation.
        max_mcp_loops : int
            Maximum number of loops allowed for this activation (budget from MCPClient).
            
        Returns
        -------
        bool
            True if the agent should continue, False if it should stop/complete/fail.
        """
        self.logger.info(f"[AML] Executing agent turn for goal: {overall_goal[:100]}...")
        
        # --- Update loop count for this turn ---
        self.state.loop_count += 1
        loop_idx = self.state.loop_count
        self.logger.debug("==== TURN %s  | phase=%s ====", loop_idx, self.state.phase)

        # Initialize should_continue flag
        should_continue_agent = True

        try:
            # 0. Check for shutdown signal and budget/turn limits FIRST
            if self._shutdown_event.is_set():
                self.logger.info("[AML] Shutdown event set, stopping agent.")
                return False  # Agent should stop
            
            # Check for budget and turn limits at the beginning of the turn
            max_budget = getattr(self.mcp_client.config, 'max_budget_usd', 5.0)
            if self.state.cost_usd >= max_budget:
                self.logger.warning("[AML] Budget exceeded -> abort")
                self._record_failure("budget_exceeded")
                return False  # Agent should stop
                
            max_turns = getattr(self.mcp_client.config, 'max_agent_turns', 40)
            if loop_idx >= max_turns:
                self.logger.warning("[AML] Turn limit exceeded -> abort") 
                self._record_failure("turn_limit")
                return False  # Agent should stop
            
            # Check if overall goal is already achieved
            if self.state.goal_achieved_flag:
                self.logger.info("[AML] Goal achieved flag already set, stopping agent.")
                return False  # Agent should stop

            # 1. Ensure we have a valid workflow and root goal for this turn
            await self._ensure_workflow_and_goal(overall_goal)

            # 2. Finish/background cheap tasks -----------------------------------
            await self.async_queue.drain()

            # 3. Gather context for this turn ------------------------------------
            context = await self._gather_context()

            # 4. Maybe spawn new micro-tasks (runs in background) ----------------
            await self._maybe_spawn_fast_tasks(context)

            # 5. Build reasoning messages & refresh tool schemas for LLM call ----
            messages = self._build_messages(context)
            # Ensure we have fresh tool schemas directly before LLM call
            await self._refresh_tool_schemas() 
            tool_schemas = await self._get_tool_schemas()

            # Determine LLM model for this turn (budget-aware)
            if self.state.cost_usd >= max_budget * 0.8:  # Use cheaper model when near budget threshold
                model_name_for_llm_call = getattr(self.mcp_client.config, 'default_cheap_and_fast_model', self.default_llm_model)
            else:
                model_name_for_llm_call = self.default_llm_model
            
            self.logger.info(f"[AML] Calling LLM with model: {model_name_for_llm_call}, {len(tool_schemas) if tool_schemas else 0} tools")
            
            # 6. Call LLM for agent decision -------------------------------------
            # The LLM orchestrator handles structured output internally
            llm_decision = await self.llms.big_reasoning_call(
                messages, 
                tool_schemas, 
                model_name=model_name_for_llm_call
            )

            # 7. Enact decision & track progress ---------------------------------
            progress_made_this_turn = await self._enact(llm_decision)

            # 8. Metacognition & maintenance -------------------------------------
            await self.metacog.maybe_reflect(context)
            await self.metacog.assess_and_transition(progress_made_this_turn)

            # Create checkpoint at key decision points (after successful progress/metacognition)
            if self.state.phase == Phase.COMPLETE or self.state.goal_achieved_flag:
                await self._create_checkpoint("goal_completion")
            elif self.state.consecutive_error_count >= 2:  # After 2 consecutive errors
                await self._create_checkpoint("error_recovery_point")
            elif self.state.loop_count % 15 == 0:  # Periodic checkpoint every 15 turns
                await self._create_checkpoint("periodic_save")

            # 9. Persist state --------------------------------------------------
            await self._save_cognitive_state()

            # Reset error count on successful turn (no exceptions)
            self.state.consecutive_error_count = 0
            
            # Agent decides whether to continue
            if self.state.phase == Phase.COMPLETE or self.state.goal_achieved_flag:
                should_continue_agent = False  # Goal achieved or phase complete
            elif self.state.cost_usd >= max_budget:
                self.logger.warning("[AML] Budget exceeded, agent stopping.")
                should_continue_agent = False  # Budget exceeded
            elif loop_idx >= max_mcp_loops:
                self.logger.warning("[AML] Max turns exceeded, agent stopping.")
                should_continue_agent = False  # Turn limit exceeded
            else:
                should_continue_agent = True  # Continue to next turn

        except Exception as e:
            self.logger.error(f"Error in agent turn {loop_idx}: {e}", exc_info=True)
            self.state.last_error_details = {"error": str(e), "turn": loop_idx}
            self.state.consecutive_error_count += 1
            
            # Store error details in memory for learning
            if self.state.workflow_id:  # Check if workflow_id is set
                try:
                    await self._store_memory_with_auto_linking(
                        content=f"Agent error on turn {loop_idx}: {str(e)}. Will attempt recovery.",
                        memory_type="correction",
                        memory_level="working", 
                        importance=7.0,
                        description="Error details for debugging and recovery"
                    )
                except Exception as mem_store_err:
                    self.logger.error(f"CRITICAL: Failed to store error-related memory: {mem_store_err}", exc_info=True)
            else:
                self.logger.error(f"Agent error on turn {loop_idx} but no workflow_id available to store error memory. Original error: {str(e)}")
            
            # Fail after too many consecutive errors
            if self.state.consecutive_error_count >= 3:
                should_continue_agent = False  # Too many errors, agent stops
            else:
                should_continue_agent = True  # Attempt to continue despite error
        
        return should_continue_agent

    async def _ensure_workflow_and_goal(self, overall_goal: str) -> None:
        """Ensure we have a valid workflow and root goal, creating them if needed."""
        if not self.state.workflow_id:
            self.logger.info("Creating new workflow for goal")
            
            try:
                # 1. Create workflow
                wf_resp_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("create_workflow"),
                    {
                        "title": f"Agent Task – {overall_goal[:60]}",
                        "description": overall_goal,
                        "goal": overall_goal,  # This will also create an initial goal thought
                        "tags": ["agent-driven"],
                        "metadata": {
                            "agent_version": "1.0",
                            "created_by": "AgentMasterLoop",
                            "start_time": int(time.time())
                        }
                    },
                )
                # Use the new helper to get the actual UMS tool data
                actual_wf_payload = await self._get_valid_ums_data_payload(wf_resp_envelope, "create_workflow")
                
                self.state.workflow_id = actual_wf_payload.get("workflow_id")
                self.state.current_thought_chain_id = actual_wf_payload.get("primary_thought_chain_id")

                if not self.state.workflow_id:
                    # _get_valid_ums_data_payload would have raised if workflow_id was missing after success
                    # This is more of a defensive check for unexpected scenarios.
                    self.logger.error(f"create_workflow tool data payload missing 'workflow_id'. Payload: {actual_wf_payload}")
                    raise RuntimeError(f"create_workflow did not return a workflow_id in its data payload: {actual_wf_payload}")
                self.logger.info(f"Created workflow {self.state.workflow_id} with thought chain {self.state.current_thought_chain_id}")

                # 2. Create root goal using UMS create_goal
                goal_resp_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("create_goal"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "title": "Complete Agent Task: " + overall_goal[:50],  # More specific title
                        "description": overall_goal,
                        "priority": 1,  # Root goals are highest priority
                        "reasoning": "Primary root goal for the agent-driven task execution.",
                        "initial_status": "active",
                        "acceptance_criteria": [  # Generic criteria - will be refined based on actual goal
                            "Required research completed.",
                            "Key outputs produced.",
                            "Goal requirements satisfied."
                        ],
                        "metadata": {
                            "created_by": "AgentMasterLoop",
                            "is_root_goal": True
                        }
                    },
                )
                actual_goal_payload = await self._get_valid_ums_data_payload(goal_resp_envelope, "create_goal")
                
                goal_data_from_ums = actual_goal_payload.get("goal", {})  # UMS create_goal returns data under a "goal" key
                if not isinstance(goal_data_from_ums, dict) or not goal_data_from_ums.get("goal_id"):
                    self.logger.error(f"create_goal tool data payload missing 'goal' object or 'goal_id'. Payload: {actual_goal_payload}")
                    raise RuntimeError(f"create_goal did not return valid goal data: {actual_goal_payload}")
                
                self.state.root_goal_id = goal_data_from_ums["goal_id"]
                self.state.current_leaf_goal_id = self.state.root_goal_id
                self.state.goal_stack = [goal_data_from_ums]  # Initialize goal stack
                self.state.needs_replan = False  # A plan will be formed based on this new root goal
                self.logger.info(f"Created root goal {self.state.root_goal_id} for workflow {self.state.workflow_id}")
                
                # 3. Create initial memories and collect their IDs
                created_memory_ids = []
                
                # Initial observation memory
                initial_content = (
                    f"Starting work on goal: {overall_goal}\n\n"
                    f"This is the beginning of the agent's reasoning process. The goal will be approached "
                    f"systematically through research, planning, and execution phases.\n\n"
                    f"Phase: {self.state.phase.value}\n"
                    f"Loop: {self.state.loop_count}\n"
                    f"Workflow ID: {self.state.workflow_id}\n"
                    f"Root Goal ID: {self.state.root_goal_id}"
                )
                
                initial_memory_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": initial_content,
                        "memory_type": "observation",
                        "memory_level": "working",
                        "importance": 8.0,
                        "confidence": 1.0,
                        "description": "Initial context observation for bootstrap",
                        "suggest_links": False,
                        "generate_embedding": True,
                        "tags": ["bootstrap", "initialization"],
                        "context_data": {
                            "phase": self.state.phase.value,
                            "loop": self.state.loop_count
                        }
                    }
                )
                
                initial_memory_data = await self._get_valid_ums_data_payload(initial_memory_envelope, "store_memory_initial_observation")
                if initial_memory_data.get("memory_id"):
                    created_memory_ids.append(initial_memory_data["memory_id"])
                    self.logger.info(f"Created initial observation memory: {initial_memory_data['memory_id']}")
                
                # Goal-focused memory
                goal_content = (
                    f"PRIMARY GOAL: {overall_goal}\n\n"
                    f"This is the main objective that needs to be accomplished. The agent should focus all "
                    f"efforts on achieving this goal through systematic analysis, planning, and execution.\n\n"
                    f"Approach:\n"
                    f"1. Understand the requirements and context\n"
                    f"2. Plan the necessary steps and deliverables\n"
                    f"3. Execute the plan systematically\n"
                    f"4. Review and validate the results\n\n"
                    f"Success criteria: All requirements satisfied with high quality output."
                )
                
                goal_memory_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": goal_content,
                        "memory_type": "plan",
                        "memory_level": "working",
                        "importance": 10.0,
                        "confidence": 1.0,
                        "description": "Primary goal definition and success criteria",
                        "suggest_links": False,
                        "generate_embedding": True,
                        "tags": ["goal", "primary", "deliverables"],
                        "action_id": None,
                        "thought_id": None,
                        "artifact_id": None
                    }
                )
                
                goal_memory_data = await self._get_valid_ums_data_payload(goal_memory_envelope, "store_memory_goal_focused")
                if goal_memory_data.get("memory_id"):
                    created_memory_ids.append(goal_memory_data["memory_id"])
                    self.logger.info(f"Created goal memory: {goal_memory_data['memory_id']}")
                
                # Planning approach memory
                planning_content = (
                    f"INITIAL PLANNING APPROACH for: {overall_goal[:100]}\n\n"
                    f"Phase-based execution strategy:\n"
                    f"1. UNDERSTAND Phase: Analyze requirements and gather information\n"
                    f"   - Research relevant topics and context\n"
                    f"   - Identify key resources and constraints\n"
                    f"   - Store findings and insights as memories\n\n"
                    f"2. PLAN Phase: Structure approach and create roadmap\n"
                    f"   - Break down goal into actionable sub-tasks\n"
                    f"   - Identify required deliverables and outputs\n"
                    f"   - Plan implementation sequence\n\n"
                    f"3. EXECUTE Phase: Implement the planned solution\n"
                    f"   - Create required outputs and deliverables\n"
                    f"   - Use appropriate tools for task completion\n"
                    f"   - Validate outputs against requirements\n\n"
                    f"Current status: Starting in {self.state.phase.value} phase."
                )
                
                planning_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": planning_content,
                        "memory_type": "plan",
                        "memory_level": "working",
                        "importance": 9.0,
                        "confidence": 1.0,
                        "description": "Initial planning approach and phase strategy",
                        "suggest_links": True,
                        "generate_embedding": True,
                        "tags": ["planning", "strategy", "phases"],
                        "link_suggestion_threshold": 0.7
                    }
                )
                
                planning_data = await self._get_valid_ums_data_payload(planning_envelope, "store_memory_planning_approach")
                if planning_data.get("memory_id"):
                    created_memory_ids.append(planning_data["memory_id"])
                    self.logger.info(f"Created planning memory: {planning_data['memory_id']}")
                
                # Context initialization memory
                context_memory_content = (
                    f"CONTEXT INITIALIZATION at {_dt.datetime.utcnow().isoformat()}\n\n"
                    f"Agent system initialized with:\n"
                    f"- Workflow ID: {self.state.workflow_id}\n"
                    f"- Root Goal ID: {self.state.root_goal_id}\n"
                    f"- Available tools: 128+ MCP tools\n"
                    f"- Memory system: UMS with working memory\n"
                    f"- Phase: {self.state.phase.value}\n\n"
                    f"Ready to begin systematic task execution."
                )
                
                context_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": context_memory_content,
                        "memory_type": "context_initialization",
                        "memory_level": "working",
                        "importance": 7.0,
                        "confidence": 1.0,
                        "description": "System context at initialization",
                        "suggest_links": False,
                        "generate_embedding": True,
                        "tags": ["context", "initialization", "system"],
                        "ttl": 0  # No expiration
                    }
                )
                
                context_data = await self._get_valid_ums_data_payload(context_envelope, "store_memory_context_init")
                if context_data.get("memory_id"):
                    created_memory_ids.append(context_data["memory_id"])
                    self.logger.info(f"Created context memory: {context_data['memory_id']}")
                
                # 4. Create cognitive state WITH the memory IDs and the CURRENT GOAL ID
                if not created_memory_ids:
                    self.logger.error("No memories were successfully created for bootstrap!")
                    raise RuntimeError("Failed to create bootstrap memories")
                
                
                self.logger.info(f"Creating cognitive state with {len(created_memory_ids)} bootstrap memories and root goal.")
                
                cognitive_state_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("save_cognitive_state"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "title": f"Initial cognitive state for: {overall_goal[:50]}",
                        "working_memory_ids": created_memory_ids,
                        "focus_area_ids": [created_memory_ids[1]] if len(created_memory_ids) > 1 else created_memory_ids[:1],
                        "context_action_ids": [],
                        "current_goal_thought_ids": []
                    }
                )
                
                try:
                    cognitive_state_data = await self._get_valid_ums_data_payload(cognitive_state_envelope, "save_cognitive_state_initial")
                    self.state.context_id = cognitive_state_data.get("state_id")
                    self.logger.info(f"Created initial cognitive state: {self.state.context_id}")
                except RuntimeError as e:
                    self.logger.error(f"Failed to create cognitive state: {e}")
                    raise RuntimeError("Failed to create initial cognitive state") from e
                
                # 5. Verify working memory is populated
                verify_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_working_memory"),
                    {
                        "context_id": self.state.context_id,
                        "include_content": False,
                        "update_access": False,
                        "limit": 20
                    }
                )
                
                try:
                    verify_data = await self._get_valid_ums_data_payload(verify_envelope, "get_working_memory_verify")
                    working_memories = verify_data.get("working_memories", [])
                    if len(working_memories) > 0:
                        self.logger.info(f"Working memory successfully populated with {len(working_memories)} memories")
                        
                        # Log memory types for debugging
                        memory_types = {}
                        for mem in working_memories:
                            mem_type = mem.get("memory_type", "unknown")
                            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                        self.logger.info(f"Working memory contains: {memory_types}")
                    else:
                        self.logger.error("Working memory verification failed: No memories found after initialization")
                        raise RuntimeError("Working memory not properly initialized")
                except RuntimeError as e:
                    self.logger.error(f"Failed to verify working memory: {e}")
                    raise RuntimeError("Could not verify working memory state") from e
                
                self.logger.info(f"Successfully created and initialized workflow {self.state.workflow_id} with root goal {self.state.root_goal_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to create workflow and goal: {e}")
                # Reset state on failure
                self.state.workflow_id = None
                self.state.root_goal_id = None
                self.state.current_leaf_goal_id = None
                self.state.context_id = None
                raise
                
        else:
            # We have a workflow_id - try to load previous cognitive state
            self.logger.info("Workflow exists, attempting to load previous cognitive state")
            await self._load_cognitive_state()
            
            if self.state.needs_replan:
                self.logger.info("Workflow exists but needs replanning")
                self.state.needs_replan = False

        # Create goal if workflow exists but no root goal
        if self.state.workflow_id and not self.state.root_goal_id:
            self.logger.info("Workflow exists but no root goal - creating root goal")
            await self._create_root_goal_only(overall_goal)

        # Ensure we have a current leaf goal if we have a root goal
        if self.state.root_goal_id and not self.state.current_leaf_goal_id:
            self.state.current_leaf_goal_id = self.state.root_goal_id
            self.logger.info(f"Set current_leaf_goal_id to root_goal_id: {self.state.root_goal_id}")

        # Verify the goal was actually created and is accessible
        if self.state.root_goal_id:
            try:
                goal_resp_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_goal_details"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "goal_id": self.state.root_goal_id
                    }
                )
                
                try:
                    goal_data_payload = await self._get_valid_ums_data_payload(goal_resp_envelope, "get_goal_details_verify")
                    goal_data_envelope = goal_data_payload  # For compatibility with original logic
                    if isinstance(goal_data_envelope, dict) and isinstance(goal_data_envelope.get("goal"), dict):
                        goal_actual_data = goal_data_envelope["goal"]  # Access the actual goal data
                        self.logger.info(f"Root goal verified: {goal_actual_data.get('title', 'Unknown Goal Title')} - {goal_actual_data.get('description', 'No Goal Description')[:100]}")
                    else:
                        self.logger.error(f"Root goal verification returned unexpected data structure: {str(goal_data_envelope)[:200]}")
                        # Reset goal IDs so they get recreated
                        self.state.root_goal_id = None
                        self.state.current_leaf_goal_id = None
                        # Try to recreate the goal
                        await self._create_root_goal_only(overall_goal)
                except RuntimeError as e:
                    self.logger.error(f"Root goal verification failed: {e}")
                    # Reset goal IDs so they get recreated
                    self.state.root_goal_id = None
                    self.state.current_leaf_goal_id = None
                    # Try to recreate the goal
                    await self._create_root_goal_only(overall_goal)
            except Exception as e:
                self.logger.error(f"Error verifying root goal: {e}")
                self.state.root_goal_id = None
                self.state.current_leaf_goal_id = None
        else:
            self.logger.error("No root goal ID available - workflow/goal creation may have failed")
            
        # Final validation - ensure we have both workflow and goal
        if not self.state.workflow_id:
            raise RuntimeError("Failed to create or load workflow_id")
        if not self.state.root_goal_id:
            raise RuntimeError("Failed to create or load root_goal_id")



    async def _create_root_goal_only(self, overall_goal: str) -> None:
        """Create only the root goal for an existing workflow."""
        try:
            goal_resp_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("create_goal"),
                {
                    "workflow_id": self.state.workflow_id,
                    "title": "Complete Agent Task",
                    "description": overall_goal,
                    "initial_status": "active",
                    "priority": 1,
                },
            )
            
            goal_payload = await self._get_valid_ums_data_payload(goal_resp_envelope, "create_goal_root_only")
            goal_data = goal_payload.get("goal", {})
            if not goal_data.get("goal_id"):
                raise RuntimeError(f"Root goal creation succeeded but no goal data returned: {goal_payload}")

            # Create a context if we don't have one
            if not self.state.context_id:
                context_resp_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("save_cognitive_state"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "title": f"Agent Working Memory - {overall_goal[:50]}",
                        "working_memory_ids": [],
                        "focus_area_ids": [],
                        "context_action_ids": [],
                        "current_goals": []
                    }
                )
                
                try:
                    context_data = await self._get_valid_ums_data_payload(context_resp_envelope, "save_cognitive_state_context_create")
                    self.state.context_id = context_data.get("state_id")
                except RuntimeError as e:
                    self.logger.warning(f"Failed to create context (non-critical): {e}")
                
            self.state.root_goal_id = goal_data["goal_id"]
            self.state.current_leaf_goal_id = self.state.root_goal_id
            self.state.needs_replan = False
            
            # Create a memory about this goal
            try:
                store_memory_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": f"Created primary goal: {overall_goal}. Goal ID: {self.state.root_goal_id}",
                        "memory_type": "plan",
                        "memory_level": "working",
                        "importance": 9.0,
                        "description": "Primary goal for this agent session",
                        "metadata": {"goal_id": self.state.root_goal_id}
                    }
                )
                memory_data = await self._get_valid_ums_data_payload(store_memory_envelope, "store_memory_root_goal")
                self.logger.debug(f"Created goal memory: {memory_data.get('memory_id', 'unknown')}")
            except Exception as mem_e:
                self.logger.warning(f"Failed to create goal memory (non-critical): {mem_e}")
            
            self.logger.info(f"Created root goal {self.state.root_goal_id} for existing workflow {self.state.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create root goal: {e}")
            self.state.last_error_details = {"error": str(e), "context": "root_goal_creation"}
            self.state.consecutive_error_count += 1
            raise

    # -------------------------------------------------------------- single turn



    # -------------------------------------------------- helper: gather context

    async def _gather_context(self) -> Dict[str, Any]:
        """Collects all information using the rich context package tool."""
        
        if not self.state.workflow_id:
            # This should never happen if we're called after _ensure_workflow_and_goal
            raise RuntimeError("Cannot gather context without workflow_id")
        
        try:
            # First, ensure we have a valid context_id
            if not self.state.context_id:
                self.logger.warning("No context_id available, creating new cognitive state")
                
                # Create an initial cognitive state
                context_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("save_cognitive_state"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "title": f"Agent context at loop {self.state.loop_count}",
                        "working_memory_ids": [],  # Will populate below
                        "focus_area_ids": [],
                        "context_action_ids": [],
                        "current_goals": [self.state.current_leaf_goal_id] if self.state.current_leaf_goal_id else []
                    }
                )
                
                try:
                    context_data = await self._get_valid_ums_data_payload(context_res, "save_cognitive_state_initial_context")
                    self.state.context_id = context_data.get("state_id")
                    self.logger.info(f"Created new context: {self.state.context_id}")
                except RuntimeError as e:
                    raise RuntimeError(f"Failed to create initial context: {e}") from e
            
            # Check if we have any working memory
            working_memory_check = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_working_memory"),
                {
                    "context_id": self.state.context_id,
                    "include_content": False,
                    "update_access": False
                }
            )
            
            has_working_memory = (
                working_memory_check.get("success") and 
                working_memory_check.get("data", {}).get("working_memories") and
                len(working_memory_check["data"]["working_memories"]) > 0
            )
            
            if not has_working_memory:
                self.logger.info("No working memory found, creating and populating initial memories")
                
                created_memory_ids = []
                
                # Create initial observation memory about the goal
                goal_desc = await self._get_current_goal_description()
                initial_content = (
                    f"Current task: {goal_desc}\n\n"
                    f"Phase: {self.state.phase.value}\n"
                    f"Loop: {self.state.loop_count}\n"
                    f"This is the agent's current working context. The goal needs to be completed through systematic research, planning, and execution."
                )
                
                initial_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": initial_content,
                        "memory_type": "observation",
                        "memory_level": "working",
                        "importance": 8.0,
                        "description": "Current task context",
                        "suggest_links": False,
                        "generate_embedding": True
                    }
                )
                
                try:
                    initial_memory_data = await self._get_valid_ums_data_payload(initial_memory_res, "store_memory_initial_context")
                    memory_id = initial_memory_data.get("memory_id")
                    if memory_id:
                        created_memory_ids.append(memory_id)
                        self.logger.debug(f"Created initial context memory: {memory_id}")
                except RuntimeError as e:
                    self.logger.warning(f"Failed to create initial context memory: {e}")
                
                # Create a planning memory
                planning_content = (
                    f"Agent planning for: {goal_desc[:200]}\n\n"
                    f"Approach:\n"
                    f"1. Research and gather information\n"
                    f"2. Analyze and synthesize findings\n"
                    f"3. Create required outputs\n"
                    f"4. Validate results\n\n"
                    f"Current focus: {self.state.phase.value} phase"
                )
                
                planning_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": planning_content,
                        "memory_type": "plan",
                        "memory_level": "working",
                        "importance": 7.5,
                        "description": "Agent planning approach",
                        "suggest_links": False,
                        "generate_embedding": True
                    }
                )
                
                try:
                    planning_memory_data = await self._get_valid_ums_data_payload(planning_memory_res, "store_memory_planning_context")
                    memory_id = planning_memory_data.get("memory_id")
                    if memory_id:
                        created_memory_ids.append(memory_id)
                        self.logger.debug(f"Created planning memory: {memory_id}")
                except RuntimeError as e:
                    self.logger.warning(f"Failed to create planning memory: {e}")
                
                # Create current state memory
                state_content = (
                    f"Agent state at loop {self.state.loop_count}:\n"
                    f"- Phase: {self.state.phase.value}\n"
                    f"- Stuck counter: {self.state.stuck_counter}\n"
                    f"- Consecutive errors: {self.state.consecutive_error_count}\n"
                    f"- Last action: {self.state.last_action_summary or 'Starting'}\n"
                    f"- Needs replan: {self.state.needs_replan}\n"
                    f"Agent is actively working on the task."
                )
                
                state_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": state_content,
                        "memory_type": "state_snapshot",
                        "memory_level": "working",
                        "importance": 6.5,
                        "description": "Current agent state",
                        "suggest_links": False,
                        "generate_embedding": True
                    }
                )
                
                try:
                    state_memory_data = await self._get_valid_ums_data_payload(state_memory_res, "store_memory_state_context")
                    memory_id = state_memory_data.get("memory_id")
                    if memory_id:
                        created_memory_ids.append(memory_id)
                        self.logger.debug(f"Created state memory: {memory_id}")
                except RuntimeError as e:
                    self.logger.warning(f"Failed to create state memory: {e}")
                
                # Now update the cognitive state with these memory IDs
                if created_memory_ids:
                    update_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("save_cognitive_state"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "title": f"Populated context at loop {self.state.loop_count}",
                            "working_memory_ids": created_memory_ids,
                            "focus_area_ids": created_memory_ids[:1],  # First memory as focus
                            "context_action_ids": [],
                            "current_goals": [self.state.current_leaf_goal_id] if self.state.current_leaf_goal_id else []
                        }
                    )
                    
                    try:
                        update_data = await self._get_valid_ums_data_payload(update_res, "save_cognitive_state_context_update")
                        self.state.context_id = update_data.get("state_id")
                        self.logger.info(f"Updated context with {len(created_memory_ids)} initial memories")
                    except RuntimeError as e:
                        self.logger.warning(f"Failed to update context with initial memories: {e}")
                    
                    # If we have a focal memory hint, set it
                    if created_memory_ids:
                        try:
                            focus_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                self._get_ums_tool_name("focus_memory"),
                                {
                                    "memory_id": created_memory_ids[0],
                                    "context_id": self.state.context_id,
                                    "add_to_working": False  # Already in working memory
                                }
                            )
                            try:
                                await self._get_valid_ums_data_payload(focus_res, "focus_memory_context")
                                self.logger.debug(f"Set focal memory to {created_memory_ids[0][:8]}")
                            except RuntimeError as e:
                                self.logger.debug(f"Could not set focal memory (non-critical): {e}")
                        except Exception as e:
                            self.logger.debug(f"Could not set focal memory (non-critical): {e}")
            
            # Now gather the rich context package
            params = {
                "workflow_id": self.state.workflow_id,
                "context_id": self.state.context_id,
                "current_plan_step_description": (
                    self.state.current_plan[0].get("description") 
                    if self.state.current_plan and len(self.state.current_plan) > 0
                    else f"Working on {self.state.phase.value} phase"
                ),
                "focal_memory_id_hint": None,
                "fetch_limits": {
                    "recent_actions": 10,
                    "important_memories": 15,
                    "key_thoughts": 8,
                    "proactive_memories": 5,
                    "procedural_memories": 3,
                    "link_traversal": 3,
                },
                "show_limits": {
                    "working_memory": 15,
                    "link_traversal": 5,
                },
                "include_core_context": True,
                "include_working_memory": True,
                "include_proactive_memories": True,
                "include_relevant_procedures": True,
                "include_contextual_links": True,
                "include_graph": True,
                "include_recent_actions": True,
                "include_goal_stack": True, # NEW: Request goal stack
                "include_contradictions": True,
                "max_memories": 20,
                "compression_token_threshold": 16000,
                "compression_target_tokens": 6000
            }
            
            self.logger.debug(f"Calling get_rich_context_package with context_id: {self.state.context_id}")
            
            rich_res_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_rich_context_package"),
                params
            )
            
            try:
                ums_tool_data_payload = await self._get_valid_ums_data_payload(rich_res_envelope, "get_rich_context_package")
                context_package = ums_tool_data_payload.get("context_package", {})  # get_rich_context_package nests its result one more time
                if not context_package or not context_package.get("core_context"):
                    self.logger.error(
                        f"Rich context package from UMS is empty or missing core_context. "
                        f"UMS Tool Payload: {str(ums_tool_data_payload)[:500]}"
                    )
                    raise RuntimeError("UMS get_rich_context_package returned empty or invalid context_package.")
            except RuntimeError as e:
                self.logger.error(f"Rich context package call failed or UMS tool reported internal errors: {e}")
                
                # Fallback: try to get basic working memory at least
                try:
                    fallback_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("get_working_memory"),
                        {
                            "context_id": self.state.context_id,
                            "include_content": True,
                            "update_access": True
                        }
                    )
                    fallback_data = await self._get_valid_ums_data_payload(fallback_envelope, "get_working_memory_fallback")
                    fallback_working_memory = fallback_data
                except Exception:
                    fallback_working_memory = {"working_memories": [], "error": "Fallback failed"}
                
                # Create a minimal fallback package
                minimal_package = {
                    "retrieval_timestamp_ums_package": _dt.datetime.utcnow().isoformat(),
                    "core_context": {
                        "workflow_id": self.state.workflow_id,
                        "workflow_goal": await self._get_current_goal_description(),
                        "workflow_status": "active",
                        "error_in_context": True
                    },
                    "current_working_memory": fallback_working_memory,
                    "recent_actions": [],
                    "error_details_from_ums": str(e)  # Include the error details
                }
                return {
                    "rich_context_package": minimal_package,
                    "contradictions": [],
                    "has_contradictions": False,
                    "context_retrieval_timestamp": minimal_package["retrieval_timestamp_ums_package"],
                    "context_sources": {"rich_package": False, "fallback_used": True, "error": True}
                }
            
            # Log what we actually got for debugging
            working_memory = context_package.get("current_working_memory", {})
            working_memories = working_memory.get("working_memories", []) if isinstance(working_memory, dict) else []
            recent_actions = context_package.get("recent_actions", [])
            core_context = context_package.get("core_context", {})
            graph_snapshot = context_package.get("graph_snapshot", {})
            contradictions_data = context_package.get("contradictions", {})
            
            self.logger.info(
                f"Rich context retrieved: "
                f"working_memory={len(working_memories)} items, "
                f"recent_actions={len(recent_actions) if isinstance(recent_actions, list) else 0} items, "
                f"core_context={bool(core_context)}, "
                f"graph_snapshot={bool(graph_snapshot)}"
            )
            
                        # Extract contradictions for metacognition
            contradictions = []
            if isinstance(contradictions_data, dict):
                contradictions = contradictions_data.get("contradictions_found", [])

            # Return the rich context with minimal transformation
            return {
                "rich_context_package": context_package,
                "contradictions": contradictions,
                "has_contradictions": len(contradictions) > 0,
                "context_retrieval_timestamp": context_package.get("retrieval_timestamp_ums_package", _dt.datetime.utcnow().isoformat()),
                "context_sources": {
                    "rich_package": True,
                    "compression_applied": "ums_compression_details" in context_package,
                    "fallback_used": False
                }
            }
                
        except Exception as e:
            self.logger.error(f"Failed to get rich context package: {e}", exc_info=True)
            
            # Last resort fallback - return minimal context
            return {
                "rich_context_package": {
                    "retrieval_timestamp_ums_package": _dt.datetime.utcnow().isoformat(),
                    "core_context": {
                        "workflow_id": self.state.workflow_id,
                        "workflow_goal": "Task in progress",
                        "workflow_status": "active"
                    },
                    "current_working_memory": {},
                    "recent_actions": [],
                    "error": str(e)
                },
                "contradictions": [],
                "has_contradictions": False,
                "context_retrieval_timestamp": _dt.datetime.utcnow().isoformat(),
                "context_sources": {
                    "rich_package": False,
                    "compression_applied": False,
                    "fallback_used": True,
                    "error": True
                }
            }









    # -------------------------------- helper: spawn background fast tasks

    async def _maybe_spawn_fast_tasks(self, ctx: Dict[str, Any]) -> None:
        """
        Fire-and-forget cheap-LLM micro-tasks using rich context package data.
        REVISED: Now properly adds created memories to working memory.
        """
        # Extract rich context package
        rich_package = ctx.get("rich_context_package")
        if not rich_package:
            return
        
        ##########################################################################
        # 1) Handle contradictions if detected
        ##########################################################################
        contradictions = ctx.get('contradictions', [])
        for pair in contradictions[:2]:
            if len(pair) >= 2:
                a_id, b_id = pair[0], pair[1]
                prompt = (
                    "You are an analyst spotting inconsistent facts.\n"
                    "Summarise the contradiction **concisely** and propose ONE clarifying "
                    "question that, if answered, would resolve the conflict.\n\n"
                    f"Memory A ID: {a_id}\n"
                    f"Memory B ID: {b_id}\n"
                    "Focus on the logical inconsistency."
                )
                schema = {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "question": {"type": "string"},
                    },
                    "required": ["summary", "question"],
                    "additionalProperties": False,
                    "strict": True
                }

                async def _on_contradiction(res: Dict[str, str], aid: str = a_id, bid: str = b_id) -> None:
                    try:
                        if res is None or not isinstance(res, dict):
                            self.logger.warning("Fast LLM call returned invalid result for contradiction analysis")
                            return
                        
                        summary = res.get('summary', 'Contradiction detected')
                        question = res.get('question', 'What additional information is needed?')
                        
                        # Store memory and add to working memory
                        await self._store_memory_with_auto_linking(
                            content=f"{summary}\n\nCLARIFY: {question}",
                            memory_type="contradiction_analysis",
                            memory_level="working",
                            importance=7.0,
                            description="Automated contradiction analysis",
                            link_to_goal=True
                        )
                        # Memory is automatically added to working memory by the revised method
                    except Exception as e:
                        self.logger.warning(f"Error processing contradiction analysis: {e}")
                
                coro = self.llms.fast_structured_call(prompt, schema)
                task_name = f"contradict_{a_id[:4]}_{b_id[:4]}"
                self.async_queue.spawn(AsyncTask(task_name, coro, callback=_on_contradiction))

        ##########################################################################
        # 2) Proactive insight generation from working memory
        ##########################################################################
        working_memory = rich_package.get("current_working_memory", {})
        working_memories = working_memory.get("working_memories", [])
        
        if len(working_memories) >= 3:
            # Extract content more robustly from working memory structure
            memory_contents = []
            for mem in working_memories[-5:]:  # Last 5 memories
                # Try multiple possible content fields
                content = (
                    mem.get("content") or 
                    mem.get("content_preview") or 
                    mem.get("description", "") or
                    f"Memory type: {mem.get('memory_type', 'unknown')}, importance: {mem.get('importance', 0)}"
                )
                if content and len(content.strip()) > 0:
                    memory_contents.append(content[:500])  # Increased from 300 to 500 chars
                    
            # Only proceed if we have meaningful content
            if len(memory_contents) >= 2 and all(len(content.strip()) > 10 for content in memory_contents):
                prompt = (
                    "Analyze these recent working memories and generate ONE key insight, "
                    "pattern, or strategic observation that could guide next actions.\n\n"
                    + "\n".join(f"Memory {i+1}: {content}" for i, content in enumerate(memory_contents))
                    + "\n\nProvide a concise insight (max 100 words)."
                )
                schema = {
                    "type": "object",
                    "properties": {"insight": {"type": "string"}},
                    "required": ["insight"],
                    "additionalProperties": False,
                    "strict": True
                }

                async def _on_insight(res: Dict[str, str]) -> None:
                    try:
                        if res is None or not isinstance(res, dict) or "insight" not in res:
                            self.logger.warning("Fast LLM call returned invalid result for insight generation")
                            return
                        
                        insight_content = res.get("insight", "").strip()
                        if not insight_content:
                            self.logger.warning("Fast LLM returned empty insight")
                            return
            
                        # Store and add to working memory
                        mem_id = await self._store_memory_with_auto_linking(
                            content=insight_content,
                            memory_type="insight",
                            memory_level="working",
                            importance=6.5,
                            description="Proactive insight from working memory analysis",
                            link_to_goal=True
                        )
                        
                        # Make it focal since it's a new insight
                        if mem_id and self.state.context_id:
                            await self._add_to_working_memory(mem_id, make_focal=True)
                    except Exception as e:
                        self.logger.warning(f"Error processing insight: {e}")
                
                coro = self.llms.fast_structured_call(prompt, schema)
                self.async_queue.spawn(AsyncTask("working_memory_insight", coro, callback=_on_insight))
            else:
                self.logger.debug(f"Insufficient working memory content for insight generation: {len(memory_contents)} memories with meaningful content")

    # -------------------------------- helper: build SMART-model prompt

    def _build_messages(self, ctx: Dict[str, Any]) -> List[Dict[str, str]]:
        """Compose chat-completion messages with clear action guidance."""
        
        # Phase-specific instructions
        phase_instructions = {
            Phase.UNDERSTAND: "Analyze the goal and search for relevant information. Use web_search if needed.",
            Phase.PLAN: "Break down the goal into concrete sub-tasks. Create specific, actionable goals.",
            Phase.EXECUTE: "Execute the planned tasks using appropriate tools. Create artifacts as needed.",
            Phase.REVIEW: "Review what has been accomplished and verify goal completion.",
        }
        
        sys_msg = f"""You are an autonomous agent with access to tools and a memory system.

Current Phase: {self.state.phase.value} - {phase_instructions.get(self.state.phase, "Process the current task")}

IMPORTANT: You must make concrete progress each turn by:
1. Using tools to gather information (browse web content, read files, query memories)
2. Creating tangible outputs (write files, record artifacts, execute code)
3. Storing key findings in memory (store memories, record thoughts)
4. Breaking down complex goals (create goals)

CRITICAL TOOL USAGE RULES:
- You MUST use ONLY the exact tool names provided in the function schema
- Tool names have server prefixes like "Ultimate_MCP_Server_browse" 
- NEVER invent tool names like "agent:update_plan" - they don't exist
- Check the available tools list carefully before selecting

EFFICIENCY: Use TOOL_MULTIPLE when possible! Use the EXACT tool names from the schema provided.

Response Format - EXACTLY this JSON structure:

For SINGLE tool:
{{
    "decision_type": "TOOL_SINGLE",
    "tool_name": "exact_tool_name_from_schema",
    "tool_args": {{...}}
}}

For MULTIPLE tools (PREFERRED when possible):
{{
    "decision_type": "TOOL_MULTIPLE", 
    "tool_calls": [
        {{"tool_name": "exact_tool_name_from_schema", "tool_args": {{"param": "value"}}, "tool_id": "unique_id_1"}},
        {{"tool_name": "another_exact_tool_name", "tool_args": {{"param": "value"}}, "tool_id": "unique_id_2"}}
    ]
}}

For thinking:
{{
    "decision_type": "THOUGHT_PROCESS",
    "content": "reasoning here"
}}

For completion:
{{
    "decision_type": "DONE", 
    "content": "completion summary"
}}

CRITICAL: Only use tool names that appear in the provided tool schemas. Do not use abbreviated or base names.
Avoid circular reasoning. Each action should move closer to the goal."""

        # Use rich context package if available
        rich_package = ctx.get("rich_context_package")
        if rich_package:
            core_context = rich_package.get("core_context", {})
            recent_actions = rich_package.get("recent_actions", [])
            working_memory = rich_package.get("current_working_memory", {})
            
            # Format recent actions concisely
            if recent_actions:
                actions_text = "\n".join([
                    f"- {action.get('action_type', 'unknown')}: {action.get('title', 'No title')[:50]} ({action.get('status', 'unknown')})"
                    for action in recent_actions[-3:]  # Only last 3 for brevity
                ])
            else:
                actions_text = "No recent actions"
            
            # Format working memory
            memory_summary = "No working memory available"
            if working_memory.get("working_memories"):
                memory_count = len(working_memory["working_memories"])
                memory_summary = f"{memory_count} active working memories with recent insights"
            
            user_msg = (
                f"**Phase**: {self.state.phase.value}\n"
                f"**Goal**: {core_context.get('workflow_goal', 'No goal set')}\n"
                f"**Last 3 actions**:\n{actions_text}\n"
                f"**Working memory**: {memory_summary}\n\n"
            )
            
            if ctx.get("has_contradictions"):
                user_msg += "⚠️ **Contradictions detected** - address these or work around them\n\n"
            
            # NEW: Add goal stack info if available
            goal_stack_info = rich_package.get("goal_stack", {})
            if goal_stack_info.get("goal_tree"):
                user_msg += "**Current Goals (Tree View):**\n"
                
                # Recursive function to format goal tree
                def format_goal_node(node, indent_level=0):
                    indent = "  " * indent_level
                    status_emoji = "✅" if node.get("status") == "completed" else "⏳" if node.get("status") == "in_progress" else "🗓️"
                    title = node.get('title', node.get('description', 'Untitled Goal'))
                    display_title = title[:80] + ('...' if len(title) > 80 else '')
                    line = f"{indent}{status_emoji} {display_title} (ID: {node.get('goal_id', '')[:8]}, Status: {node.get('status', 'N/A')})\n"
                    for child in node.get("children", []):
                        line += format_goal_node(child, indent_level + 1)
                    return line

                for root_goal in goal_stack_info["goal_tree"]:
                    user_msg += format_goal_node(root_goal)
                user_msg += "\n"
            elif goal_stack_info.get("total_goals", 0) > 0:
                user_msg += f"**Goals Found:** {goal_stack_info['total_goals']} goals (details omitted due to size/filter)\n\n"
            
            # Add phase-specific guidance
            if self.state.phase == Phase.UNDERSTAND:
                user_msg += "Focus: Research and gather information about your goal. Use available tools to browse web content and store findings."
            elif self.state.phase == Phase.PLAN:
                user_msg += "Focus: Break down your goal into concrete, actionable sub-tasks. Create goals and store planning information."
            elif self.state.phase == Phase.EXECUTE:
                user_msg += "Focus: Create the required outputs and deliverables. Use tools to write files, execute code, and record artifacts."
            elif self.state.phase == Phase.REVIEW:
                user_msg += "Focus: Verify all outputs exist and meet requirements. Use tools to check artifacts and generate analysis."
            
            user_msg += "\n\nWhat specific action will you take next?"
        else:
            # Fallback for when rich context is unavailable
            user_msg = (
                f"**Phase**: {self.state.phase.value}\n"
                f"**Loop**: {self.state.loop_count}\n"
                "Context unavailable - proceed with basic reasoning.\n"
                "What specific action will you take next?"
            )
        
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

    # -------------------------------- helper: get tool-schemas for SMART model

    async def _get_current_goal_description(self) -> str:
        """Helper to get current goal description."""
        if not self.state.current_leaf_goal_id:
            return "General task"
        
        try:
            goal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_goal_details"),
                {"workflow_id": self.state.workflow_id, "goal_id": self.state.current_leaf_goal_id}
            )
            try:
                goal_data = await self._get_valid_ums_data_payload(goal_res, "get_goal_details_current_description")
                if goal_data.get("goal"):
                    return goal_data["goal"].get("description", "Task")
            except RuntimeError:
                pass  # Fall through to return "Task"
        except Exception:
            pass
        
        return "Task"
        
    async def _get_tool_schemas(self) -> Optional[List[Dict[str, Any]]]:
        """Return JSON-schema list of the most relevant tools."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            
            console = Console()
            
            if not self.tool_schemas:
                with console.capture() as capture:
                    console.print(Panel(
                        Text("❌ NO TOOL SCHEMAS AVAILABLE AT ALL!", style="bold red"),
                        title="🔧 Tool Schema Status",
                        border_style="red"
                    ))
                self.logger.error(f"Tool schema error:\n{capture.get()}")
                return []
            
            # Log basic status
            with console.capture() as capture:
                console.print(Panel(
                    Text(f"📊 Total available: {len(self.tool_schemas)}\n"
                         f"🎯 Goal ID: {self.state.current_leaf_goal_id or 'None'}\n"
                         f"📋 Phase: {self.state.phase.value}", style="cyan"),
                    title="🔧 Tool Schema Query",
                    border_style="cyan"
                ))
            self.logger.info(f"Tool schema status:\n{capture.get()}")
            
            if not self.state.current_leaf_goal_id:
                # No goal yet, return basic tools
                basic_tools = self.tool_schemas[:15]
                with console.capture() as capture:
                    console.print(Panel(
                        Text(f"🎯 No goal set - returning {len(basic_tools)} basic tools", style="yellow"),
                        title="🔧 Tool Selection Strategy",
                        border_style="yellow"
                    ))
                self.logger.info(f"Tool selection:\n{capture.get()}")
                return basic_tools
            
            # Get current goal description
            goal_desc = await self._get_current_goal_description()
            
            # Get ranked tools
            ranked_tools = await self._rank_tools_for_goal(goal_desc, self.state.phase, limit=15)
            
            # CRITICAL FIX: Never return empty - always fall back to basic tools
            if not ranked_tools:
                with console.capture() as capture:
                    console.print(Panel(
                        Text(f"⚠️ Tool ranking returned EMPTY for goal: {goal_desc[:60]}...\n"
                             f"🔄 Falling back to first 15 tools", style="bold yellow"),
                        title="🔧 Tool Ranking Fallback",
                        border_style="yellow"
                    ))
                self.logger.warning(f"Tool ranking fallback:\n{capture.get()}")
                return self.tool_schemas[:15]
            
            # Log successful ranking
            with console.capture() as capture:
                tool_names = []
                for schema in ranked_tools[:5]:  # Show first 5
                    if "function" in schema:
                        name = schema["function"].get("name", "unknown")
                    else:
                        name = schema.get("name", "unknown")
                    tool_names.append(name)
                
                console.print(Panel(
                    Text(f"✅ Ranked {len(ranked_tools)} tools for goal\n"
                         f"🏆 Top 5: {', '.join(tool_names)}\n"
                         f"📖 Goal: {goal_desc[:60]}...", style="green"),
                    title="🔧 Tool Ranking Success",
                    border_style="green"
                ))
            self.logger.info(f"Tool ranking success:\n{capture.get()}")
            
            # CRITICAL: Log all tool names that will be used in structured output constraints
            all_tool_names = []
            for schema in ranked_tools:
                if "function" in schema:
                    name = schema["function"].get("name", "unknown")
                else:
                    name = schema.get("name", "unknown")
                if name != "unknown":
                    all_tool_names.append(name)
            
            self.logger.info(f"[SCHEMA_CONSTRAINT] Returning {len(all_tool_names)} tools for LLM constraints: {all_tool_names}")
            
            return ranked_tools
            
        except Exception as e:
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.text import Text
                
                console = Console()
                with console.capture() as capture:
                    console.print(Panel(
                        Text(f"💥 ERROR in _get_tool_schemas: {str(e)}", style="bold red"),
                        title="🔧 Tool Schema Error",
                        border_style="red"
                    ))
                self.logger.error(f"Tool schema error:\n{capture.get()}")
            except Exception:
                self.logger.error(f"Error in _get_tool_schemas: {e}")
            
            # Safety fallback
            return self.tool_schemas[:15] if self.tool_schemas else []

    # -------------------------------- helper: deliverable creation

    async def _handle_deliverable_creation(self, tool_name: str, tool_args: Dict[str, Any], tool_result: Dict[str, Any]) -> Optional[str]:
        """
        Helper to identify if a tool call created a deliverable and record it as an artifact.
        Returns the artifact_id if created, None otherwise.
        """
        if not tool_result.get("success"):
            return None

        artifact_id = None
        artifact_name = None
        artifact_type = None
        artifact_path = None
        artifact_content = None # For inline content

        # Heuristic for common deliverable creation tools (expand this list as agent gains more output capabilities)
        if "write_file" in tool_name.lower():
            artifact_name = tool_args.get("path") or tool_args.get("filename")
            artifact_path = tool_args.get("path")
            artifact_type = "file"
            # If the tool also returned the content, use that as content for artifact
            if tool_result.get("data") and isinstance(tool_result["data"], str):
                artifact_content = tool_result["data"] # Store content for small files
                self.logger.debug(f"[AML] Captured content from write_file for artifact: {len(artifact_content)} chars")

        elif "generate_html" in tool_name.lower() or "generate" in tool_name.lower() and "html" in tool_name.lower():
            artifact_name = tool_args.get("output_path", f"generated_html_{uuid.uuid4().hex[:8]}.html")
            artifact_type = "html"
            artifact_path = tool_args.get("output_path")
            if tool_result.get("data") and isinstance(tool_result["data"], str):
                artifact_content = tool_result["data"] # HTML content
                self.logger.debug(f"[AML] Captured HTML content for artifact: {len(artifact_content)} chars")

        elif "generate_report" in tool_name.lower() or "write_report" in tool_name.lower():
            artifact_name = tool_args.get("report_name", f"generated_report_{uuid.uuid4().hex[:8]}.md")
            artifact_type = "text" # Or 'markdown' if UMS supports it as a type
            artifact_path = tool_args.get("output_path")
            if tool_result.get("data") and isinstance(tool_result["data"], str):
                artifact_content = tool_result["data"] # Report content
                self.logger.debug(f"[AML] Captured report content for artifact: {len(artifact_content)} chars")
                
        # IMPORTANT: Truncate content before sending to record_artifact if it's too large,
        # UMS record_artifact handles this internally (MAX_TEXT_LENGTH), but good to be aware.

        if artifact_name and artifact_type: # Ensure we have minimum info to record
            self.logger.info(f"[AML] Detected potential deliverable: {artifact_name} ({artifact_type}), attempting to record as artifact.")
            try:
                artifact_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("record_artifact"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "name": artifact_name,
                        "artifact_type": artifact_type,
                        "action_id": None, # Link to the action that produced it (if AML tracks it)
                        "description": f"Output generated by tool {tool_name} for goal {self.state.current_leaf_goal_id[:8] if self.state.current_leaf_goal_id else 'unknown'}",
                        "path": artifact_path,
                        "content": artifact_content, # Pass content directly, UMS will truncate/store
                        "is_output": True, # Mark as a key output artifact
                        "tags": ["deliverable", "agent_output", artifact_type],
                    }
                )
                try:
                    artifact_data = await self._get_valid_ums_data_payload(artifact_res, "record_artifact")
                    artifact_id = artifact_data.get("artifact_id")
                    if artifact_id:
                        self.logger.info(f"[AML] Successfully recorded artifact: {artifact_id} for {artifact_name}")
                        return artifact_id
                except RuntimeError as e:
                    self.logger.warning(f"[AML] Failed to record artifact: {e}")
            except Exception as e:
                self.logger.error(f"[AML] Error calling record_artifact for {artifact_name}: {e}", exc_info=True)
        return None

    # -------------------------------- helper: file access error handling

    async def _handle_file_access_error(self, tool_name: str, tool_args: Dict[str, Any], error_message: str) -> bool:
        """
        Helper to diagnose file access issues and store the diagnosis.
        Returns True if diagnosis was attempted and stored, signaling a need to replan.
        """
        # Heuristic to detect if it's a file-related tool (can be expanded)
        is_file_tool = any(kw in tool_name.lower() for kw in ["file", "path", "read", "write", "save", "load", "artifact", "report"])
        is_file_error = any(kw in error_message.lower() for kw in ["permission denied", "not found", "access denied", "no such file or directory", "os error", "disk", "ioerror"])

        if not is_file_tool or not is_file_error:
            self.logger.debug(f"[AML] Not a file access error or not a file tool: {tool_name}, {error_message}")
            return False # Not a file access error we want to diagnose automatically

        self.logger.warning(f"[AML] Detected file access error for {tool_name}: {error_message}. Attempting diagnosis.")
        
        path_to_check = None
        # Try to extract the path from tool_args if available
        if "path" in tool_args: 
            path_to_check = tool_args["path"]
        elif "filename" in tool_args: 
            path_to_check = tool_args["filename"]
        elif "output_path" in tool_args: 
            path_to_check = tool_args["output_path"] # For generate_report etc.
        elif "report_name" in tool_args: 
            path_to_check = tool_args["report_name"]

        try:
            diagnosis_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("diagnose_file_access_issues"),
                {
                    "path_to_check": path_to_check,
                    "operation_type": "artifacts" if "artifact" in tool_name.lower() or "report" in tool_name.lower() else "filesystem",
                    "db_path": None # Let the UMS tool use its default db_path
                }
            )

            try:
                ums_data = await self._get_valid_ums_data_payload(diagnosis_res, "diagnose_file_access_issues")
                diag_summary = ums_data.get("summary", "File access diagnosis performed.")
                diag_status = ums_data.get("status", "UNKNOWN")
                
                content_to_store = f"File access diagnosis (Status: {diag_status}): {diag_summary}\n"
                content_to_store += "\nIssues found:\n" + "\n".join(ums_data.get("issues_found", ["None specified"]))
                content_to_store += "\nRecommendations:\n" + "\n".join(ums_data.get("recommendations", ["None specified"]))
                
                mem_id = await self._store_memory_with_auto_linking(
                    content=content_to_store,
                    memory_type="file_access_diagnosis",
                    memory_level="working",
                    importance=9.0, # High importance for recovery
                    description=f"Diagnosis for {tool_name} failure: {diag_status}",
                    link_to_goal=True
                )
                if mem_id:
                    self.state.last_action_summary = f"File access diagnosis performed (status: {diag_status}). Requires replan."
                    self.state.consecutive_error_count = 0 # Reset error count as a recovery action was taken
                    self.state.needs_replan = True # Agent needs to replan to act on diagnosis
                    self.logger.info(f"[AML] Stored file access diagnosis in memory: {mem_id}")
                    return True # Indicate that a diagnosis was handled and replan is needed
            except RuntimeError as e:
                self.logger.warning(f"[AML] diagnose_file_access_issues tool failed: {e}")
                return False # Diagnosis tool itself failed, or no data. Not handled.
        except Exception as e:
            self.logger.error(f"[AML] Unexpected error calling diagnose_file_access_issues: {e}", exc_info=True)
            return False # Unexpected error, not handled.

    # -------------------------------- helper: enact decision from model

    async def _enact(self, decision: Any) -> bool:
        """
        Execute the SMART-model output with guaranteed structure.

        Returns
        -------
        progress_made : bool
            Heuristic flag used by metacognition.
        """
        self.logger.debug("[AML] decision raw: %s", decision)

        # Handle both dict and non-dict formats
        if not isinstance(decision, dict):
            self.logger.error(f"[AML] Decision is not a dict: {type(decision)}")
            return False

        # Handle both "decision_type" and "decision" keys for compatibility
        decision_type = decision.get("decision_type") or decision.get("decision")
        
        if not decision_type:
            self.logger.error(f"[AML] No decision type found in: {decision}")
            return False
            
        # Normalize the decision type to match expected values
        decision_type = str(decision_type).upper()
        if decision_type == "THOUGHT_PROCESS":
            decision_type = "THOUGHT_PROCESS"
        elif decision_type == "TOOL_SINGLE":
            decision_type = "TOOL_SINGLE"
        elif decision_type == "TOOL_MULTIPLE":
            decision_type = "TOOL_MULTIPLE"
        elif decision_type == "DONE":
            decision_type = "DONE"
        
        self.logger.info(f"[AML] Processing decision_type: {decision_type}")
        
        if decision_type == "TOOL_SINGLE":
            # Execute the specified tool
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args", {})
            
            if not tool_name:
                self.logger.error("[AML] TOOL_SINGLE decision missing tool_name")
                return False
            
            self.logger.info("[AML] → executing tool %s with args %s", tool_name, tool_args)
            result = await self.tool_exec.run(tool_name, tool_args)
            success = result.get("success", False)
            
            # Record tool effectiveness for learning
            if self.state.current_leaf_goal_id:
                goal_desc = await self._get_current_goal_description()
                await self.record_tool_effectiveness(goal_desc, tool_name, success)
            
            if success:
                # Handle specific UMS tool results
                if "get_artifact_by_id" in tool_name.lower() and result.get("data"):
                    artifact_content = result["data"].get("content")
                    artifact_id = result["data"].get("artifact_id")
                    artifact_name = result["data"].get("name", "Unnamed Artifact")
                    
                    if artifact_content:
                        mem_id = await self._store_memory_with_auto_linking(
                            content=artifact_content,
                            memory_type="artifact_content_read",
                            memory_level="working",
                            importance=7.0, # High importance for active processing
                            description=f"Read content of artifact {artifact_id[:8]}: {artifact_name}",
                            link_to_goal=True # Link this new memory to the current goal
                        )
                        if mem_id:
                            self.state.last_action_summary = f"Successfully read artifact {artifact_id[:8]} (content stored in memory {mem_id[:8]})"
                            self.logger.info(f"[AML] Stored artifact content in memory: {mem_id}")
                            return True 
                        else:
                            self.logger.warning(f"[AML] Failed to store artifact content in memory for {artifact_id}")
                    else:
                        self.logger.warning(f"[AML] get_artifact_by_id succeeded but no content returned for {artifact_id}")

                elif "get_similar_memories" in tool_name.lower() and result.get("data"):
                    similar_mems = result["data"].get("similar_memories", [])
                    source_mem_id = tool_args.get("memory_id", "Unknown")
                    if similar_mems:
                        content_to_store = f"Found {len(similar_mems)} similar memories to memory {source_mem_id[:8]}:\n"
                        for i, sm in enumerate(similar_mems[:5]): # Limit displayed count in memory
                            content_to_store += f"- ID: {sm.get('memory_id', '')[:8]}, Similarity: {sm.get('similarity', 0.0):.2f}, Desc: {sm.get('description', '')[:50]}...\n"
                        
                        mem_id = await self._store_memory_with_auto_linking(
                            content=content_to_store,
                            memory_type="similar_memories_found",
                            memory_level="working",
                            importance=7.5, # High importance for relevant new context
                            description=f"Results of similarity search for {source_mem_id[:8]}",
                            link_to_goal=True
                        )
                        if mem_id:
                            self.state.last_action_summary = f"Successfully searched for similar memories (results stored in memory {mem_id[:8]})"
                            self.logger.info(f"[AML] Stored get_similar_memories results in memory: {mem_id}")
                            return True # Counts as progress for context gathering
                    else:
                        self.logger.info(f"[AML] No similar memories found for {source_mem_id[:8]}")
                        self.state.last_action_summary = f"No similar memories found for {source_mem_id[:8]}"
                        return True # Still progress as a finding.

                elif "query_goals" in tool_name.lower() and result.get("data"):
                    goals_found = result["data"].get("goals", [])
                    if goals_found:
                        content_to_store = f"Found {len(goals_found)} goals matching query criteria (status: {tool_args.get('status', 'any')}, priority: {tool_args.get('priority', 'any')}):\n"
                        for i, goal in enumerate(goals_found[:5]): # Limit for memory content to keep it concise
                            content_to_store += f"- ID: {goal.get('goal_id', '')[:8]}, Status: {goal.get('status', 'N/A')}, Title: {goal.get('title', '')[:50]}...\n"
                        
                        mem_id = await self._store_memory_with_auto_linking(
                            content=content_to_store,
                            memory_type="goals_query_results",
                            memory_level="working",
                            importance=8.0, # High importance as it's about task structure
                            description="Results from querying goals",
                            link_to_goal=True
                        )
                        if mem_id:
                            self.state.last_action_summary = f"Successfully queried goals (results in memory {mem_id[:8]})"
                            self.logger.info(f"[AML] Stored query_goals results in memory: {mem_id}")
                            self.state.needs_replan = True # Querying goals often implies the agent needs to re-evaluate its plan
                            return True # Counts as progress
                    else:
                        self.logger.info(f"[AML] No goals found matching query: {tool_args}")
                        self.state.last_action_summary = f"No goals found matching query: {tool_args}"
                        self.state.needs_replan = True # Agent might need to replan if expected goals aren't found
                        return True # Still progress as a finding.

                elif "consolidate_memories" in tool_name.lower() and result.get("data"):
                    consolidated_content = result["data"].get("consolidated_content")
                    stored_mem_id = result["data"].get("stored_memory_id")
                    consolidation_type = result["data"].get("consolidation_type")
                    if consolidated_content and stored_mem_id:
                        self.state.last_action_summary = f"Successfully consolidated memories ({consolidation_type}) into new memory {stored_mem_id[:8]}"
                        self.logger.info(f"[AML] Consolidated memories. New memory ID: {stored_mem_id}")
                        return True # This is clear progress in knowledge synthesis
                    else:
                        self.logger.warning(f"[AML] Consolidate_memories succeeded but no content or ID returned: {result}")
                        # If no content/ID, might not be real progress for the agent, but tool still "succeeded"
                        return True # Still counted as progress for tool execution

                elif "hybrid_search_memories" in tool_name.lower() and success and result.get("data"):
                    found_memories = result["data"].get("memories", [])
                    search_query = tool_args.get("query", "Unknown query")
                    if found_memories:
                        content_to_store = f"Hybrid search for '{search_query}' found {len(found_memories)} memories:\n"
                        for i, mem_data in enumerate(found_memories[:5]): # Store top 5 in memory content
                            mem_desc = mem_data.get('description', 'No description')[:70]
                            mem_type = mem_data.get('memory_type', 'unknown')
                            hybrid_score = mem_data.get('hybrid_score', 0.0)
                            content_to_store += f"- {mem_desc}... (Type: {mem_type}, Score: {hybrid_score:.2f})\n"
                        
                        mem_id = await self._store_memory_with_auto_linking(
                            content=content_to_store,
                            memory_type="search_results",
                            memory_level="working",
                            importance=7.0, # High importance for new search results
                            description=f"Results of hybrid search for '{search_query[:50]}'",
                            link_to_goal=True
                        )
                        if mem_id:
                            # Optionally, add the most relevant individual memories to working memory too
                            for mem_data in found_memories[:min(len(found_memories), 3)]: # Add top 3 to working memory
                                if mem_data.get('memory_id'):
                                    await self._add_to_working_memory(mem_data['memory_id'])
                            self.state.last_action_summary = f"Successfully performed hybrid search (results in memory {mem_id[:8]})"
                            self.logger.info(f"[AML] Stored hybrid_search_memories results in memory: {mem_id}")
                            return True # Counts as progress for context gathering
                        else:
                            self.logger.warning(f"[AML] Failed to store hybrid search results in memory")
                    else:
                        self.logger.info(f"[AML] Hybrid search for '{search_query}' found no results")
                        self.state.last_action_summary = f"Hybrid search for '{search_query}' found no results"
                        return True # Still progress as a finding

                # Check for deliverable creation after all specific UMS tool handling
                artifact_id = await self._handle_deliverable_creation(tool_name, tool_args, result)
                if artifact_id:
                    self.state.last_action_summary = f"Successfully executed {tool_name} (Recorded artifact: {artifact_id[:8]})"
                else:
                    # Only update summary if no specific UMS tool handling occurred and no artifact was recorded
                    if not any(ums_tool in tool_name.lower() for ums_tool in ["get_artifact_by_id", "get_similar_memories", "query_goals", "consolidate_memories", "hybrid_search_memories"]):
                        self.state.last_action_summary = f"Successfully executed {tool_name}"
                return True
            else:
                error_msg = result.get("error_message", "Unknown error")
                diagnosed = await self._handle_file_access_error(tool_name, tool_args, error_msg)
                if diagnosed:
                    return False # Force a stop for replanning after diagnosis

                self.state.last_action_summary = f"Failed to execute {tool_name}: {error_msg}"
                return False
                
        elif decision_type == "TOOL_MULTIPLE":
            # Execute multiple tools in parallel - much more efficient!
            tool_calls = decision.get("tool_calls", [])
            
            if not tool_calls:
                self.logger.error("[AML] TOOL_MULTIPLE decision missing tool_calls")
                return False
            
            self.logger.info("[AML] → executing %d tools in parallel", len(tool_calls))
            
            # Use the existing run_parallel method from ToolExecutor
            parallel_result = await self.tool_exec.run_parallel(tool_calls)
            
            success = parallel_result.get("success", False)
            results = parallel_result.get("results", [])
            timing_info = parallel_result.get("timing", {})
            
            # Count successes and failures
            successful_count = sum(1 for r in results if r.get("success", False))
            total_count = len(results)
            
            # Check for deliverable creation and handle individual tool failures in parallel mode
            recorded_artifacts_count = 0
            ums_tool_results_count = 0
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("tool_name")
                tool_args = tool_call.get("tool_args", {})
                tool_single_result = results[i] if i < len(results) else {"success": False}
                
                if tool_single_result.get("success", False):
                    # Handle specific UMS tool results (simplified for parallel mode)
                    if any(ums_tool in tool_name.lower() for ums_tool in ["get_artifact_by_id", "get_similar_memories", "query_goals", "consolidate_memories", "hybrid_search_memories"]):
                        ums_tool_results_count += 1
                        # Note: Individual UMS tool result processing is simplified in parallel mode
                        # The LLM can call these tools individually if detailed processing is needed
                    
                    # Check for deliverable creation
                    artifact_id = await self._handle_deliverable_creation(tool_name, tool_args, tool_single_result)
                    if artifact_id:
                        recorded_artifacts_count += 1
                        self.logger.info(f"[AML] Recorded artifact from parallel tool: {artifact_id} for {tool_name}")
                else:
                    # Handle individual tool failures in parallel mode
                    error_msg = tool_single_result.get("error_message", "Unknown error")
                    diagnosed = await self._handle_file_access_error(tool_name, tool_args, error_msg)
                    if diagnosed:
                        return False # Force a stop for replanning after diagnosis (first one is enough)
            
            # Record effectiveness for each tool
            if self.state.current_leaf_goal_id:
                goal_desc = await self._get_current_goal_description()
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("tool_name")
                    if tool_name and i < len(results):
                        tool_success = results[i].get("success", False)
                        await self.record_tool_effectiveness(goal_desc, tool_name, tool_success)
            
            # Log timing information
            if timing_info:
                timing_summary = ", ".join([f"{tid}: {time:.2f}s" for tid, time in timing_info.items()])
                self.logger.debug(f"[AML] Parallel execution timing: {timing_summary}")
            
            if successful_count > 0:
                self.state.last_action_summary = f"Parallel execution: {successful_count}/{total_count} tools succeeded"
                if recorded_artifacts_count > 0:
                    self.state.last_action_summary += f" ({recorded_artifacts_count} artifact(s) recorded)"
                if ums_tool_results_count > 0:
                    self.state.last_action_summary += f" ({ums_tool_results_count} UMS tool(s) processed)"
                return True
            else:
                self.state.last_action_summary = f"Parallel execution failed: 0/{total_count} tools succeeded"
                return False
                
        elif decision_type == "THOUGHT_PROCESS":
            # Store reasoning as memory
            thought = decision.get("content", "")
            
            if not thought.strip():
                self.logger.warning("[AML] Empty thought content")
                return False
                
            mem_id = await self._store_memory_with_auto_linking(
                content=thought,
                memory_type="reasoning_step",
                memory_level="working",
                importance=6.0,
                description="Reasoning from SMART model"
            )
            
            if mem_id:
                # Link any referenced memories as supporting evidence
                evid_ids = re.findall(r"mem_[0-9a-f]{8}", thought)
                if evid_ids:
                    await self.mem_graph.register_reasoning_trace(
                        thought_mem_id=mem_id,
                        evidence_ids=evid_ids,
                    )
            
            self.state.last_action_summary = "Generated reasoning thought"
            return bool(thought.strip())
            
        elif decision_type == "DONE":
            # Validate completion before accepting
            is_valid = await self.metacog._goal_completed()
            
            if is_valid:
                self.state.phase = Phase.COMPLETE
                self.state.goal_achieved_flag = True
                self.state.last_action_summary = "Task completed and validated"

                # Generate and record the final workflow report
                self.logger.info("[AML] Goal completed. Generating final workflow report.")
                try:
                    report_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("generate_workflow_report"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "report_format": "markdown", # Or "html" if the agent's final deliverable is HTML
                            "include_details": True,
                            "include_thoughts": True,
                            "include_artifacts": True,
                            "style": "professional" # Choose a style
                        }
                    )
                    try:
                        report_data = await self._get_valid_ums_data_payload(report_res, "generate_workflow_report")
                        report_content = report_data.get("report")
                        report_format = report_data.get("format", "markdown")
                        report_title = report_data.get("title", f"Workflow Report {self.state.workflow_id[:8]}")

                        if report_content:
                            # Record the generated report as an artifact
                            report_artifact_id = await self._handle_deliverable_creation(
                                "generate_workflow_report", # Tool name for artifact heuristic
                                {"output_path": f"{report_title}.{report_format}", "report_format": report_format}, # Mimic args
                                {"success": True, "data": report_content} # Mimic tool result with content
                            )
                            if report_artifact_id:
                                self.state.last_action_summary += f" (Final report recorded as artifact: {report_artifact_id[:8]})"
                                self.logger.info(f"[AML] Final workflow report recorded as artifact: {report_artifact_id}")
                        else:
                            self.logger.warning("[AML] Generated workflow report but no content returned")
                    except RuntimeError as e:
                        self.logger.warning(f"[AML] Failed to generate workflow report: {e}")
                except Exception as e:
                    self.logger.error(f"[AML] Error generating final workflow report: {e}", exc_info=True)

                return True
            else:
                # Not actually done
                self.logger.warning("Agent claimed completion but validation failed")
                
                # Create a corrective memory
                await self._store_memory_with_auto_linking(
                    content="Premature completion attempt - validation failed. Need to create expected outputs.",
                    memory_type="correction",
                    memory_level="working",
                    importance=8.0,
                    description="Completion validation failed"
                )
                
                self.state.last_action_summary = "Completion validation failed - continuing work"
                self.state.stuck_counter += 1
                return False
        else:
            # Log the unexpected decision type for debugging
            self.logger.error(f"[AML] Unexpected decision type: {decision_type}, full decision: {decision}")
            # Convert unknown decision types to thought process to avoid complete failure
            if decision.get("content") or decision.get("reasoning"):
                content = decision.get("content") or decision.get("reasoning")
                mem_id = await self._store_memory_with_auto_linking(
                    content=f"Unexpected decision type '{decision_type}': {content}",
                    memory_type="reasoning_step",
                    memory_level="working",
                    importance=5.0,
                    description="Recovered from unexpected decision format"
                )
                self.state.last_action_summary = "Processed unexpected decision format"
                return bool(content.strip()) if content else False
            return False

    # ----------------------------------------------------- after-turn misc



    # -------------------------------------------------------- failure handling



    def _record_failure(self, reason: str) -> bool:
        """
        Record a failure state, save it, return False to signal stop.
        """
        self.state.last_error_details = {"reason": reason, "loop_count": self.state.loop_count}
        self.state.consecutive_error_count += 1
        
        # Try to save state with error details
        try:
            import asyncio
            asyncio.create_task(self._save_cognitive_state())
        except Exception:
            pass  # Don't let state saving errors mask the original failure
            
        self.logger.error(f"[AML] Task failed: {reason}")
        return False

    async def execute_llm_decision(self, llm_decision: Dict[str, Any]) -> bool:
        """
        Execute a decision from the LLM and return whether the agent should continue.
        
        This method is called by MCPClient after getting a decision from the LLM.
        It should execute the decision and return True to continue or False to stop.
        
        Parameters
        ----------
        llm_decision : Dict[str, Any]
            The decision dictionary from the LLM call
            
        Returns
        -------
        bool
            True if the agent should continue, False if it should stop
        """
        try:
            # Check if we should shutdown
            if self._shutdown_event.is_set():
                self.logger.info("[AML] Shutdown event set, stopping execution")
                return False
                
            # Execute the decision using the existing _enact method
            progress_made = await self._enact(llm_decision)
            
            # Check completion status
            if self.state.goal_achieved_flag or self.state.phase == Phase.COMPLETE:
                self.logger.info("[AML] Goal achieved or phase complete")
                return False
                
            # If no progress was made, increment stuck counter
            if not progress_made:
                self.state.stuck_counter += 1
                if self.state.stuck_counter >= 5:  # Stop if stuck for too long
                    self.logger.warning("[AML] No progress made for 5 turns, stopping")
                    return False
            else:
                # Reset stuck counter on progress
                self.state.stuck_counter = 0
                
            # Continue if we made progress and haven't hit limits
            return True
            
        except Exception as e:
            self.logger.error(f"[AML] Error executing LLM decision: {e}", exc_info=True)
            self.state.last_error_details = {"error": str(e), "context": "execute_llm_decision"}
            self.state.consecutive_error_count += 1
            
            # Stop on too many consecutive errors
            if self.state.consecutive_error_count >= 3:
                self.logger.error("[AML] Too many consecutive errors, stopping")
                return False
                
            return True  # Try to continue despite error

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the agent by setting the shutdown event and cleaning up resources.
        """
        self.logger.info("[AML] Shutting down AgentMasterLoop...")
        self._shutdown_event.set()
        
        # Cancel any running async tasks
        if self.async_queue:
            self.async_queue.cancel_all()
            
        # Save final state
        try:
            await self._save_cognitive_state()
        except Exception as e:
            self.logger.warning(f"[AML] Error saving state during shutdown: {e}")
            
        self.logger.info("[AML] AgentMasterLoop shutdown complete")





            
    async def _store_memory_with_auto_linking(
        self, 
        content: str, 
        memory_type: str = "reasoning_step",
        memory_level: str = "working",
        importance: float = 5.0,
        description: str = "",
        link_to_goal: bool = True
    ) -> Optional[str]:
        """
        Store a memory with automatic linking via UMS store_memory tool.
        REVISED: Now properly adds memory to working memory context.
        """
        try:
            # First, store the memory
            store_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("store_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "content": content,
                    "memory_type": memory_type,
                    "memory_level": memory_level,
                    "importance": importance,
                    "description": description or f"Auto-stored {memory_type}",
                    "suggest_links": True,
                    "max_suggested_links": 5,
                    "link_suggestion_threshold": 0.7,
                    "action_id": None,
                    "generate_embedding": True
                }
            )
            
            try:
                memory_data = await self._get_valid_ums_data_payload(store_envelope, "store_memory_auto_link")
                memory_id = memory_data.get("memory_id")
                if not memory_id:
                    self.logger.warning(f"Memory storage succeeded but no memory_id returned: {memory_data}")
                    return None
            except RuntimeError as e:
                self.logger.warning(f"Memory storage failed: {e}")
                return None
                
            # CRITICAL NEW STEP: Add to working memory if we have a context
            if self.state.context_id and memory_level == "working":
                # Call _add_to_working_memory which encapsulates the logic and its own fallbacks.
                # It returns True if successful.
                added_to_wm_success = await self._add_to_working_memory(memory_id)
                if added_to_wm_success:
                    self.logger.debug(f"Added memory {memory_id} to working memory")
                else:
                    # Log if _add_to_working_memory itself failed to update the state
                    self.logger.warning(f"Failed to add memory {memory_id} to working memory context {self.state.context_id} after storing.")
            
            # If requested, associate memory with current goal
            if link_to_goal and self.state.current_leaf_goal_id and memory_id:
                try:
                    current_meta = await self.mem_graph._get_metadata(memory_id)
                    current_meta["associated_goal_id"] = self.state.current_leaf_goal_id
                    
                    metadata_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("update_memory_metadata"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "memory_id": memory_id,
                            "metadata": current_meta
                        }
                    )
                    # Use helper to validate the metadata update
                    await self._get_valid_ums_data_payload(metadata_envelope, "update_memory_metadata")
                except Exception as e:
                    self.logger.debug(f"Goal association failed (non-critical): {e}")
                    
            return memory_id
                
        except Exception as e:
            self.logger.warning(f"Auto-linking memory storage failed: {e}")
            return None
    # -------------------------------- helper: spawn background fast tasks

    async def record_tool_effectiveness(self, goal_desc: str, tool_name: str, success: bool) -> None:
        """Record tool effectiveness for learning purposes."""
        try:
            # Simple effectiveness tracking - could be enhanced later
            if goal_desc not in self._tool_effectiveness_cache:
                self._tool_effectiveness_cache[goal_desc] = {}
            
            if tool_name not in self._tool_effectiveness_cache[goal_desc]:
                self._tool_effectiveness_cache[goal_desc][tool_name] = {"success": 0, "total": 0}
            
            self._tool_effectiveness_cache[goal_desc][tool_name]["total"] += 1
            if success:
                self._tool_effectiveness_cache[goal_desc][tool_name]["success"] += 1
                
        except Exception as e:
            self.logger.debug(f"Failed to record tool effectiveness: {e}")

