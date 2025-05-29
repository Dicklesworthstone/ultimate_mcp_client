from __future__ import annotations

"""Robust Agent Loop implementation using MCPClient infrastructure."""

###############################################################################
# SECTION 0. Imports & typing helpers
###############################################################################

import asyncio
import dataclasses
import datetime as _dt
import enum
import heapq
import json
import math
import os
import re
import sqlite3
import statistics
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from pathlib import Path

import logging

import networkx as nx

###############################################################################
# SECTION 1. Agent configuration constants
###############################################################################

# These constants provide default values for agent behavior when MCPClient config is unavailable.

# Metacognition & reflection timing
REFLECTION_INTERVAL = 15  # generate reflection memories every N turns
GRAPH_MAINT_EVERY = 10  # turns between graph‑maintenance phases  
STALL_THRESHOLD = 3  # consecutive non‑progress turns → forced reflection

# Fast LLM call budget (per micro-task)
FAST_CALL_MAX_USD = 0.02  # per micro‑call budget ceiling

# Default model names (fallback only - use mcp_client.config.default_model instead)
DEFAULT_SMART_MODEL = "gpt-4.1"
DEFAULT_FAST_MODEL = "gemini-2.5-flash-preview-05-20"

###############################################################################
# SECTION 2. Enumerations & simple data classes
###############################################################################


class GoalStatus(str, enum.Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ABANDONED = "abandoned"

class LinkKind(str, enum.Enum):
    """Canonical, machine-friendly edge types the UMS already recognises."""
    RELATED = "related"
    CAUSAL = "causal"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    TASK = "task"
    REFERENCES = "references"
    ELABORATES = "elaborates"    
    QUESTION_OF = "question_of"   
    CONSEQUENCE_OF = "consequence_of"

class Phase(str, enum.Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    GRAPH_MAINT = "graph_maintenance"
    REVIEW = "review"
    COMPLETE = "complete"


class DecisionType(str, enum.Enum):
    THOUGHT = "thought_process"
    TOOL_SINGLE = "call_tool"
    TOOL_BATCH = "multiple_tools_executed_by_mcp"
    DONE = "idle"
    ERROR = "error"

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
    last_graph_maint_turn: int = 0
    graph_health: float = 0.9
    pending_attachments: List[str] = dataclasses.field(default_factory=list)
    created_at: _dt.datetime = dataclasses.field(default_factory=_dt.datetime.utcnow)
    
    # New fields for enhanced agent state management
    current_plan: Optional[List[Dict[str, Any]]] = dataclasses.field(default_factory=list)
    goal_stack: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    current_thought_chain_id: Optional[str] = None
    last_action_summary: Optional[str] = None
    last_error_details: Optional[Dict[str, Any]] = None
    consecutive_error_count: int = 0
    needs_replan: bool = True
    goal_achieved_flag: bool = False
    recent_action_signatures: List[str] = dataclasses.field(default_factory=list)
    max_action_history: int = 10

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

    async def fetch_contradicting_evidence(self, mem_id: str, limit: int = 5) -> list[str]:
        """
        Return memory_ids of facts or evidence that *contradict* the given memory.
        """
        try:
            # Use UMS tool to get memory links
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_linked_memories"), 
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_id": mem_id,
                    "link_type": "CONTRADICTS",
                    "bidirectional": True,
                    "limit": limit
                }
            )
            if res.get("success") and res.get("data"):
                links = res["data"].get("links", [])
                contradicting_ids = []
                for link in links:
                    # Collect both source and target IDs, excluding the original memory
                    if link.get("source_memory_id") != mem_id:
                        contradicting_ids.append(link["source_memory_id"])
                    if link.get("target_memory_id") != mem_id:
                        contradicting_ids.append(link["target_memory_id"])
                return list(set(contradicting_ids))[:limit]  # Remove duplicates and limit
            return []
        except Exception:
            return []
        
    async def detect_inconsistencies(self) -> List[Tuple[str, str]]:
        """
        Use the server-side `get_contradictions` tool.
        Fallback to client-side analysis when the tool is unavailable.
        """
        try:
            # Try using the server-side contradictions tool first
            contradictions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_contradictions"),
                {"workflow_id": self.state.workflow_id, "limit": 50}
            )
            if contradictions_res.get("success") and contradictions_res.get("data"):
                pairs_data = contradictions_res["data"].get("pairs", [])
                pairs = [(p["a"], p["b"]) for p in pairs_data]
                
                # Filter out resolved contradictions
                filtered_pairs = []
                for src, tgt in pairs:
                    try:
                        link_meta_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                            self.ums_server_name,
                            self._get_ums_tool_name("get_memory_link_metadata"), 
                            {
                                "workflow_id": self.state.workflow_id,
                                "source_memory_id": src,
                                "target_memory_id": tgt,
                                "link_type": "CONTRADICTS",
                            }
                        )
                        if link_meta_res.get("success") and link_meta_res.get("data"):
                            metadata = link_meta_res["data"].get("metadata", {})
                            if "resolved_at" not in metadata:
                                filtered_pairs.append((src, tgt))
                        else:
                            # If we can't check metadata, include it (fail-safe)
                            filtered_pairs.append((src, tgt))
                    except Exception:
                        # If we can't check metadata, include it (fail-safe)
                        filtered_pairs.append((src, tgt))
                return filtered_pairs
        except Exception:
            pass  # Continue to fallback logic
            
        # Fallback to client-side contradiction detection
        return await self._detect_inconsistencies_fallback()

    async def _detect_inconsistencies_fallback(self) -> List[Tuple[str, str]]:
        """
        Client-side fallback for contradiction detection when server-side tools are unavailable.
        
        This replaces the original SQL-based detection with UMS tool calls.
        """
        result: Set[Tuple[str, str]] = set()
        
        # Rule 1: explicit contradicts links ---------------------------------
        try:
            # Get all CONTRADICTS links for this workflow
            contradicts_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("query_graph_by_link_type"),
                {
                    "workflow_id": self.state.workflow_id,
                    "link_type": "CONTRADICTS",
                    "limit": 100
                }
            )
            if contradicts_res.get("success") and contradicts_res.get("data"):
                pairs = contradicts_res["data"].get("pairs", [])
                
                # Filter out resolved contradictions
                for pair in pairs:
                    src, tgt = pair["a"], pair["b"]
                    # Check if this contradiction has been marked as resolved
                    try:
                        link_meta_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                            self.ums_server_name,
                            self._get_ums_tool_name("get_memory_link_metadata"), 
                            {
                                "workflow_id": self.state.workflow_id,
                                "source_memory_id": src,
                                "target_memory_id": tgt,
                                "link_type": "CONTRADICTS",
                            }
                        )
                        if link_meta_res.get("success") and link_meta_res.get("data"):
                            metadata = link_meta_res["data"].get("metadata", {})
                            if "resolved_at" not in metadata:
                                result.add((src, tgt))
                        else:
                            # If we can't check metadata, include it (fail-safe)
                            result.add((src, tgt))
                    except Exception:
                        # If we can't check metadata, include it (fail-safe)
                        result.add((src, tgt))
        except Exception:
            pass  # Continue to other rules if this fails
        
        # Rule 2: naive negation check on recent memories ---------------------
        try:
            recent_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("query_memories"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": "working",
                    "limit": 40,
                    "sort_by": "created_at",
                    "sort_order": "desc"
                }
            )
            if recent_res.get("success") and recent_res.get("data"):
                recent_memories = recent_res["data"].get("memories", [])
                for i, mem_i in enumerate(recent_memories):
                    text_i = mem_i.get("content", "").lower()
                    for j in range(i + 1, len(recent_memories)):
                        mem_j = recent_memories[j]
                        text_j = mem_j.get("content", "").lower()
                        # simple negation pattern: one text contains "not" + other statement substring
                        if (" not " in text_i or " no " in text_i) and mem_j.get("content", "")[:40].lower() in text_i:
                            result.add((mem_i["memory_id"], mem_j["memory_id"]))
                        if (" not " in text_j or " no " in text_j) and mem_i.get("content", "")[:40].lower() in text_j:
                            result.add((mem_j["memory_id"], mem_i["memory_id"]))
        except Exception:
            pass  # Continue if this rule fails
         
        # Rule 3: Soft negative feedback loops (A→B→…→A via CAUSAL edges) ----
        try:
            causal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("query_graph_by_link_type"),
                {
                    "workflow_id": self.state.workflow_id,
                    "link_type": "CAUSAL",
                    "limit": 200
                }
            )
            if causal_res.get("success") and causal_res.get("data"):
                g = nx.DiGraph()
                causal_pairs = causal_res["data"].get("pairs", [])
                for pair in causal_pairs:
                    g.add_edge(pair["a"], pair["b"])
                
                try:
                    for cycle in nx.simple_cycles(g):
                        if len(cycle) > 1:
                            result.add((cycle[0], cycle[-1]))
                except nx.NetworkXError:
                    pass  # ignore if graph is empty or malformed
        except Exception:
            pass  # Continue if this rule fails
        
        return list(result)

    async def consolidate_cluster(self, min_size: int = 6) -> None:
        """
        Detect dense sub‑graphs and create summarising semantic memories.
        
        Replaces direct SQL queries with UMS tool calls.
        """
        try:
            # Get all memories with their link counts via UMS
            memories_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("query_memories"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": "working",
                    "include_link_counts": True,
                    "limit": 100
                }
            )
            
            if not memories_res.get("success") or not memories_res.get("data"):
                return
                
            memories = memories_res["data"].get("memories", [])
            
            # Find hubs (memories with >= min_size links)
            hubs = [mem for mem in memories if mem.get("link_count", 0) >= min_size]
                
            for hub in hubs:
                hub_id = hub["memory_id"]
                
                # Get linked memories for this hub
                links_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    self.ums_server_name,
                    self._get_ums_tool_name("get_linked_memories"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "memory_id": hub_id,
                        "limit": 20
                    }
                )
            
                if not links_res.get("success") or not links_res.get("data"):
                    continue
                    
                links = links_res["data"].get("links", [])
                if not links:
                    continue
                    
                # Get content of linked memories
                linked_memory_ids = [link["target_memory_id"] for link in links if link["target_memory_id"] != hub_id]
                if not linked_memory_ids:
                    continue
                    
                contents = []
                importances = []
                for mem_id in linked_memory_ids:
                    mem_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                        self.ums_server_name,
                        self._get_ums_tool_name("get_memory_by_id"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "memory_id": mem_id
                        }
                    )
                    if mem_res.get("success") and mem_res.get("data"):
                        memory = mem_res["data"]
                        contents.append(memory.get("content", ""))
                        importances.append(memory.get("importance", 5.0))
                
                if not contents:
                    continue
                    
                # Create summary using LLM
                combined = "\n".join(contents)
                summary_prompt = f"Summarize the following related memories into a concise overview (max 500 chars):\n\n{combined[:2000]}"
                
                try:
                    summary_res = await self.mcp_client.query_llm_structured(
                        prompt=summary_prompt,
                        schema={
                            "type": "object",
                            "properties": {"summary": {"type": "string"}},
                            "required": ["summary"]
                        },
                        use_cheap_model=True
                    )
                    summary = summary_res.get("summary", combined[:500])
                except Exception:
                    summary = textwrap.shorten(combined, 500, placeholder=" …")
                
                # Store summary memory
                avg_importance = statistics.mean(importances) if importances else 5.0
                summary_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    self.ums_server_name,
                    self._get_ums_tool_name("store_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": summary,
                        "memory_level": "semantic",
                        "memory_type": "summary",
                        "importance": avg_importance,
                        "description": f"Consolidated summary of {len(linked_memory_ids)} related memories around {hub_id}",
                    }
                )
                
                if summary_res.get("success") and summary_res.get("data"):
                    new_id = summary_res["data"].get("memory_id")
                    if new_id:
                        # Create GENERALIZES link hub -> summary
                        await self.auto_link(
                            src_id=new_id,
                            tgt_id=hub_id,
                            context_snip="consolidated summary",
                            kind_hint=LinkKind.GENERALIZES
                        )
                        
        except Exception as e:
            # Log error but don't fail the entire operation
            self.mcp_client.logger.warning(f"Error during cluster consolidation: {e}")

        # house-keeping: mild link-strength decay (optional, cheap)
        await self.decay_link_strengths()

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
            
            if not candidates_res.get("success") or not candidates_res.get("data"):
                return
                
            candidates = candidates_res["data"].get("memories", [])
            
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
                            "updated_at": int(time.time())
                        }
                    )
                    
        except Exception as e:
            self.mcp_client.logger.warning(f"Error during memory promotion: {e}")

    async def snapshot_context_graph(self) -> Dict[str, Any]:
        """Return small adjacency list + node metadata for reasoning.

        We include:
        * latest 25 working‑level memories
        * their outgoing links (any type)
        * each linked memory's content snippet (<= 200 chars)
        """
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Tuple[str, str, str]] = []

        try:
            # Get recent working memories
            recent_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("query_memories"),
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_level": "working",
                    "limit": 25,
                    "sort_by": "created_at",
                    "sort_order": "desc"
                }
            )
            
            if not recent_res.get("success") or not recent_res.get("data"):
                return {"nodes": nodes, "edges": edges}
                
            recent_memories = recent_res["data"].get("memories", [])
            
            for memory in recent_memories:
                mem_id = memory["memory_id"]
                nodes[mem_id] = {
                    "id": mem_id,  # Make sure this key exists
                    "type": memory.get("memory_type", "UNKNOWN"),
                    "snippet": memory.get("content", "")[:200],
                    "tags": await self._get_tags_for_memory(mem_id) or set()  # Ensure fallback
                }
                # Get outgoing links for this memory
                links_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    self.ums_server_name,
                    self._get_ums_tool_name("get_linked_memories"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "memory_id": mem_id,
                        "direction": "outgoing"
                    }
                )
                
                if links_res.get("success") and links_res.get("data"):
                    links = links_res["data"].get("links", [])
                    for link in links:
                        tgt_id = link["target_memory_id"]
                        link_type = link.get("link_type", "RELATED")
                        edges.append((mem_id, tgt_id, link_type))
                        
                        # Add target node if not already present
                        if tgt_id not in nodes:
                            tgt_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                                self.ums_server_name,
                                self._get_ums_tool_name("get_memory_by_id"),
                                {
                                    "workflow_id": self.state.workflow_id,
                                    "memory_id": tgt_id
                                }
                            )
                            if tgt_res.get("success") and tgt_res.get("data"):
                                tgt_memory = tgt_res["data"]
                                nodes[tgt_id] = {
                                    "id": tgt_id,
                                    "type": tgt_memory.get("memory_type", "UNKNOWN"),
                                    "snippet": tgt_memory.get("content", "")[:200],
                                    "tags": await self._get_tags_for_memory(tgt_id)
                                }
                                
        except Exception as e:
            self.mcp_client.logger.warning(f"Error creating context graph snapshot: {e}")
            
        return {"nodes": nodes, "edges": edges}

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
            if meta_src_res.get("success") and meta_src_res.get("data"):
                meta_src = meta_src_res["data"].get("metadata", {}).get("link_type_cache", {})
            if tgt_id in meta_src:
                return meta_src[tgt_id]
                    
            meta_tgt_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("get_memory_metadata"),
                {"workflow_id": self.state.workflow_id, "memory_id": tgt_id}
            )
            if meta_tgt_res.get("success") and meta_tgt_res.get("data"):
                meta_tgt = meta_tgt_res["data"].get("metadata", {}).get("link_type_cache", {})
            if src_id in meta_tgt:
                return meta_tgt[src_id]
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
            if res.get("success") and res.get("data"):
                content = res["data"].get("content", "")
                # TODO: Update access count via UMS tool if available
                return content
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
            if res.get("success") and res.get("data"):
                tags = res["data"].get("tags", [])
                return set(tags)
            return set()
        except Exception:
            return set()

    def flush(self) -> None:
        """No-op since we no longer use buffered writes."""
        pass

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
            if res.get("success") and res.get("data"):
                return res["data"].get("vector")
            
            # If no embedding exists, try to create one
            create_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("create_embedding"),
                {"workflow_id": self.state.workflow_id, "memory_id": memory_id}
            )
            if create_res.get("success") and create_res.get("data"):
                return create_res["data"].get("vector")
                
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
            if res.get("success") and res.get("data"):
                return res["data"].get("cosine")
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
            if res.get("success") and res.get("data"):
                return res["data"].get("metadata", {}) or {}
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


###############################################################################
# SECTION 4. Async micro-task infra (outline)
###############################################################################
# The remaining sections are unchanged stubs from V4 and will be fleshed out in
# follow‑up iterations. They're left intact so that the file keeps compiling.


class StructuredCall:
    """Utility class for making *cheap/fast* LLM calls that must return a
    **strict JSON object** matching a user‑supplied JSON‑Schema.

    The implementation leverages the existing `MCPClient.query_llm_structured()` 
    method which already handles provider clients, cost tracking, and retry logic.
    This makes the class a thin wrapper that ensures the existing MCP client
    infrastructure is properly utilized.

    Features
    --------
    • Automatic budget enforcement via the global constant ``FAST_CALL_MAX_USD``.
    • Up‑to‑`max_retries` auto‑repair passes when the model returns invalid JSON
      (it appends an *in‑context instruction* asking the model to fix the
      formatting and re‑tries).
    • *Optional* JSON‑Schema validation using the **jsonschema** library if it is
      available in the environment; otherwise falls back to a minimal
      hand‑rolled required‑field check so that runtime dependencies remain light.
    • Thread‑safe async execution by leveraging MCPClient's existing async infrastructure.
    • When the MCP client did **not** define `fast_llm_call`, the constructor
      monkey‑patches ``mcp_client.fast_llm_call = self.query`` so that other
      components (e.g. `MemoryGraphManager`) can transparently invoke the fast
      model regardless of the concrete MCP implementation.
    """

    def __init__(
        self,
        mcp_client,
        model_name: str = None,
        cost_cap_usd: float = None,
        max_retries: int = 2,
    ) -> None:
        self.mcp_client = mcp_client
        
        # Use MCPClient config values with fallbacks to constants
        if model_name is None:
            model_name = getattr(mcp_client.config, 'default_cheap_and_fast_model', DEFAULT_FAST_MODEL)
        if cost_cap_usd is None:
            cost_cap_usd = FAST_CALL_MAX_USD
            
        self.model_name = model_name
        self.cost_cap = cost_cap_usd
        self.max_retries = max_retries

        # If the enclosing MCP client does *not* expose a convenience wrapper
        # for cheap LLM hits, install this StructuredCall.query so that callers
        # like `MemoryGraphManager` can always do `await mcp.fast_llm_call(...)`.
        if not hasattr(self.mcp_client, "fast_llm_call"):
            self.mcp_client.fast_llm_call = self.query

        # Try importing optional deps only once so that ImportError is neat.
        try:
            import jsonschema  # noqa: F401

            self._jsonschema_available = True
        except ImportError:
            self._jsonschema_available = False

    # ---------------------------------------------------------------------
    # Public coroutine
    # ---------------------------------------------------------------------

    async def query(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Send *prompt* to the **fast/cheap** model and enforce *schema* output.

        The call will automatically *retry* up to ``self.max_retries`` times if
        the returned content cannot be parsed as valid JSON **or** fails the
        schema validation.
        """
        # First, if the MCP client already implements its own fast helper we use
        # it directly – that covers the production path and also lets that
        # helper handle cost tracking.
        if hasattr(self.mcp_client, "fast_llm_call") and self.mcp_client.fast_llm_call is not self.query:  # type: ignore[attr-defined]
            return await self.mcp_client.fast_llm_call(prompt, schema)  # type: ignore[misc]

        # Check budget before making any calls
        if hasattr(self.mcp_client, 'cheap_total_cost') and self.mcp_client.cheap_total_cost > self.cost_cap:
            raise RuntimeError("StructuredCall budget exceeded – aborting")

        # Use MCPClient's query_llm_structured method
        response_data = await self._call_mcp_structured(prompt, schema)

        # ------------------------------------------------------------------
        # Parse + validate with retry logic
        # ------------------------------------------------------------------
        for attempt in range(self.max_retries + 1):
            try:
                # MCPClient.query_llm_structured should return a dict already
                if isinstance(response_data, dict):
                    parsed = response_data
                else:
                    # If it returned a string, try to parse it as JSON
                    parsed = json.loads(str(response_data))
            except (json.JSONDecodeError, TypeError, ValueError):
                if attempt == self.max_retries:
                    raise
                # Ask model to correct its output
                response_data = await self._repair(prompt, str(response_data))
                continue

            # JSON Schema validation (optional) ---------------------------
            if self._jsonschema_available:
                try:
                    import jsonschema

                    jsonschema.validate(instance=parsed, schema=schema)  # type: ignore[arg-type]
                except jsonschema.ValidationError:
                    if attempt == self.max_retries:
                        raise
                    response_data = await self._repair(prompt, str(response_data))
                    continue
            else:
                # Minimal required‑field check
                required = schema.get("required", [])
                if not all(k in parsed for k in required):
                    if attempt == self.max_retries:
                        raise ValueError("Missing required JSON fields: " + ", ".join(required))
                    response_data = await self._repair(prompt, str(response_data))
                    continue
            # Success ---------------------------------------------------
            return parsed

        # Should never reach here
        raise RuntimeError("StructuredCall: exhausted retries without valid JSON")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_mcp_structured(self, prompt: str, schema: Dict[str, Any]) -> Any:
        """
        Call MCPClient.query_llm_structured() with the appropriate parameters.
        """
        try:
            return await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": prompt}],
                response_schema=schema,
                model_override=self.model_name,
                use_cheap_model=True
            )
        except Exception:
            # If query_llm_structured signature is different, try the simple version
            try:
                return await self.mcp_client.query_llm_structured(
                    prompt=prompt,
                    schema=schema,
                    use_cheap_model=True
                )
            except Exception as e:
                raise RuntimeError(f"Failed to call MCPClient.query_llm_structured: {e}") from e

    async def _repair(self, original_prompt: str, bad_response: str) -> Any:
        """Ask the model to *just* return valid JSON according to *original* request."""
        repair_prompt = (
            original_prompt + " The previous response was not valid JSON or failed schema validation. "
            "Return *only* a valid JSON object that satisfies the schema – no explanations."  # noqa: E501
        )
        return await self._call_mcp_structured(repair_prompt, {})  # Empty schema for repair attempt


class AsyncTask:
    """Represents a *single* asynchronous micro-call delegated to the fast LLM.

    The task life-cycle is fully tracked so the orchestrator can make informed
    scheduling decisions (e.g. cost accounting, detecting starved tasks).
    """

    __slots__ = ("name", "coro", "callback", "weight", "created_at", "started_at", "done_at", "_task", "_result", "_exception")

    def __init__(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        callback: Optional[Callable[[Any], Awaitable[None] | None]] = None,
        *,
        weight: float = 1.0,
    ) -> None:
        self.name: str = name
        self.coro: Coroutine[Any, Any, Any] = coro
        self.callback = callback
        self.weight: float = weight  # arbitrary unit for prioritisation

        self.created_at: float = asyncio.get_event_loop().time()
        self.started_at: Optional[float] = None
        self.done_at: Optional[float] = None

        self._task: Optional[asyncio.Task[Any]] = None
        self._result: Any = None
        self._exception: Optional[BaseException] = None

    # ------------------------------------------------------------------ api

    def start(self) -> None:
        """Schedule the coroutine on the current loop and attach bookkeeping."""
        if self._task is not None:
            # Idempotent: start() can be called only once.
            return
        self.started_at = asyncio.get_event_loop().time()
        self._task = asyncio.create_task(self.coro, name=self.name)
        self._task.add_done_callback(self._on_done)  # type: ignore[arg-type]

    # Properties -------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()

    @property
    def result(self) -> Any:  # may raise if exception
        if self._task is None or not self._task.done():
            raise RuntimeError("Task not finished yet")
        return self._task.result()

    @property
    def exception(self) -> Optional[BaseException]:
        return self._exception

    # Internal ---------------------------------------------------------------

    def _on_done(self, task: asyncio.Task[Any]) -> None:  # pragma: no cover
        self.done_at = asyncio.get_event_loop().time()
        try:
            self._result = task.result()
        except BaseException as exc:  # noqa: BLE001 – capture any exception
            self._exception = exc
        # Fire user callback *after* capturing result/exception so that user
        # code can inspect them.
        if self.callback is not None:
            # If callback is async we schedule it; otherwise run synchronously.
            if asyncio.iscoroutinefunction(self.callback):  # type: ignore[arg-type]
                asyncio.create_task(self.callback(self._result))
            else:
                try:
                    self.callback(self._result)  # type: ignore[misc]
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

    def inject_flush_cb(self, cb: Callable[[], None]) -> None:
        """Allows outer code to register a flush callback executed after drain."""
        self._flush_cb = cb

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

        # Ensure buffered graph writes hit disk before the big-model turn.
        if hasattr(self, "_flush_cb"):
            try:
                self._flush_cb()
            except Exception:
                pass  # never block on flush

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

    @property
    def queued(self) -> int:
        return self._pending.qsize()

    @property
    def completed(self) -> int:
        return len(self._completed)

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
        self.planner: Optional["ProceduralAgenda"] = None  # Set later by AML
        self.logger = mcp_client.logger  # Add logger reference

    def set_planner(self, planner: "ProceduralAgenda") -> None:
        """Link the planner for contradiction escalation to BLOCKER goals."""
        self.planner = planner

    def set_agent(self, agent: "AgentMasterLoop") -> None:
        """Link the main agent for auto-linking capabilities."""
        self.agent = agent

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Convert a base tool name to the UMS-prefixed version."""
        return f"ums:{base_tool_name}"

    # ---------------------------------------------------------------- public

    async def maybe_reflect(self, turn_ctx: Dict[str, Any]) -> None:
        """Generate a reflection memory when cadence or stuckness criteria hit."""
        
        # Check if reflection triggered by loop detection
        loop_triggered = turn_ctx.get("loop_detected", False)
        
        conditions = [
            self.state.loop_count - self.state.last_reflection_turn >= REFLECTION_INTERVAL,
            self.state.stuck_counter >= STALL_THRESHOLD,
            loop_triggered  # New condition
        ]
        
        if not any(conditions):
            return

        # Use contradictions from context instead of fetching separately
        # This harmonizes with the new _gather_context method
        contradictions = turn_ctx.get('contradictions', [])
        
        # Only fetch contradictions if not already provided by context
        if not contradictions:
            try:
                # Try using the UMS contradictions tool first
                contradictions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("query_graph_by_link_type"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "link_type": "CONTRADICTS",
                        "limit": 20,
                    },
                )
                if contradictions_res.get("success") and contradictions_res.get("data"):
                    contradictions = contradictions_res["data"].get("pairs", [])
                    # Convert to list of tuples if needed
                    contradictions = [(p["a"], p["b"]) for p in contradictions]
                else:
                    contradictions = []
            except Exception:
                # Fallback to the MemoryGraphManager's detect_inconsistencies method
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
            
        reflection_prompt = self._build_reflection_prompt(turn_ctx)
                
        # Add loop-specific prompt enhancement
        if loop_triggered:
            reflection_prompt += f"""
            
            CRITICAL: Action loop detected! Recent actions: {self.state.recent_action_signatures[-5:]}
            
            To break this loop, consider:
            1. What assumption might be wrong?
            2. What different approach could work?
            3. Is there missing information needed?
            4. Should we skip this and move to next goal?
            
            Be specific about the alternative approach."""
            
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "next_steps": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["summary", "next_steps", "confidence"],
        }
        result = await self.llm.fast_structured_call(reflection_prompt, schema)
        await self._store_reflection(result)
        self.state.last_reflection_turn = self.state.loop_count
        # reset stuck counter after reflection
        self.state.stuck_counter = 0

        # Optimize working memory periodically to keep it focused
        if self.state.loop_count % 5 == 0:  # Every 5 turns
            try:
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    "optimize_working_memory",
                    {
                        "context_id": self.state.workflow_id,
                        "strategy": "balanced"
                    }
                )
                self.logger.debug("Optimized working memory")
            except Exception as e:
                self.logger.debug(f"Working memory optimization failed (non-critical): {e}")

        # Schedule lightweight graph upkeep (doesn't need smart model)
        if self.state.loop_count % GRAPH_MAINT_EVERY == 0:
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

    def _build_reflection_prompt(self, ctx: Dict[str, Any]) -> str:
        """Compose a concise reflection prompt for the cheap LLM."""
        # Use recent_actions_text instead of recent_actions (which is a list)
        recent_actions = ctx.get("recent_actions_text", "")
        # Use working_memory_summary instead of working_memory (which is a dict)
        memory_snips = ctx.get("working_memory_summary", "")
        prompt = textwrap.dedent(
            f"""
            You are the agent's inner voice tasked with reflection.
            Current phase: {self.state.phase}
            Loop: {self.state.loop_count}
            Graph-health: {getattr(self.state, 'graph_health', 0.9):.2f}
            Recent actions: {recent_actions[:800]}
            Working memory highlights: {memory_snips[:800]}

            1. Summarise progress in 2-3 sentences.
            2. Recommend immediate next step.
            3. Give numeric confidence 0-1.
            Respond as JSON with keys summary, next_steps, confidence.
            """
        ).strip()
        if ctx.get("has_contradictions"):
            # Use contradictions instead of contradictions_list
            contradictions = ctx.get("contradictions", [])
            prompt += (
                "\nDetected potential contradictions in working memory. "
                f"IDs: {contradictions[:3]}.  Analyse whether one of them "
                "blocks progress and propose a resolution strategy."
            )
        return prompt

    async def _store_reflection(self, data: Dict[str, Any]) -> None:
        """Persist reflection into UMS as both memory and thought."""
        content = data["summary"] + "\nNEXT: " + data["next_steps"]
        
        # Access the AgentMasterLoop's auto-linking method through agent reference
        if hasattr(self, 'agent') and hasattr(self.agent, '_store_memory_with_auto_linking'):
            # Use auto-linking for better semantic connections
            try:
                await self.agent._store_memory_with_auto_linking(
                    content=content,
                    memory_type="reflection",
                    memory_level="episodic",
                    importance=min(max(float(data.get("confidence", 0.5)), 0.1), 1.0),
                    description="Automated self-reflection"
                )
                return
            except Exception as e:
                self.logger.warning(f"Auto-linking reflection storage failed, using fallback: {e}")
        
        # Fallback to direct storage
        try:
            await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("store_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "content": content,
                    "memory_level": "episodic",
                    "memory_type": "reflection",
                    "importance": min(max(float(data.get("confidence", 0.5)), 0.1), 1.0),
                    "description": "Automated self-reflection",
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to store reflection: {e}")

    def _infer_expected_artifact_type(self, goal_description: str) -> Optional[str]:
        """Infer what type of artifact a goal expects to create."""
        goal_lower = goal_description.lower()
        
        type_patterns = {
            "file": ["file", "document", "save", "export"],
            "code": ["code", "script", "program", "function", "implement"],
            "text": ["report", "article", "write", "summary", "essay"],
            "json": ["json", "data structure", "configuration"],
            "visualization": ["chart", "graph", "visualiz", "plot", "diagram"],
            "web": ["html", "webpage", "website", "interactive"],
        }
        
        for artifact_type, patterns in type_patterns.items():
            if any(pattern in goal_lower for pattern in patterns):
                return artifact_type
        
        return None
        
    async def _goal_completed(self) -> bool:
        """Check if the current goal is completed with proper evidence and artifact validation."""
        try:
            # Get goal details via UMS tool
            goal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "get_goal_details",
                {"workflow_id": self.state.workflow_id, "goal_id": self.state.current_leaf_goal_id}
            )
            if not goal_res.get("success") or not goal_res.get("data"):
                return False
                
            goal_data = goal_res["data"]
            status = goal_data.get("status", "")
            if status != "completed":
                return False
            
            # Use the enhanced validation method
            is_valid, reason = await self._validate_goal_outputs(self.state.current_leaf_goal_id)
            
            if not is_valid:
                self.logger.info(f"Goal completion validation failed: {reason}")
                
                # Store detailed validation failure
                await self.mcp_client._execute_tool_and_parse_for_agent(
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
                
                return False
            
            # Additional edge-aware progress check: verify goal has evidence links
            links_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "get_linked_memories",
                {
                    "workflow_id": self.state.workflow_id, 
                    "memory_id": self.state.current_leaf_goal_id,
                    "direction": "outgoing"
                }
            )
            
            if not links_res.get("success") or not links_res.get("data"):
                return False
            
            # Look for evidence that this goal contributed to parent goal
            evidence_links = [
                link for link in links_res["data"].get("links", [])
                if link.get("link_type") in ["CONSEQUENCE_OF", "SUPPORTS"] 
                and link.get("target_memory_id") != self.state.current_leaf_goal_id  # avoid self-loops
            ]
            
            if not evidence_links:
                self.logger.info("Goal lacks evidence links - completion may be premature")
                return False
            
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
                if meta_res.get("success") and meta_res.get("data"):
                    current_meta = meta_res["data"].get("metadata", {})
                else:
                    current_meta = {}
                count = current_meta.get(pair_key, 0) + 1
                
                # Update count
                current_meta[pair_key] = count
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME, 
                    "update_workflow_metadata",
                    {"workflow_id": self.state.workflow_id, "metadata": current_meta}
                )
                
                # Escalate if threshold reached
                if count >= 3 and self.planner:
                    blocker_title = f"RESOLVE: Contradiction {pair[0][:8]}↔{pair[1][:8]}"
                    blocker_desc = f"Persistent contradiction detected {count} times. Requires explicit resolution."
                    await self.planner.add_goal(blocker_title, blocker_desc, priority=1)  # highest priority

                    # Tag both memories so other components can suppress them
                    for mem in pair:
                        try:
                            await self.mcp_client._execute_tool_and_parse_for_agent(
                                UMS_SERVER_NAME,
                                "add_tag_to_memory",
                                {"workflow_id": self.state.workflow_id,
                                 "memory_id": mem,
                                 "tag": "BLOCKER"},
                            )
                        except Exception:
                            pass
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
                "get_goal_details",
                {"workflow_id": self.state.workflow_id, "goal_id": goal_id}
            )
            
            if not goal_res.get("success") or not goal_res.get("data"):
                return False, "Could not retrieve goal details"
                
            goal_data = goal_res["data"]
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
                
                if artifacts_res.get("success"):
                    artifacts = artifacts_res.get("data", {}).get("artifacts", [])
                    goal_created = goal_data.get("created_at", 0)
                    recent_artifacts = [
                        a for a in artifacts 
                        if a.get("created_at", 0) >= goal_created
                    ]
                    
                    if not recent_artifacts:
                        return False, f"Goal expects artifact creation but none found since goal creation"
                else:
                    return False, "Could not verify artifacts for creation goal"
            
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
            
            if memories_res.get("success") and memories_res.get("data"):
                memories = memories_res["data"].get("links", [])
                if len(memories) < 2:  # At least goal creation + some evidence
                    return False, "Insufficient memory trace for goal completion"
            else:
                return False, "Could not verify memory traces for goal"
            
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
            
            if resolution_res.get("success") and resolution_res.get("data"):
                resolution_mem_id = resolution_res["data"].get("memory_id")
                
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
            
        except Exception as e:
            self.logger.warning(f"Failed to resolve contradiction between {mem_a} and {mem_b}: {e}")


class GraphReasoner:
    """
    Performs higher-order reasoning over the cognitive graph maintained in UMS.

    Responsibilities
    ----------------
    • Convert adjacency snapshots into a `networkx` digraph for algorithmic metrics
    • Compute node centrality / community clusters to surface salient memories
    • Query the SMART or FAST LLMs (via `LLMOrchestrator`) for narrative and
      decision-making that *ground* in the graph structure
    • Suggest next actions, detect plan gaps, and provide human-readable
      explanations of relationships

    Parameters
    ----------
    mcp_client : MCPClient
        Main interface to MCP for tool calls.
    orchestrator : LLMOrchestrator
        Dual-model orchestrator used for all LLM calls (big and structured).
    mem_graph : MemoryGraphManager
        Memory graph manager for UMS operations.
    """

    # --------------------------------------------------------------------- init

    def __init__(self, mcp_client, orchestrator, mem_graph):
        self.mcp_client = mcp_client
        self.llms = orchestrator
        self.mem_graph = mem_graph

    # ----------------------------------------------------------------- builders

    @staticmethod
    def _snapshot_to_digraph(snapshot: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a `networkx.DiGraph` out of the `snapshot_context_graph()` result.

        The snapshot dict should use keys:
            • 'nodes': List[{id, type, importance, tags, ...}]
            • 'edges': List[{source, target, link_type, strength}]
        Returns
        -------
        nx.DiGraph
        """
        g = nx.DiGraph()
        for node in snapshot["nodes"]:
            g.add_node(
                node["id"],
                **{k: v for k, v in node.items() if k != "id"},
            )
        for edge in snapshot["edges"]:
            g.add_edge(
                edge["source"],
                edge["target"],
                link_type=edge.get("link_type", "related"),
                strength=edge.get("strength", 1.0),
            )
        return g

    async def build_argument_tree(self, claim_id: str, depth: int = 2) -> str:
        """
        Produce a plantUML-compatible text diagram of the supports/contradicts chain
        rooted at `claim_id` up to `depth` hops.
        """
        snap = await self.mem_graph.snapshot_context_graph()
        g = self._snapshot_to_digraph(snap)
        sub = nx.ego_graph(g, claim_id, radius=depth, directed=True)
        lines = ["@startmindmap"]
        for n in sub.nodes:
            label = sub.nodes[n].get("snippet", "")[:60].replace("\n", " ")
            lines.append(f"  * {n[:6]}: {label}")
            for succ in sub.successors(n):
                edge = sub[n][succ]["link_type"]
                lines.append(f"    ** ({edge}) ➜ {succ[:6]}")
        lines.append("@endmindmap")
        return "\n".join(lines)
                

    # --------------------------------------------------------- public services

    async def analyse_graph(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute structural metrics & clusters.

        Returns
        -------
        dict with:
            • centrality: Dict[node_id, score]
            • communities: List[List[node_id]]  (Louvain/top modular communities)
            • summary_stats: Dict[str, Any]
        """
        g = self._snapshot_to_digraph(snapshot)
        centrality = nx.pagerank(g, alpha=0.85)
        # Greedy modularity community detection
        communities = list(nx.community.louvain_communities(g))
        density = nx.density(g)
        avg_deg = statistics.mean(dict(g.degree()).values()) if g.nodes else 0

        return {
            "centrality": centrality,
            "communities": [list(c) for c in communities],
            "summary_stats": {
                "node_count": g.number_of_nodes(),
                "edge_count": g.number_of_edges(),
                "density": density,
                "avg_degree": avg_deg,
            },
        }

    async def evaluate_plan_progress(
        self,
        snapshot: Dict[str, Any],
        current_goal_id: str,
        max_tokens: int = 350,
    ) -> Dict[str, Any]:
        """
        Ask the SMART LLM to critique progress relative to current goal.

        Combines graph metrics, linked memories, and embedded goal content.
        """
        graph_info = await self.analyse_graph(snapshot)
        # NEW graph-health score (simple proxy = 1-density*0.5)
        health = 1.0 - graph_info["summary_stats"]["density"] * 0.5
        self.llms.state.graph_health = max(0.0, min(1.0, health))
        prompt = (
            "You are a meta-reasoning assistant. Using the following memory graph "
            "(with centrality scores and community clusters) evaluate how close "
            f"the agent is to achieving the goal `{current_goal_id}` and list up "
            "to 3 key blockers or missing steps.\n\n"
            f"GRAPH_INFO:\n```json\n{graph_info}\n```\n"
            "Respond as JSON: "
            "{'progress_level': 'low|medium|high', 'blockers': [..]}."
        )
        schema = {
            "type": "object",
            "properties": {
                "progress_level": {"type": "string"},
                "blockers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["progress_level", "blockers"],
        }
        return await self.llms.fast_structured_call(prompt, schema)

    async def suggest_next_actions(
        self,
        snapshot: Dict[str, Any],
        current_goal_id: str,
        limit: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Produce a ranked list of recommended next actions.

        Returns
        -------
        List[{'title': str, 'tool_hint': str, 'rationale': str}]
        """
        # Hide memories tagged as BLOCKER to avoid redundant suggestions
        snapshot["nodes"] = [
            n for n in snapshot["nodes"]
            if "tags" not in n or "BLOCKER" not in n["tags"]
        ]
        
        graph_info = await self.analyse_graph(snapshot)
        prompt = (
            "You are an autonomous agent strategist. Given the current memory "
            f"graph and the active goal `{current_goal_id}`, suggest up to {limit} "
            "next concrete actions. Each action should include a short title, an "
            "MCP/UMS tool you recommend to achieve it, and a brief rationale.\n\n"
            f"GRAPH_INFO:\n```json\n{graph_info}\n```\n"
            f"Respond as JSON list of max {limit} objects with keys "
            "title, tool_hint, rationale."
        )
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "tool_hint": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["title", "tool_hint", "rationale"],
            },
            "minItems": 1,
            "maxItems": limit,
        }
        return await self.llms.fast_structured_call(prompt, schema)

    async def explain_relationship(self, snapshot: Dict[str, Any], node_a: str, node_b: str) -> str:
        """
        Use the SMART model to narratively explain how two memories/thoughts are
        related, referencing link types and graph paths.
        """
        g = self._snapshot_to_digraph(snapshot)
        if not (g.has_node(node_a) and g.has_node(node_b)):
            return f"No path because at least one node is missing."

        try:
            path = nx.shortest_path(g, source=node_a, target=node_b)
        except nx.NetworkXNoPath:
            path = None

        description = {
            "node_a": g.nodes[node_a],
            "node_b": g.nodes[node_b],
            "path": path,
            "edges": [g.get_edge_data(path[i], path[i + 1]) for i in range(len(path) - 1)] if path else None,
        }
        prompt = (
            "Explain in plain language how the first memory relates to the second "
            "based on the graph data below. Reference link types explicitly.\n\n"
            f"REL_DATA:\n```json\n{description}\n```\n"
            "Return a concise paragraph (max 120 words)."
        )
        messages = [
            {"role": "system", "content": "You are an explanatory AI assistant."},
            {"role": "user", "content": prompt},
        ]
        # Use big model for richer narrative
        decision = await self.llms.big_reasoning_call(messages, tool_schemas=None)
        return decision.get("content", "") if isinstance(decision, dict) else str(decision)

    async def summarise_subgraph(self, snapshot: Dict[str, Any], focus_nodes: List[str], target_tokens: int = 250) -> str:
        """
        Compresses the subgraph centred on `focus_nodes` into a narrative summary.

        Uses the UMS summarize_text tool for token control.
        """
        sub_nodes = set(focus_nodes)
        for node in focus_nodes:
            sub_nodes.update(n for n in snapshot["nodes"] if n["id"] == node)
        # Build minimal snapshot
        sub_snapshot = {
            "nodes": [n for n in snapshot["nodes"] if n["id"] in sub_nodes],
            "edges": [e for e in snapshot["edges"] if e["source"] in sub_nodes and e["target"] in sub_nodes],
        }
        text_blob = repr(sub_snapshot)
        
        try:
            result = await self.mcp_client._execute_tool_and_parse_for_agent(
            UMS_SERVER_NAME,
                self._get_ums_tool_name("summarize_text"),
            {
                    "text": text_blob,
                "target_tokens": target_tokens,
                    "context": "graph_snapshot",
                "workflow_id": self.llms.state.workflow_id,
                }
            )
            if result.get("success") and result.get("data"):
                return result["data"].get("summary", text_blob[:target_tokens])
            else:
                # Fallback to simple truncation if UMS tool fails
                return text_blob[:target_tokens] + ("..." if len(text_blob) > target_tokens else "")
        except Exception:
            # Fallback to simple truncation if UMS tool is unavailable
            return text_blob[:target_tokens] + ("..." if len(text_blob) > target_tokens else "")


###############################################################################
# SECTION 6. Orchestrators & ToolExecutor (outline)
###############################################################################


class ToolExecutor:
    def __init__(self, mcp_client, state: AMLState, mem_graph):
        self.mcp_client = mcp_client
        self.state = state
        self.mem_graph = mem_graph

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Convert a base tool name to the UMS-prefixed version."""
        return f"ums:{base_tool_name}"

    async def run(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """
        Execute a tool and handle memory creation/linking.
        
        Parameters
        ----------
        tool_name : str
            The original MCP tool name (e.g., "store_memory")
        tool_args : dict
            Tool arguments
            
        Returns
        -------
        Any
            The tool execution result
        """
        # Determine the server name - for UMS tools, use UMS_Server
        # Common UMS tool names (without ums: prefix)
        ums_tools = {
            "store_memory", "get_memory_by_id", "update_memory", "query_memories",
            "create_memory_link", "get_linked_memories", "update_memory_link_metadata", 
            "get_memory_metadata", "update_memory_metadata", "get_memory_tags",
            "create_embedding", "get_embedding", "vector_similarity",
            "create_goal", "update_goal_status", "get_goals", "query_goals", "get_goal_details",
            "create_workflow", "update_workflow_metadata", "get_workflow_metadata",
            "query_graph_by_link_type", "get_contradictions", "decay_link_strengths",
            "add_tag_to_memory", "summarize_context_block", "get_recent_memories_with_links",
            "get_subgraph", "get_similar_memories", "update_workflow_status"
        }
        
        if tool_name in ums_tools:
            server_name = UMS_SERVER_NAME
            # For UMS tools, pass the tool name directly - MCPClient handles server:tool mapping
            actual_tool_name = tool_name
        else:
            # For non-UMS tools, try to determine the server or use a default
            server_name = "Unknown_Server"  # MCPClient should handle server resolution
            actual_tool_name = tool_name
        
        # Execute the tool via MCPClient
        result_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
            server_name, actual_tool_name, tool_args
        )
        
        # Check if the tool execution was successful
        if not result_envelope.get("success", False):
            error_msg = result_envelope.get("error_message", "Unknown tool execution error")
            self.mcp_client.logger.error(f"Tool {tool_name} failed: {error_msg}")
            
            # Create an error memory for failed tool execution
            try:
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME, 
                    self._get_ums_tool_name("store_memory"), 
                    {
                        'workflow_id': self.state.workflow_id,
                        'content': f"TOOL ERROR: {tool_name} failed with: {error_msg}",
                        'memory_type': 'ERROR_LOG',
                        'memory_level': 'working',
                        'importance': 3.0,
                        'description': f"Tool execution error for {tool_name}",
                    }
                )
            except Exception as memory_error:
                self.mcp_client.logger.warning(f"Failed to create error memory: {memory_error}")
            
            # Return the error envelope for the caller to handle
            return result_envelope

        # ------------------------------------------------------------------
        # After successful tool call → create an ACTION memory + link to goal
        # ------------------------------------------------------------------
        try:
            # Create ACTION memory using UMS tools
            action_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                "store_memory",
                {
                    'workflow_id': self.state.workflow_id,
                    'content': json.dumps({"tool": tool_name, "args": tool_args})[:1500],
                    'memory_level': 'working',
                    'memory_type': 'ACTION',
                    'description': f"Tool call {tool_name}",
                    'importance': 4.0,
                }
            )
            
            if action_memory_res.get("success") and action_memory_res.get("data"):
                action_mem_id = action_memory_res["data"].get("memory_id")
                
                if action_mem_id:
                    # Link provenance: goal -> action
                    await self.mem_graph.auto_link(
                        src_id=self.state.current_leaf_goal_id,
                        tgt_id=action_mem_id,
                        context_snip="attempts to satisfy goal via tool",
                    )

            # Link inputs -> action (if args reference memory ids)
            for val in tool_args.values():
                if isinstance(val, str) and val.startswith("mem_"):  # crude check
                    await self.mem_graph.auto_link(
                        src_id=val,
                        tgt_id=action_mem_id,
                        context_snip="input to tool",
                    )

                    # Link action -> outputs (if result contains memory_ids)
                    result_data = result_envelope.get("data", {})
                    if isinstance(result_data, dict) and "memory" in result_data and "memory_id" in result_data["memory"]:
                        output_id = result_data["memory"]["memory_id"]
            await self.mem_graph.auto_link(
                src_id=action_mem_id,
                tgt_id=output_id,
                context_snip="tool output",
                kind_hint=LinkKind.SEQUENTIAL,
            )
                        
        except Exception as memory_error:
            # Log memory creation/linking errors but don't fail the tool execution
            self.mcp_client.logger.warning(f"Failed to create action memory or links for {tool_name}: {memory_error}")

        # Return the actual tool result
        return result_envelope

    async def run_parallel(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple independent tools in parallel.
        
        Parameters
        ----------
        tool_calls : List[Dict[str, Any]]
            Each dict must have 'tool_name' and 'tool_args' keys.
            Optional 'tool_id' key for tracking.
        
        Returns
        -------
        Dict[str, Any]
            {
                "success": bool,
                "results": List[Dict[str, Any]],  # Ordered results matching input order
                "timing": Dict[str, float],       # Execution times per tool
                "batch_memory_id": str            # ID of the batch execution memory
            }
        """
        if not tool_calls:
            return {"success": True, "results": [], "timing": {}, "batch_memory_id": None}
        
        start_time = time.time()
        
        # Create tasks with proper error handling
        tasks = []
        tool_identifiers = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call['tool_name']
            tool_args = call.get('tool_args', {})
            tool_id = call.get('tool_id', f"{tool_name}_{i}")
            
            # Create coroutine for this tool
            async def execute_with_timing(name=tool_name, args=tool_args, tid=tool_id):
                t_start = time.time()
                try:
                    result = await self.run(name, args)
                    return {
                        "tool_id": tid,
                        "tool_name": name,
                        "success": result.get("success", False),
                        "result": result,
                        "execution_time": time.time() - t_start,
                        "error": None
                    }
                except Exception as e:
                    self.mcp_client.logger.error(f"Parallel execution error for {name}: {e}")
                    return {
                        "tool_id": tid,
                        "tool_name": name,
                        "success": False,
                        "result": None,
                        "execution_time": time.time() - t_start,
                        "error": str(e)
                    }
            
            tasks.append(execute_with_timing())
            tool_identifiers.append(tool_id)
        
        # Execute all tasks concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Create batch execution memory
        total_time = time.time() - start_time
        successful_count = sum(1 for r in task_results if r["success"])
        
        batch_summary = (
            f"Parallel execution of {len(tool_calls)} tools in {total_time:.2f}s\n"
            f"Success: {successful_count}/{len(tool_calls)}\n"
            f"Tools: {', '.join(r['tool_name'] for r in task_results)}"
        )
        
        batch_mem_res = await self.mcp_client._execute_tool_and_parse_for_agent(
            UMS_SERVER_NAME,
            self._get_ums_tool_name("store_memory"),
            {
                'workflow_id': self.state.workflow_id,
                'content': batch_summary,
                'memory_type': 'ACTION_BATCH',
                'memory_level': 'working',
                'importance': 5.0,
                'metadata': {
                    'tool_count': len(tool_calls),
                    'total_time': total_time,
                    'success_count': successful_count,
                    'tool_times': {r['tool_id']: r['execution_time'] for r in task_results}
                }
            }
        )
        
        batch_memory_id = batch_mem_res.get("data", {}).get("memory", {}).get("memory_id") if batch_mem_res.get("success") else None

        # Link batch memory to current goal
        if batch_memory_id and self.state.current_leaf_goal_id:
            await self.mem_graph.auto_link(
                src_id=self.state.current_leaf_goal_id,
                tgt_id=batch_memory_id,
                context_snip="parallel tool execution for goal",
                kind_hint=LinkKind.SEQUENTIAL
            )
        
        # Build ordered results matching input order
        ordered_results = []
        timing_info = {}
        
        for i, tool_id in enumerate(tool_identifiers):
            for result in task_results:
                if result["tool_id"] == tool_id:
                    ordered_results.append(result["result"] if result["result"] else {
                        "success": False,
                        "error": result["error"]
                    })
                    timing_info[tool_id] = result["execution_time"]
                    break
        
        return {
            "success": successful_count > 0,  # At least one succeeded
            "results": ordered_results,
            "timing": timing_info,
            "batch_memory_id": batch_memory_id
        }

class LLMOrchestrator:
    def __init__(self, mcp_client, state: AMLState):
        self.mcp_client = mcp_client
        self.state = state
        
        # Use MCPClient config for fast model with fallback
        fast_model = getattr(mcp_client.config, 'default_cheap_and_fast_model', DEFAULT_FAST_MODEL)
        self._fast_query = StructuredCall(mcp_client, model_name=fast_model)

    async def fast_structured_call(self, prompt: str, schema: dict[str, Any]):
        """Thin wrapper that delegates to `StructuredCall.query` which now uses MCPClient infrastructure."""
        result = await self._fast_query.query(prompt, schema)
        # Cost tracking is now handled by MCPClient, no need to accumulate here
        return result

    async def big_reasoning_call(self, messages: list[dict], tool_schemas=None, model_name: str = None):
        """
        Single entry for SMART model calls; automatically uses MCPClient infrastructure.
        
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
        
        try:
            # Use MCPClient's process_agent_llm_turn method which handles the full agent interaction
            resp = await self.mcp_client.process_agent_llm_turn(
                prompt_messages=messages, 
                tool_schemas=tool_schemas, 
                model_name=target_model
            )
            # Cost tracking is automatically handled by MCPClient.process_streaming_query
            return resp
        except Exception as e:
            # If process_agent_llm_turn doesn't exist or has different signature, 
            # fall back to basic query_llm
            try:
                # Convert messages to a simple prompt for basic query_llm
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    prompt_text = last_message.get("content", "")
                else:
                    prompt_text = ""
                
                resp = await self.mcp_client.query_llm(
                    prompt_text, 
                    model_override=target_model,
                    max_tokens=2000  # reasonable default for reasoning calls
                )
                return {"content": resp}
            except Exception as fallback_e:
                raise RuntimeError(f"Failed to call MCPClient LLM methods: {e}, fallback also failed: {fallback_e}") from e

###############################################################################
# SECTION 7. Procedural Agenda & Planner helpers (outline)
###############################################################################


def _ts() -> int:  # current Unix epoch in seconds
    return int(time.time())


@dataclass(order=True)
class _PQItem:
    """
    Internal priority-queue wrapper.

    The queue orders by:
        1. `priority` (lower is higher priority)
        2. `sequence`  (creation order within same parent / priority)
    """

    priority: int
    sequence: int
    goal_id: str = field(compare=False)


@dataclass
class Goal:
    """Lightweight in-memory mirror of the `goals` table record."""

    goal_id: str
    parent_goal_id: Optional[str]
    title: str
    description: str
    status: GoalStatus
    priority: int
    sequence_number: int
    created_at: int
    updated_at: int

    # The queue container will set this
    _pq_key: Optional[_PQItem] = field(init=False, default=None)


# ---------------------------------------------------------------------------
# ProceduralAgenda
# ---------------------------------------------------------------------------


class ProceduralAgenda:
    """
    Manages a living queue of goals / sub-goals for the AgentMasterLoop.

    Features
    --------
    • Pulls from / syncs with UMS `goals` table (via MCP util calls)
    • Priority-queue for fast retrieval of next actionable goal
    • Supports dynamic reprioritisation and sequence assignment
    • Emits convenience helpers for other components (e.g. `GraphReasoner`)

    Parameters
    ----------
    mcp_client:
        The MCP client / SDK instance able to call UMS tools.
    state:
        The shared AMLState object (used for workflow_id context).
    """

    # ------------------------------------------------------------------ init

    def __init__(self, mcp_client, state: AMLState):
        self.mcp_client = mcp_client
        self.state = state
        self._goals: Dict[str, Goal] = {}
        self._pq: List[_PQItem] = []
        self._seq_counter: int = self._initial_seq_value()

        # Note: _load_from_ums() will be called when goals are actually needed
        # since it contains async operations that can't be called from __init__

    # ----------------------------------------------------------- public API

    # ---- CRUD operations --------------------------------------------------

    async def add_goal(
        self,
        title: str,
        description: str,
        priority: int = 3,
        parent_goal_id: Optional[str] = None,
    ) -> str:
        """
        Create a new goal both in memory *and* in the UMS persistence layer.

        Returns
        -------
        goal_id : str
        """
        goal_id = str(uuid.uuid4())
        seq = self._next_sequence(parent_goal_id)

        goal = Goal(
            goal_id=goal_id,
            parent_goal_id=parent_goal_id,
            title=title,
            description=description,
            status=GoalStatus.PLANNED,
            priority=priority,
            sequence_number=seq,
            created_at=_ts(),
            updated_at=_ts(),
        )
        self._store_goal(goal)
        await self._persist_goal(goal)
        return goal_id

    async def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        goal = self._goals[goal_id]
        goal.status = status
        goal.updated_at = _ts()
        await self._persist_goal(goal)

    async def reprioritise_goal(self, goal_id: str, new_priority: int) -> None:
        goal = self._goals[goal_id]
        if new_priority == goal.priority:
            return
        goal.priority = new_priority
        goal.updated_at = _ts()
        # remove old pq item and re-insert
        self._remove_from_queue(goal)
        self._insert_to_queue(goal)
        await self._persist_goal(goal)

    # ---- retrieval & iteration -------------------------------------------

    async def next_goal(self) -> Optional[Goal]:
        """Pop the highest-priority PLANNED/ACTIVE goal from queue."""
        while self._pq:
            top = heapq.heappop(self._pq)
            goal = self._goals.get(top.goal_id)
            if goal and goal.status in {GoalStatus.PLANNED, GoalStatus.ACTIVE}:
                goal.status = GoalStatus.ACTIVE
                goal.updated_at = _ts()
                await self._persist_goal(goal)
                return goal
        return None

    def active_goals(self) -> List[Goal]:
        """Return a snapshot list of all ACTIVE goals ordered by priority."""
        return sorted(
            (g for g in self._goals.values() if g.status == GoalStatus.ACTIVE),
            key=lambda g: (g.priority, g.sequence_number),
        )

    def all_goals(self) -> List[Goal]:
        return list(self._goals.values())

    # ------------------------------------------------------------------ internals

    # ---- queue helpers ----------------------------------------------------

    def _insert_to_queue(self, goal: Goal):
        item = _PQItem(goal.priority, goal.sequence_number, goal.goal_id)
        goal._pq_key = item
        heapq.heappush(self._pq, item)

    def _remove_from_queue(self, goal: Goal):
        if goal._pq_key and goal._pq_key in self._pq:
            self._pq.remove(goal._pq_key)
            heapq.heapify(self._pq)
            goal._pq_key = None

    # ---- sequence handling ------------------------------------------------

    def _initial_seq_value(self) -> int:
        return max((g.sequence_number for g in self._goals.values()), default=0)

    def _next_sequence(self, parent_goal_id: Optional[str]) -> int:
        self._seq_counter += 1
        return self._seq_counter

    # ---- storage abstractions --------------------------------------------

    def _store_goal(self, goal: Goal):
        self._goals[goal.goal_id] = goal
        self._insert_to_queue(goal)

    # ---- UMS persistence layer -------------------------------------------

    async def _persist_goal(self, goal: Goal):
        """Create or update goal in UMS via MCP tools.
        
        Uses separate create and update operations as UMS likely has distinct tools for these.
        """
        try:
            # Try to create a new goal first
            create_result = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("create_goal"),
                {
            "workflow_id": self.state.workflow_id,
            "goal_id": goal.goal_id,
            "parent_goal_id": goal.parent_goal_id,
            "title": goal.title,
            "description": goal.description,
            "status": goal.status.value,
            "priority": goal.priority,
            "sequence_number": goal.sequence_number,
                    "created_at": goal.created_at,
            "updated_at": goal.updated_at,
        }
            )
            
            # If create fails (goal might already exist), try update
            if not create_result.get("success"):
                update_result = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("update_goal_status"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "goal_id": goal.goal_id,
                        "status": goal.status.value,
                        "priority": goal.priority,
                        "updated_at": goal.updated_at,
                    }
                )
                if not update_result.get("success"):
                    # Both create and update failed, log warning
                    self.mcp_client.logger.warning(f"Failed to persist goal {goal.goal_id}: both create_goal and update_goal_status failed")
                    
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to persist goal {goal.goal_id}: {e}")
        
        # Optional: dump simple markdown checklist after every change
        # This is useful for debugging but can be disabled for production
        self._maybe_dump_todo_markdown()

    def _maybe_dump_todo_markdown(self):
        """Optional markdown checklist generation for debugging purposes."""
        try:
            # Only create if in development/debug mode (check via environment or config)
            if os.environ.get("AGENT_DEBUG_MODE") or getattr(self.mcp_client.config, 'debug_mode', False):
                with open("todo.md", "w", encoding="utf-8") as fh:
                    fh.write("# Agent Goals\n\n")
                    for g in sorted(self._goals.values(), key=lambda x: x.sequence_number):
                        ck = "x" if g.status == GoalStatus.COMPLETED else " "
                        priority_marker = "!" * min(g.priority, 3)  # Visual priority indicator
                        fh.write(f"- [{ck}] {priority_marker} {g.title}  <!-- {g.goal_id} -->\n")
                        if g.description and g.description != g.title:
                            fh.write(f"  - {g.description}\n")
        except OSError:
            pass  # never block core loop on FS hiccup

    async def _load_from_ums(self):
        """
        Fetch existing goals for the workflow and seed in-memory structures.

        Uses UMS goal retrieval tools with proper error handling and structure validation.
        """
        try:
            # Try the standard get_goals tool first
            result = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_goals"),
                {"workflow_id": self.state.workflow_id},
            )
            
            if not result.get("success"):
                # Try alternative tool names if the first fails
                alt_result = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME, 
                    self._get_ums_tool_name("query_goals"),
                    {"workflow_id": self.state.workflow_id}
                )
                if alt_result.get("success"):
                    result = alt_result
                    
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to load goals from UMS: {e}")
            return  # Start with empty goal set
            
        # Extract goals from result with robust structure handling
            goals = []
        if result.get("success") and result.get("data"):
            data = result["data"]
            if isinstance(data, dict):
                # Try different possible structures
                goals = (data.get("goals") or 
                        data.get("goal_list") or 
                        data.get("items") or 
                        [])
            elif isinstance(data, list):
                goals = data
        
        # Process each goal with validation
        for g in goals:
            if not isinstance(g, dict) or "goal_id" not in g:
                continue  # Skip malformed goal entries
                
            try:
                goal = Goal(
                    goal_id=g["goal_id"],
                    parent_goal_id=g.get("parent_goal_id"),
                        title=g.get("title", "Untitled Goal"),
                        description=g.get("description", ""),
                        status=GoalStatus(g.get("status", "planned")),
                    priority=g.get("priority", 3),
                    sequence_number=g.get("sequence_number", 0),
                    created_at=g.get("created_at", _ts()),
                    updated_at=g.get("updated_at", _ts()),
                )
                self._goals[goal.goal_id] = goal
                if goal.status in {GoalStatus.PLANNED, GoalStatus.ACTIVE}:
                    self._insert_to_queue(goal)
            except (ValueError, KeyError) as e:
                self.mcp_client.logger.warning(f"Skipping malformed goal {g.get('goal_id', 'unknown')}: {e}")
                continue

    # ----------------------------------------------------------------- dunder

    def __len__(self):
        return len(self._goals)

    def __repr__(self) -> str:  # pragma: no cover
        items = "\n  ".join(
            f"[{g.status.name:<9}] p={g.priority:<2} {g.title} ({g.goal_id})" for g in sorted(self._goals.values(), key=lambda x: x.sequence_number)
        )
        return f"<ProceduralAgenda {len(self)} goals>\n  {items}"


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

    def __init__(self, mcp_client, default_llm_model_string: str, agent_state_file: str) -> None:
        """
        Initialize the AgentMasterLoop with MCP client integration.
        
        Parameters
        ----------
        mcp_client : MCPClient
            The MCP client instance for tool execution and LLM calls
        default_llm_model_string : str
            Default model name to use for reasoning calls
        agent_state_file : str
            Path to the local state file for persistence
        """
        self.mcp_client = mcp_client
        self.default_llm_model = default_llm_model_string
        self.agent_state_file = agent_state_file
        self.logger = logging.getLogger("AML")
        
        # Agent state - will be loaded/initialized in initialize()
        self.state = AMLState()
        
        # Component instances - initialized in _initialize_components()
        self.mem_graph: Optional[MemoryGraphManager] = None
        self.tool_exec: Optional[ToolExecutor] = None
        self.llms: Optional[LLMOrchestrator] = None
        self.async_queue: Optional[AsyncTaskQueue] = None
        self.metacog: Optional[MetacognitionEngine] = None
        self.planner: Optional[ProceduralAgenda] = None
        self.graph_reasoner: Optional[GraphReasoner] = None
        
        # Tool schemas cache
        self.tool_schemas: List[Dict[str, Any]] = []
        self.ums_tool_schemas: Dict[str, Dict[str, Any]] = {}
        
        # Tool effectiveness tracking
        self._tool_effectiveness_cache: Dict[str, Dict[str, int]] = {}

        # Tool categorization map  
        self._tool_category_map = self._build_tool_categories()

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
            # Load persistent state
            await self._load_state_from_file()
            
            # Validate loaded state against UMS
            await self._validate_and_reset_state()
            
            # Initialize components with validated state
            self._initialize_components()
            
            # Refresh tool schemas
            await self._refresh_tool_schemas()
            
            # Load existing goals from UMS if we have a workflow
            if self.state.workflow_id and self.planner:
                try:
                    await self.planner._load_from_ums()
                    self.logger.debug(f"Loaded {len(self.planner)} goals from UMS for workflow {self.state.workflow_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to load goals from UMS during initialization: {e}")
            
            # Save validated/updated state
            await self._save_state_internal()
            
            self.logger.info(f"AgentMasterLoop initialized with workflow_id: {self.state.workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"AgentMasterLoop initialization failed: {e}", exc_info=True)
            return False

    async def _load_state_from_file(self) -> None:
        """Load agent state from persistent file using aiofiles."""
        try:
            import json

            import aiofiles
            
            if os.path.exists(self.agent_state_file):
                async with aiofiles.open(self.agent_state_file, 'r') as f:
                    state_data = json.loads(await f.read())
                
                # Reconstruct AMLState from saved data
                # Convert datetime string back to datetime object
                if 'created_at' in state_data:
                    state_data['created_at'] = _dt.datetime.fromisoformat(state_data['created_at'])
                
                # Create new state with loaded data
                # Convert phase string back to enum if present
                if 'phase' in state_data and isinstance(state_data['phase'], str):
                    try:
                        state_data['phase'] = Phase(state_data['phase'])
                    except ValueError:
                        state_data['phase'] = Phase.UNDERSTAND  # fallback
                loaded_state = AMLState(**{k: v for k, v in state_data.items() if hasattr(AMLState, k)})
                self.state = loaded_state
                self.logger.info(f"Loaded agent state from {self.agent_state_file}")
            else:
                self.logger.info(f"No existing state file found at {self.agent_state_file}, using default state")
                
        except Exception as e:
            self.logger.warning(f"Failed to load agent state from {self.agent_state_file}: {e}")
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
            
            if not workflow_res.get("success"):
                self.logger.warning(f"Workflow {self.state.workflow_id} not found in UMS, resetting state")
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
                
                if not goal_res.get("success"):
                    self.logger.warning(f"Root goal {self.state.root_goal_id} not found in UMS, resetting goals")
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

    def _initialize_components(self) -> None:
        """Initialize all agent components with the validated state."""
        # Initialize core components
        self.mem_graph = MemoryGraphManager(self.mcp_client, self.state)
        self.async_queue = AsyncTaskQueue(max_concurrency=6)
        # auto-flush graph after each drain so turns see consistent graph state
        self.async_queue.inject_flush_cb(self.mem_graph.flush)

        # Create orchestrators / engines
        self.llms = LLMOrchestrator(self.mcp_client, self.state)
        self.tool_exec = ToolExecutor(self.mcp_client, self.state, self.mem_graph)
        self.metacog = MetacognitionEngine(self.mcp_client, self.state, self.mem_graph, self.llms, self.async_queue)
        self.planner = ProceduralAgenda(self.mcp_client, self.state)
        self.graph_reasoner = GraphReasoner(self.mcp_client, self.llms, self.mem_graph)

        # Link components after initialization
        self.metacog.set_planner(self.planner)
        self.metacog.set_agent(self)

    async def _refresh_tool_schemas(self) -> None:
        """Refresh tool schemas from MCPClient for LLM calls."""
        try:
            # Determine the provider for the current model
            provider = self.mcp_client.get_provider_from_model(self.default_llm_model)
            if not provider:
                self.logger.warning(f"Could not determine provider for model {self.default_llm_model}")
                provider = "openai"  # fallback
                
            # Use the CORRECT method that actually exists
            self.tool_schemas = self.mcp_client._format_tools_for_provider(provider)
            
            if not self.tool_schemas:
                self.logger.warning("No tool schemas returned from _format_tools_for_provider")
                self.tool_schemas = []
                
            self.logger.debug(f"Refreshed {len(self.tool_schemas)} tool schemas using provider {provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh tool schemas: {e}")
            self.tool_schemas = []
            
    def _build_tool_categories(self) -> Dict[str, List[str]]:
        """Categorize tools by their primary function."""
        categories = {
            "search": ["web_search", "search_files", "browse", "google", "search_"],
            "file": ["read_file", "write_file", "create_", "edit_file", "list_dir"],
            "memory": ["store_memory", "get_memory", "query_memories", "create_memory_link"],
            "analysis": ["execute_python", "analyze_", "summarize_", "extract_"],
            "document": ["convert_document", "generate_", "create_artifact"],
            "goal": ["create_goal", "update_goal_status", "get_goal"],
        }
        return categories

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Construct the full MCP tool name for UMS tools."""
        return f"{UMS_SERVER_NAME}:{base_tool_name}"        

    async def _rank_tools_for_goal(self, goal_desc: str, phase: Phase, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Intelligently select the most relevant tools for the current goal and phase.
        
        Returns tools sorted by relevance score.
        """
        # Start with phase-appropriate categories
        phase_categories = {
            Phase.UNDERSTAND: ["search", "file", "analysis"],
            Phase.PLAN: ["goal", "memory"],
            Phase.EXECUTE: ["file", "document", "analysis", "search"],
            Phase.REVIEW: ["memory", "goal"],
        }
        
        relevant_categories = phase_categories.get(phase, ["search", "file", "memory"])
        
        # Quick filter by category
        category_tools = []
        other_tools = []
        
        for tool in self.tool_schemas:
            # Handle both flat and nested tool schema structures
            if "function" in tool:
                tool_name = tool["function"].get("name", "unknown").lower()
            else:
                tool_name = tool.get("name", "unknown").lower()  
            in_category = False
            
            for cat, patterns in self._tool_category_map.items():
                if cat in relevant_categories:
                    if any(pattern in tool_name for pattern in patterns):
                        category_tools.append(tool)
                        in_category = True
                        break
            
            if not in_category:
                other_tools.append(tool)
        
        # If we have few enough tools, just return them
        if len(category_tools) <= limit:
            return category_tools + other_tools[:limit - len(category_tools)]
        
        # Use fast LLM to rank within category
        tool_summaries = []
        for i, tool in enumerate(category_tools[:20]):  # Cap at 20 for context
            # Use the same extraction logic as above
            if "function" in tool:
                name = tool["function"].get("name", f"tool_{i}")
                desc = tool["function"].get("description", "")[:100]
            else:
                name = tool.get("name", f"tool_{i}")
                desc = tool.get("description", "")[:100]
            tool_summaries.append(f"{i}: {name} - {desc}")
        
        ranking_prompt = f"""Current goal: {goal_desc}
    Current phase: {phase.value}

    Rank the top {limit} most useful tools for achieving this goal.
    Consider:
    1. Direct relevance to the goal
    2. Appropriate for current phase ({phase.value})
    3. Likely to produce concrete progress

    Tools:
    {chr(10).join(tool_summaries)}

    Return indices of top {limit} tools in order of usefulness."""

        schema = {
            "type": "object",
            "properties": {
                "ranked_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": limit
                },
                "reasoning": {"type": "string"}  # For debugging
            },
            "required": ["ranked_indices"]
        }
        
        try:
            result = await self.llms.fast_structured_call(ranking_prompt, schema)
            indices = result.get("ranked_indices", [])
            
            # Get tools by indices
            ranked_tools = []
            for idx in indices:
                if 0 <= idx < len(category_tools):
                    ranked_tools.append(category_tools[idx])
            
            # Add effectiveness scores from cache
            for tool in ranked_tools:
                # Use consistent extraction logic
                if "function" in tool:
                    tool_name = tool["function"].get("name", "unknown")
                else:
                    tool_name = tool.get("name", "unknown")
                    
                cache_key = f"{goal_desc[:50]}:{tool_name}"
                
                if cache_key in self._tool_effectiveness_cache:
                    stats = self._tool_effectiveness_cache[cache_key]
                    if stats['total'] > 0:
                        tool['effectiveness_score'] = stats['success'] / stats['total']
                
            # Fill remaining slots with other tools
            remaining = limit - len(ranked_tools)
            if remaining > 0:
                ranked_tools.extend(other_tools[:remaining])
                
            return ranked_tools
            
        except Exception as e:
            self.logger.warning(f"Tool ranking failed, using category defaults: {e}")
            return category_tools[:limit]

    async def record_tool_effectiveness(self, goal_desc: str, tool_name: str, success: bool):
        """Track tool effectiveness for future ranking."""
        cache_key = f"{goal_desc[:50]}:{tool_name}"
        
        if cache_key not in self._tool_effectiveness_cache:
            self._tool_effectiveness_cache[cache_key] = {"success": 0, "total": 0}
        
        self._tool_effectiveness_cache[cache_key]["total"] += 1
        if success:
            self._tool_effectiveness_cache[cache_key]["success"] += 1
        
        # Persist to UMS for long-term learning
        await self.mcp_client._execute_tool_and_parse_for_agent(
            UMS_SERVER_NAME,
            self._get_ums_tool_name("store_memory"),
            {
                "workflow_id": self.state.workflow_id,
                "content": f"Tool {tool_name} {'succeeded' if success else 'failed'} for goal: {goal_desc[:200]}",
                "memory_type": "tool_effectiveness", 
                "memory_level": "semantic",  # Long-term learning
                "importance": 3.0,
                "metadata": {
                    "tool_name": tool_name,
                    "goal_desc_hash": hash(goal_desc[:50]),
                    "success": success
                }
            }
        )
        
    async def _save_state_internal(self) -> None:
        """Save current agent state to persistent file."""
        try:
            import json

            import aiofiles
            
            # Convert state to serializable dict
            state_dict = dataclasses.asdict(self.state)
            # Convert datetime to string
            if 'created_at' in state_dict:
                state_dict['created_at'] = self.state.created_at.isoformat()
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.agent_state_file), exist_ok=True)
            
            async with aiofiles.open(self.agent_state_file, 'w') as f:
                await f.write(json.dumps(state_dict, indent=2))
                
            self.logger.debug(f"Saved agent state to {self.agent_state_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save agent state: {e}")

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
                "get_working_memory",
                {
                    "workflow_id": self.state.workflow_id,
                    "focus_goal_id": self.state.current_leaf_goal_id,
                    "limit": 20
                }
            )
            
            working_memory_ids = []
            if working_memory_res.get("success") and working_memory_res.get("data"):
                memories = working_memory_res["data"].get("memories", [])
                working_memory_ids = [mem.get("memory_id") for mem in memories if mem.get("memory_id")]
            
            # Create checkpoint memory
            checkpoint_content = {
                "checkpoint_reason": reason,
                "workflow_id": self.state.workflow_id,
                "current_goal_id": self.state.current_leaf_goal_id,
                "root_goal_id": self.state.root_goal_id,
                "phase": self.state.phase.value,
                "loop_count": self.state.loop_count,
                "working_memory_ids": working_memory_ids[:15],  # Limit to avoid huge checkpoints
                "recent_actions": self.state.recent_action_signatures[-5:],
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
            
            if checkpoint_res.get("success") and checkpoint_res.get("data"):
                checkpoint_id = checkpoint_res["data"].get("memory_id")
                self.logger.info(f"Created checkpoint {checkpoint_id[:8]} for reason: {reason}")
                return checkpoint_id
            
        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {e}")
        
        return None

    # ---------------------------------------------------------------- run-loop

    async def run_main_loop(self, overall_goal: str, max_mcp_loops: int) -> Optional[Dict[str, Any]]:
        """
        Main preparation method for agent reasoning turn.
        
        This method prepares all the context and returns the parameters needed
        for MCPClient.process_agent_llm_turn() rather than calling the LLM directly.
        
        Parameters
        ----------
        overall_goal : str
            The natural-language goal for this specific task activation.
        max_mcp_loops : int
            Maximum number of loops allowed for this activation (budget from MCPClient).
            
        Returns
        -------
        Tuple[List[Dict[str, str]], List[Dict[str, Any]], bool, Optional[str], Dict[str, Dict[str, Any]]]
            Returns (prompt_messages, tool_schemas, force_structured_output, force_tool_choice, ums_tool_schemas)
            for MCPClient.process_agent_llm_turn()
        """
        self.logger.info(f"[AML] Preparing agent turn for goal: {overall_goal[:100]}...")
        
        # Ensure we have a valid workflow and root goal
        await self._ensure_workflow_and_goal(overall_goal)
        
        # Prepare context for this turn
        self.state.loop_count += 1
        loop_idx = self.state.loop_count
        self.logger.debug("==== TURN %s  | phase=%s ====", loop_idx, self.state.phase)

        # Check if we should shutdown
        if self._shutdown_event.is_set():
            self.logger.info("[AML] Shutdown event set, stopping preparation")
            # Return empty parameters to signal stop
            return [], [], False, None, {}

        try:
            # 0. Finish/background cheap tasks -----------------------------------
            await self.async_queue.drain()

            # 1. Gather context ---------------------------------------------------
            context = await self._gather_context()

            # 2. Maybe spawn new micro-tasks (runs in background) -----------------
            await self._maybe_spawn_fast_tasks(context)

            # 3. Build reasoning messages & tool schemas --------------------------
            messages = self._build_messages(context)
            tool_schemas = await self._get_tool_schemas()

            # 4. Check for loops and add to context if detected
            if len(self.state.recent_action_signatures) >= 3:
                loop_detected, loop_info = self._detect_action_loops()
                if loop_detected:
                    context["loop_detected"] = True
                    context["loop_info"] = loop_info

            # 5. Return parameters for MCPClient to handle
            return {
                "prompt_messages": messages,
                "tool_schemas": tool_schemas or [],
                "force_structured_output": False,
                "force_tool_choice": None,
                "ums_tool_schemas": self.ums_tool_schemas
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing agent turn {loop_idx}: {e}", exc_info=True)
            self.state.last_error_details = {"error": str(e), "turn": loop_idx}
            self.state.consecutive_error_count += 1
            
            # Return empty parameters to signal error
            return {
                    "prompt_messages": [],
                    "tool_schemas": [],
                    "force_structured_output": False,
                    "force_tool_choice": None,
                    "ums_tool_schemas": {}
                }

    async def _ensure_workflow_and_goal(self, overall_goal: str) -> None:
        """Ensure we have a valid workflow and root goal, creating them if needed."""
        if not self.state.workflow_id:
            self.logger.info("Creating new workflow for goal")
            await self._create_workflow_and_goal(overall_goal)
        elif self.state.needs_replan:
            self.logger.info("Workflow exists but needs replanning")
            # Could update the goal description or create sub-goals here
            self.state.needs_replan = False

        # Create goal if workflow exists but no root goal
        if self.state.workflow_id and not self.state.root_goal_id:
            self.logger.info("Workflow exists but no root goal - creating root goal")
            await self._create_root_goal_only(overall_goal)

        # Ensure we have a current leaf goal if we have a root goal
        if self.state.root_goal_id and not self.state.current_leaf_goal_id:
            self.state.current_leaf_goal_id = self.state.root_goal_id
            self.logger.info(f"Set current_leaf_goal_id to root_goal_id: {self.state.root_goal_id}")            

    async def _create_workflow_and_goal(self, overall_goal: str) -> None:
        """Create a new UMS workflow and root goal for the current task."""
        try:
            # 1. Create workflow
            wf_resp = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("create_workflow"),
                {
                    "title": f"Agent Task – {overall_goal[:60]}",
                    "description": overall_goal,
                },
            )
            
            if not wf_resp.get("success") or not wf_resp.get("data"):
                raise RuntimeError("Failed to create workflow in UMS")
                
            self.state.workflow_id = wf_resp["data"]["workflow_id"]

            # 2. Create root goal
            goal_resp = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("create_goal"),
                {
                    "workflow_id": self.state.workflow_id,
                    "title": "Complete Agent Task",
                    "description": overall_goal,
                    "status": "active",
                    "priority": 1,
                },
            )
            
            if not goal_resp.get("success") or not goal_resp.get("data"):
                raise RuntimeError("Failed to create root goal in UMS")
                
            self.state.root_goal_id = goal_resp["data"]["goal"]["goal_id"]
            self.state.current_leaf_goal_id = self.state.root_goal_id
            self.state.needs_replan = False
            
            self.logger.info(f"Created workflow {self.state.workflow_id} and root goal {self.state.root_goal_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow and goal: {e}")
            self.state.last_error_details = {"error": str(e), "context": "workflow_creation"}
            self.state.consecutive_error_count += 1
            raise

    async def _create_root_goal_only(self, overall_goal: str) -> None:
        """Create only the root goal for an existing workflow."""
        try:
            goal_resp = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("create_goal"),
                {
                    "workflow_id": self.state.workflow_id,
                    "title": "Complete Agent Task",
                    "description": overall_goal,
                    "status": "active",
                    "priority": 1,
                },
            )
            
            if not goal_resp.get("success") or not goal_resp.get("data"):
                raise RuntimeError("Failed to create root goal in UMS")
                
            self.state.root_goal_id = goal_resp["data"]["goal"]["goal_id"]
            self.state.current_leaf_goal_id = self.state.root_goal_id
            self.state.needs_replan = False
            
            self.logger.info(f"Created root goal {self.state.root_goal_id} for existing workflow {self.state.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create root goal: {e}")
            self.state.last_error_details = {"error": str(e), "context": "root_goal_creation"}
            self.state.consecutive_error_count += 1
            raise

    # -------------------------------------------------------------- single turn

    async def _turn(self) -> str:
        """Executes exactly one *agent turn* (one SMART-model reasoning)."""
        self.state.loop_count += 1
        loop_idx = self.state.loop_count
        self.logger.debug("==== TURN %s  | phase=%s ====", loop_idx, self.state.phase)

        try:
            # 0. Finish/background cheap tasks -----------------------------------
            await self.async_queue.drain()

            # 1. Gather context ---------------------------------------------------
            context = await self._gather_context()

            # 2. Maybe spawn new micro-tasks (runs in background) -----------------
            await self._maybe_spawn_fast_tasks(context)

            # 3. Build reasoning messages & tool schemas --------------------------
            messages = self._build_messages(context)
            tool_schemas = self._get_tool_schemas()

            # 4. Call SMART model via MCPClient ----------------------------------
            # Use the budget-aware model selection from MCPClient config
            max_budget = getattr(self.mcp_client.config, 'max_budget_usd', 5.0)
            if self.state.cost_usd >= max_budget * 0.8:  # Use cheaper model when near budget
                model_name = getattr(self.mcp_client.config, 'default_cheap_and_fast_model', self.default_llm_model)
            else:
                model_name = self.default_llm_model
                
            decision = await self.llms.big_reasoning_call(messages, tool_schemas, model_name=model_name)

            # 5. Enact decision & track progress ----------------------------------
            progress = await self._enact(decision)

            # 6. Metacognition & maintenance --------------------------------------
            await self.metacog.maybe_reflect(context)
            await self.metacog.assess_and_transition(progress)

            # Create checkpoint at key decision points
            if self.state.phase == Phase.COMPLETE or self.state.goal_achieved_flag:
                await self._create_checkpoint("goal_completion")
            elif self.state.consecutive_error_count >= 2:
                await self._create_checkpoint("error_recovery_point")
            elif self.state.loop_count % 15 == 0:  # Periodic checkpoints
                await self._create_checkpoint("periodic_save")

            # 7. Persist state -----------------------------------------------------
            await self._save_state_internal()

            # 8. Budget & termination checks --------------------------------------
            if self.state.phase == Phase.COMPLETE or self.state.goal_achieved_flag:
                return "finished"
                
            max_budget = getattr(self.mcp_client.config, 'max_budget_usd', 5.0)
            if self.state.cost_usd >= max_budget:
                self.logger.warning("[AML] Budget exceeded -> abort")
                return self._record_failure("budget_exceeded")
                
            max_turns = getattr(self.mcp_client.config, 'max_agent_turns', 40)
            if loop_idx >= max_turns:
                self.logger.warning("[AML] Turn limit exceeded -> abort") 
                return self._record_failure("turn_limit")
                
            # Reset error count on successful turn
            self.state.consecutive_error_count = 0
            return "continue"
            
        except Exception as e:
            self.logger.error(f"Error in agent turn {loop_idx}: {e}", exc_info=True)
            self.state.last_error_details = {"error": str(e), "turn": loop_idx}
            self.state.consecutive_error_count += 1
            
            # Fail after too many consecutive errors
            if self.state.consecutive_error_count >= 3:
                return self._record_failure("consecutive_errors")
        return "continue"

    # -------------------------------------------------- helper: gather context

    async def _gather_context(self) -> Dict[str, Any]:
        """Collects all information using the rich context package tool."""
        
        # Step 1: Use the rich context package tool - this should be our primary context source
        rich_context = {}
        try:
            rich_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_rich_context_package"),
                {
                    "workflow_id": self.state.workflow_id,
                    "focus_goal_id": self.state.current_leaf_goal_id,
                    "include_graph": True,
                    "include_recent_actions": True,
                    "include_contradictions": True,
                    "max_memories": 20
                }
            )
            if rich_res.get("success") and rich_res.get("data"):
                rich_context = rich_res["data"]
                self.logger.info(f"[Context] Rich context package loaded successfully")
            else:
                error_msg = rich_res.get("error_message", "Unknown error")
                self.logger.warning(f"[Context] Rich context package failed: {error_msg}")
        except Exception as e:
            self.logger.warning(f"[Context] Failed to get rich context package: {e}")

        # Extract context package
        context_package = rich_context.get("context_package", {})
        
        # Extract recent actions and format them
        recent_actions = context_package.get("recent_actions", [])
        recent_actions_text = "No recent actions"
        if recent_actions:
            # Format recent actions into readable text
            action_summaries = []
            for action in recent_actions[:5]:  # Limit to last 5 actions
                action_type = action.get("action_type", "unknown")
                description = action.get("description", "No description")
                status = action.get("status", "unknown")
                action_summaries.append(f"- {action_type}: {description} (Status: {status})")
            recent_actions_text = "\n".join(action_summaries)
        
        # Extract graph snapshot
        graph_snapshot = context_package.get("graph_snapshot", {"nodes": [], "edges": []})
        # Convert nodes list to dict format if needed for compatibility
        graph_nodes = {}
        if isinstance(graph_snapshot.get("nodes"), list):
            for node in graph_snapshot.get("nodes", []):
                node_id = node.get("memory_id")
                if node_id:
                    graph_nodes[node_id] = {
                        "snippet": node.get("content", "")[:2000],  # Ensure snippet exists
                        "type": node.get("memory_type", "unknown"),
                        "importance": node.get("importance", 5.0),
                        **node  # Include other fields
                    }
        else:
            # If already a dict, ensure each node has a snippet field
            for node_id, node_data in graph_snapshot.get("nodes", {}).items():
                if isinstance(node_data, dict):
                    graph_nodes[node_id] = {
                        "snippet": node_data.get("content", node_data.get("snippet", ""))[:2000],
                        "type": node_data.get("memory_type", node_data.get("type", "unknown")),
                        "importance": node_data.get("importance", 5.0),
                        **node_data
                    }
                else:
                    # Fallback for unexpected data structure
                    graph_nodes[node_id] = {
                        "snippet": str(node_data)[:2000],
                        "type": "unknown",
                        "importance": 5.0
                    }
        
        # Extract working memory from core context
        core_context = context_package.get("core_context", {})
        working_memory = {
            "important_memories": core_context.get("important_memories", []),
            "key_thoughts": core_context.get("key_thoughts", []),
            "workflow_summary": core_context.get("workflow_summary", "No summary available"),
        }
        
        # Create working memory summary
        working_memory_summary = "Context from rich package"
        if core_context.get("workflow_summary"):
            working_memory_summary = core_context["workflow_summary"][:200] + "..." if len(core_context["workflow_summary"]) > 200 else core_context["workflow_summary"]
        
        # Extract contradictions for agent awareness
        contradictions = context_package.get("contradictions", {})
        contradictions_found = contradictions.get("contradictions_found", [])
        
        # Extract and format the data from rich_context for compatibility
        context = {
            "rich_context": rich_context,
            "working_memory": working_memory,
            "goal_stack": [core_context],  # Simplified goal stack using core context
            "recent_actions": recent_actions,
            
            # Extract key fields for compatibility with existing prompt builder
            "graph_snapshot": {
                "nodes": graph_nodes,
                "edges": graph_snapshot.get("edges", []),
                "node_count": graph_snapshot.get("node_count", len(graph_nodes)),
                "edge_count": graph_snapshot.get("edge_count", len(graph_snapshot.get("edges", [])))
            },
            "top_similar": context_package.get("proactive_memories", {}).get("memories", []),
            "recent_actions_text": recent_actions_text,
            "working_memory_summary": working_memory_summary,
            
            # Add new fields from enhanced context package
            "contradictions": contradictions_found,
            "contextual_links": context_package.get("contextual_links", {}),
            "relevant_procedures": context_package.get("relevant_procedures", {}).get("procedures", []),
            
            # Metadata
            "context_retrieval_timestamp": context_package.get("retrieval_timestamp_ums_package"),
            "has_contradictions": len(contradictions_found) > 0,
            "context_sources": {
                "core_context": bool(core_context),
                "proactive_memories": bool(context_package.get("proactive_memories")),
                "graph_snapshot": bool(graph_snapshot.get("nodes")),
                "contradictions": bool(contradictions_found),
                "contextual_links": bool(context_package.get("contextual_links")),
            }
        }
        
        # Log context summary for debugging
        self.logger.info(
            f"[Context] Assembled context: "
            f"{len(recent_actions)} recent actions, "
            f"{len(context['top_similar'])} similar memories, "
            f"{context['graph_snapshot']['node_count']} graph nodes, "
            f"{len(contradictions_found)} contradictions"
        )
        
        return context

    # -------------------------------- helper: spawn background fast tasks


    async def _maybe_spawn_fast_tasks(self, ctx: Dict[str, Any]) -> None:
        """
        Fire-and-forget cheap-LLM micro-tasks that enrich the memory graph without
        blocking the main SMART-model turn.

        Tasks launched here must be:
        • inexpensive (< FAST_CALL_MAX_USD each)
        • side-effect-free beyond writing new memories / links
        • idempotent – TURN N can safely re-run even if TURN N-1 crashed
        """
        ##########################################################################
        # 1) Summarise overly long working memories
        ##########################################################################
        def _long_nodes(graph_snapshot: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
            return [
                (mid, info)
                for mid, info in graph_snapshot["nodes"].items()
                if len(info["snippet"]) > 1000
            ]

        for mem_id, info in _long_nodes(ctx["graph_snapshot"]):
            prompt = (
                "Summarise the following text in **≤ 120 words** (plain language):\n"
                "---\n"
                f"{info['snippet'][:2000]}\n"
                "---"
            )
            schema = {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            }

            async def _on_summary(res: Dict[str, str], target_id: str = mem_id) -> None:
                # Use auto-linking storage for better semantic connections
                summary_mem_id = await self._store_memory_with_auto_linking(
                    content=res["summary"],
                    memory_type="summary",
                    memory_level="working",
                    importance=5.0,
                    description=f"Auto-summary of {target_id}",
                    link_to_goal=False  # Don't auto-link to goal, manually link to original
                )
                
                if summary_mem_id:
                    # Link summary ➜ original with ELABORATES so graph queries know
                    await self.mem_graph.auto_link(
                        src_id=summary_mem_id,
                        tgt_id=target_id,
                        context_snip="machine-generated summary",
                        kind_hint=LinkKind.ELABORATES
                    )

            coro = self.llms.fast_structured_call(prompt, schema)
            self.async_queue.spawn(
                AsyncTask(f"summarise_{mem_id[:8]}", coro, callback=_on_summary)
            )

        ##########################################################################
        # 2) Detect and digest contradictions
        ##########################################################################
        # Use contradictions from context instead of separate call
        contradictions = ctx.get('contradictions', [])
        if not contradictions:
            # Fallback to original method if context doesn't have them
            contradictions = await self.mem_graph.detect_inconsistencies()
        
        for a_id, b_id in contradictions[:3]:  # cap to 3 per turn
            prompt = (
                "You are an analyst spotting inconsistent facts.\n"
                "Summarise the contradiction **concisely** and propose ONE clarifying "
                "question that, if answered, would resolve the conflict.\n\n"
                f"A: {self.mem_graph._get_memory_content(a_id)[:350]}\n"
                f"B: {self.mem_graph._get_memory_content(b_id)[:350]}"
            )
            schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["summary", "question"],
            }

            async def _on_contradiction(
                res: Dict[str, str],
                aid: str = a_id,
                bid: str = b_id,
            ) -> None:
                # Use auto-linking storage for better semantic connections
                contr_mem_id = await self._store_memory_with_auto_linking(
                    content=f"{res['summary']}\n\nCLARIFY: {res['question']}",
                    memory_type="contradiction_analysis",
                    memory_level="working",
                    importance=7.0,  # Higher importance for contradiction analysis
                    description="Automated contradiction digest",
                    link_to_goal=True  # Link to goal since contradictions block progress
                )
                
                if contr_mem_id:
                    # Link both original memories to the analysis node
                    await self.mem_graph.auto_link(
                        src_id=aid,
                        tgt_id=contr_mem_id,
                        context_snip="contradiction analysis references",
                        kind_hint=LinkKind.REFERENCES
                    )
                    await self.mem_graph.auto_link(
                        src_id=bid,
                        tgt_id=contr_mem_id,
                        context_snip="contradiction analysis references",
                        kind_hint=LinkKind.REFERENCES
                    )
            coro = self.llms.fast_structured_call(prompt, schema)
            task_name = f"contradict_{a_id[:4]}_{b_id[:4]}"
            self.async_queue.spawn(AsyncTask(task_name, coro, callback=_on_contradiction))

    # -------------------------------- helper: build SMART-model prompt

    def _build_messages(self, ctx: Dict[str, Any]) -> List[Dict[str, str]]:
        """Compose chat-completion messages fed into SMART model."""
        sys_msg = (
            "You are the high-level reasoning engine of an autonomous agent. "
            "You have access to planning context, working memories, and can "
            "call tools. Return a JSON instruction with a 'decision_type' key."
        )
        
        # Extract context fields with robust defaults
        active_goals = ctx.get('active_goals', [])
        if not active_goals:
            # Extract from goal_stack or use fallback
            goal_stack = ctx.get('goal_stack', [])
            if goal_stack and goal_stack[0]:
                active_goals = [goal_stack[0].get('workflow_title', 'Current task')]
            else:
                active_goals = ['No active goals']

        # Extract goal_path from goal_stack
        goal_path = ctx.get('goal_path', [])
        if not goal_path and ctx.get('goal_stack'):
            # Create a simple goal path from the goal stack
            goal_path = [goal.get('workflow_id', 'unknown') for goal in ctx['goal_stack'] if goal]
        
        recent_actions = ctx.get('recent_actions_text', 'No recent actions')
        working_memory = ctx.get('working_memory_summary', 'No working memory available')
        graph_snapshot = ctx.get('graph_snapshot', {'nodes': {}, 'edges': []})
        
        # Emergency fallback indicator
        emergency_note = "🚨 EMERGENCY FALLBACK MODE: UMS context tools failed\n" if ctx.get("emergency_fallback") else ""
        contradiction_note = "⚠️ Contradictions detected\n" if ctx.get("has_contradictions") else ""
        
        user_msg = (
            f"**Phase**: {self.state.phase}\n"
            f"**Active goals**: {active_goals}\n"
            f"**Recent actions**:\n{recent_actions}\n"
            f"**Goal path (leaf→root)**: {goal_path}\n"
            f"**Focused memories (most relevant to goal)**:\n{working_memory}\n"
            f"**Graph snapshot** (truncated):\n{str(graph_snapshot)[:500]}...\n\n"
            f"{emergency_note}{contradiction_note}"
            "What should be the next step? If a tool call is required, specify "
            "tool name and arguments. Else, think in prose."
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
            if goal_res.get("success") and goal_res.get("data"):
                return goal_res["data"]["goal"].get("description", "Task")
        except Exception:
            pass
        
        return "Task"
        
    async def _get_tool_schemas(self) -> Optional[List[Dict[str, Any]]]:
        """Return JSON-schema list of the most relevant tools."""
        if not self.state.current_leaf_goal_id:
            # No goal yet, return basic tools
            return self.tool_schemas[:15]
        
        # Get current goal description
        goal_desc = await self._get_current_goal_description()
        
        # Get ranked tools
        ranked_tools = await self._rank_tools_for_goal(goal_desc, self.state.phase, limit=15)
        
        return ranked_tools

    # -------------------------------- helper: enact decision from model

    def _create_action_signature(self, decision: Any) -> str:
        """Create a comparable signature for an action/decision."""
        if isinstance(decision, dict):
            dtype = decision.get("decision_type", "unknown")
            
            if "tool" in dtype.lower():
                # For tool calls, signature is tool_name + key args
                tool_name = decision.get("tool_name", "unknown")
                tool_args = decision.get("tool_args", {})
                
                # Extract key arguments (ignore IDs, timestamps)
                key_args = []
                for k, v in sorted(tool_args.items()):
                    if k not in {"workflow_id", "memory_id", "timestamp", "created_at"}:
                        if isinstance(v, str) and len(v) > 50:
                            key_args.append(f"{k}={v[:50]}...")
                        else:
                            key_args.append(f"{k}={v}")
                
                return f"{tool_name}({','.join(key_args[:3])})"  # First 3 key args
                
            elif dtype == "THOUGHT_PROCESS":
                # For thoughts, use first 50 chars
                content = str(decision.get("content", ""))[:50]
                return f"think:{content}"
                
            else:
                return f"{dtype}"
        else:
            # Plain text response
            return f"text:{str(decision)[:30]}"


    def _detect_action_loops(self) -> Tuple[bool, str]:
        """
        Detect various types of action loops.
        
        Returns (is_loop_detected, description_of_loop)
        """
        if len(self.state.recent_action_signatures) < 3:
            return False, ""
        
        recent = self.state.recent_action_signatures
        
        # Pattern 1: Exact repetition (A-A-A)
        if len(set(recent[-3:])) == 1:
            return True, f"Repeating same action 3x: {recent[-1]}"
        
        # Pattern 2: Binary loop (A-B-A-B)
        if len(recent) >= 4 and recent[-1] == recent[-3] and recent[-2] == recent[-4]:
            return True, f"Alternating between: {recent[-2]} and {recent[-1]}"
        
        # Pattern 3: Triple loop (A-B-C-A-B-C)
        if len(recent) >= 6:
            pattern = recent[-3:]
            if recent[-6:-3] == pattern:
                return True, f"Repeating pattern: {' -> '.join(pattern)}"
        
        # Pattern 4: Majority repetition (same action appears 5+ times in last 8)
        if len(recent) >= 8:
            from collections import Counter
            counts = Counter(recent[-8:])
            most_common, count = counts.most_common(1)[0]
            if count >= 5:
                return True, f"Action '{most_common}' repeated {count}/8 times"
        
        # Pattern 5: Semantic similarity (same intent, different form)
        if len(recent) >= 4:
            # Check for repeated memory operations that might indicate confusion
            memory_ops = [sig for sig in recent[-4:] if any(op in sig for op in ['store_memory', 'get_memory', 'query_memories'])]
            if len(memory_ops) >= 3:
                return True, f"Repeated memory operations: {', '.join(set(memory_ops))}"
            
            # Check for repeated search/retrieval operations
            search_ops = [sig for sig in recent[-4:] if any(op in sig for op in ['search', 'get_', 'query_', 'fetch'])]
            if len(search_ops) >= 3:
                return True, f"Repeated search/retrieval operations: {', '.join(set(search_ops))}"
                
            # Check for repeated creation attempts
            create_ops = [sig for sig in recent[-4:] if any(op in sig for op in ['create_', 'store_', 'add_'])]
            if len(create_ops) >= 3:
                return True, f"Repeated creation attempts: {', '.join(set(create_ops))}"
        
        return False, ""
        
    async def _enact(self, decision: Any) -> bool:
        """
        Execute the SMART-model output (simplified since MCPClient handles most tool execution).

        Returns
        -------
        progress_made : bool
            Heuristic flag used by metacognition.
        """
        self.logger.debug("[AML] decision raw: %s", decision)

        # Create action signature for loop detection
        action_signature = self._create_action_signature(decision)
        
        # Update recent actions (keep last N)
        self.state.recent_action_signatures.append(action_signature)
        if len(self.state.recent_action_signatures) > self.state.max_action_history:
            self.state.recent_action_signatures.pop(0)
        
        # Check for loops
        loop_detected, loop_info = self._detect_action_loops()
        if loop_detected:
            self.logger.warning(f"[AML] Loop detected: {loop_info}")
            
            # Force reflection
            self.state.stuck_counter = STALL_THRESHOLD
            
            # Store loop detection as a memory with auto-linking
            await self._store_memory_with_auto_linking(
                content=f"LOOP DETECTED: {loop_info}. Recent actions: {self.state.recent_action_signatures[-5:]}",
                memory_type="warning",
                memory_level="working",
                importance=8.0,
                description="Action loop detected - forcing reflection"
            )
            
        # If model produced a dict, we expect certain keys
        if isinstance(decision, dict):
            dtype = decision.get("decision_type", "").upper()
            
            if dtype == "TOOL_EXECUTED_BY_MCP":
                # MCPClient already executed the tool via process_agent_llm_turn, process the result
                tool_name = decision.get("tool_name", "unknown")
                tool_result = decision.get("result", {})
                success = tool_result.get("success", False)
                
                if success:
                    self.state.last_action_summary = f"Successfully executed {tool_name}"
                    # The tool result is already parsed by MCPClient's _execute_tool_and_parse_for_agent
                    return True
                else:
                    error_msg = tool_result.get("error_message", "Unknown error")
                    self.state.last_action_summary = f"Failed to execute {tool_name}: {error_msg}"
                    return False
                    
            elif dtype in {"CALL_TOOL", "TOOL_SINGLE"}:
                # MCPClient didn't handle this tool - execute it directly
                tool_name = decision["tool_name"]
                tool_args = decision.get("tool_args", {})
                self.logger.info("[AML] → executing tool %s", tool_name)
                result = await self.tool_exec.run(tool_name, tool_args)
                success = result.get("success", False)
                self.state.last_action_summary = f"Executed {tool_name}: {'success' if success else 'failed'}"
                return success
                
            elif dtype == "MULTIPLE_TOOLS_EXECUTED_BY_MCP":
                # MCPClient handled multiple tools in parallel
                results = decision.get("results", [])
                successful_count = sum(1 for r in results if r.get("success", False))
                total_count = len(results)
                
                self.state.last_action_summary = f"Parallel execution: {successful_count}/{total_count} tools succeeded"
                return successful_count > 0
            elif dtype == "THOUGHT_PROCESS":
                thought = decision.get("content", "")
                mem_id = await self._store_memory_with_auto_linking(
                    content=thought,
                    memory_type="reasoning_step",
                    memory_level="working",
                    importance=6.0,  # Slightly higher importance for explicit thoughts
                    description="Thought from SMART model"
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
            elif dtype == "DONE":
                # Before marking complete, validate
                is_valid = await self.metacog._goal_completed()
                
                if is_valid:
                    self.state.phase = Phase.COMPLETE
                    self.state.goal_achieved_flag = True
                    self.state.last_action_summary = "Task completed and validated"
                    return True
                else:
                    # Not actually done
                    self.logger.warning("Agent claimed completion but validation failed")
                    
                    # Create a corrective memory with auto-linking
                    await self._store_memory_with_auto_linking(
                        content="Premature completion attempt - validation failed. Need to create expected outputs.",
                        memory_type="correction",
                        memory_level="working",
                        importance=8.0,
                        description="Completion validation failed"
                    )
                    
                    self.state.last_action_summary = "Completion validation failed - continuing work"
                    self.state.stuck_counter += 1  # Mild penalty to encourage different approach
                    
                    return False
            elif dtype == "UPDATE_PLAN":
                # Handle agent internal planning updates
                new_plan = decision.get("plan", [])
                self.state.current_plan = new_plan
                self.state.last_action_summary = f"Updated plan with {len(new_plan)} steps"
                return True
            else:
                # Unknown decision -> still treat as progress
                self.state.last_action_summary = f"Processed unknown decision type: {dtype}"
                return True
        else:
            # If it's plain text, store as a reasoning step with auto-linking
            text = str(decision)
            mem_id = await self._store_memory_with_auto_linking(
                content=text,
                memory_type="reasoning_step",
                memory_level="working",
                importance=5.0,
                description="Unstructured reasoning output"
            )
            if mem_id:
                evid_ids = re.findall(r"mem_[0-9a-f]{8}", text)
                if evid_ids:
                    await self.mem_graph.register_reasoning_trace(
                        thought_mem_id=mem_id,
                        evidence_ids=evid_ids,
                    )
                self.state.last_action_summary = "Processed unstructured response"
                return bool(text.strip())

    # ----------------------------------------------------- after-turn misc

    def _save_state(self) -> None:
        """Persist minimal state back to UMS to allow recovery."""
        self.mcp_client._execute_tool_and_parse_for_agent(
            UMS_SERVER_NAME,
            self._get_ums_tool_name("update_workflow_metadata"),
            {
                "workflow_id": self.state.workflow_id,
                "metadata": {
                    "loop_count": self.state.loop_count,
                    "phase": self.state.phase.value,
                    "cost_usd": self.state.cost_usd,
                    "updated": int(time.time()),
                },
            },
        )
        self.mem_graph.flush()

    # -------------------------------------------------------- failure handling

    def _hard_fail(self, reason: str) -> str:
        """Marks workflow failed and returns 'failed'."""
        self.mcp_client._execute_tool_and_parse_for_agent(
            UMS_SERVER_NAME,
            self._get_ums_tool_name("update_workflow_status"),
            {"workflow_id": self.state.workflow_id, "status": "failed", "reason": reason},
        )
        return "failed"

    def _record_failure(self, reason: str) -> str:
        """
        Record a failure state, save it, return 'failed'.
        """
        self.state.last_error_details = {"reason": reason, "loop_count": self.state.loop_count}
        self.state.consecutive_error_count += 1
        
        # Try to save state with error details
        try:
            import asyncio
            asyncio.create_task(self._save_state_internal())
        except Exception:
            pass  # Don't let state saving errors mask the original failure
            
        self.logger.error(f"[AML] Task failed: {reason}")
        return "failed"

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
            await self._save_state_internal()
        except Exception as e:
            self.logger.warning(f"[AML] Error saving state during shutdown: {e}")
            
        self.logger.info("[AML] AgentMasterLoop shutdown complete")

    def _format_recent_actions(self, recent_actions: List[Dict[str, Any]]) -> str:
        """Format recent actions from UMS into readable text."""
        if not recent_actions:
            return "No recent actions"
        
        formatted = []
        for action in recent_actions:
            action_type = action.get("action_type", "unknown")
            action_id = action.get("action_id", "")[:6]
            status = action.get("status", "unknown")
            description = action.get("description", "")[:100]
            
            formatted.append(f"• {action_type} {action_id}: {status} - {description}")
        
        return "\n".join(formatted)
    
    def _format_working_memory(self, working_memory: Dict[str, Any]) -> str:
        """Format working memory from UMS into readable summary."""
        if not working_memory:
            return "No working memory available"
        
        memories = working_memory.get("memories", [])
        if not memories:
            return working_memory.get("summary", "Working memory available but no specific memories")
        
        # Get the most important memories
        sorted_memories = sorted(
            memories, 
            key=lambda m: m.get("importance", 0.0), 
            reverse=True
        )[:10]
        
        snippets = []
        for mem in sorted_memories:
            content = mem.get("content", "")[:200]
            mem_type = mem.get("memory_type", "unknown")
            snippets.append(f"[{mem_type}] {content}")
        
        return "\n".join(snippets)

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
        Store a memory and automatically create links to related memories and goals.
        
        Implements the UMS "research_and_store" pattern:
        1. Store the memory
        2. Find semantically similar memories  
        3. Auto-link to related memories
        4. Link to current goal if requested
        
        Returns the memory_id if successful, None otherwise.
        """
        try:
            # Step 1: Store the memory
            store_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("store_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "content": content,
                    "memory_type": memory_type,
                    "memory_level": memory_level,
                    "importance": importance,
                    "description": description or f"Auto-stored {memory_type}",
                }
            )
            
            if not store_res.get("success") or not store_res.get("data"):
                return None
                
            memory_id = store_res["data"].get("memory_id")
            if not memory_id:
                return None
            
            # Step 2: Find semantically similar memories (if content is substantial)
            if len(content) > 50:  # Only auto-link substantial content
                try:
                    similar_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("search_semantic_memories"),
                        {
                            "workflow_id": self.state.workflow_id,
                            "query": content[:500],  # Use first 500 chars as query
                            "k": 3,  # Get top 3 similar memories
                            "memory_level": None,  # Search all levels
                            "exclude_memory_ids": [memory_id]  # Don't match self
                        }
                    )
                    
                    if similar_res.get("success") and similar_res.get("data"):
                        similar_memories = similar_res["data"].get("memories", [])
                        
                        # Step 3: Auto-link to related memories
                        for similar_mem in similar_memories:
                            similar_id = similar_mem.get("memory_id")
                            if similar_id and similar_id != memory_id:
                                # Create link with context about similarity
                                await self.mem_graph.auto_link(
                                    src_id=memory_id,
                                    tgt_id=similar_id,
                                    context_snip="auto-linked: semantically related",
                                    kind_hint=LinkKind.RELATED
                                )
                                
                except Exception as e:
                    self.logger.debug(f"Auto-linking to similar memories failed: {e}")
            
            # Step 4: Link to current goal if requested
            if link_to_goal and self.state.current_leaf_goal_id:
                try:
                    await self.mem_graph.auto_link(
                        src_id=self.state.current_leaf_goal_id,
                        tgt_id=memory_id,
                        context_snip="goal-related memory",
                        kind_hint=LinkKind.SUPPORTS
                    )
                except Exception as e:
                    self.logger.debug(f"Auto-linking to goal failed: {e}")
            
            self.logger.debug(f"Stored memory {memory_id[:8]} with auto-linking")
            return memory_id
            
        except Exception as e:
            self.logger.warning(f"Auto-linking memory storage failed: {e}")
            return None

    # -------------------------------- helper: spawn background fast tasks


###############################################################################
# SECTION 9. Utility stubs (unchanged)
###############################################################################


def get_goal_desc(goal_id: str, *, db_path: str | Path = "ums.db") -> str:
    """
    Retrieve the *description* field for a goal given its `goal_id`.

    This helper talks **directly** to the UMS SQLite store (`ums.db`).  It is a
    convenience wrapper mostly useful in logging or prompt-building code where
    we need to display a short human-readable blurb for a goal identifier.

    Parameters
    ----------
    goal_id : str
        Primary-key of the `goals` table.
    db_path : str or pathlib.Path, optional
        Filesystem path to the UMS SQLite file.  Defaults to the canonical
        `"ums.db"` that the MCP mounts in the sandbox root.

    Returns
    -------
    str
        The description string if the goal exists, otherwise `"<unknown goal>"`.

    Notes
    -----
    • The function opens the database in **read-only** mode (URI flag) so it is
      100 % safe to call from anywhere without risking write locks.
    • For performance it only fetches the *single* column we care about.
    """
    if not os.path.exists(db_path):
        return "<unknown goal>"

    try:
        # URI read-only connection so we never interfere with MCP writers.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = lambda c, r: r[0]  # return scalar column
        cur = conn.execute("SELECT description FROM goals WHERE goal_id = ? LIMIT 1", (goal_id,))
        desc = cur.fetchone()
        return desc if desc is not None else "<unknown goal>"
    except sqlite3.Error:
        return "<unknown goal>"
    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover – extremely unlikely
            pass


def pick_model(phase: Phase, *, cost_sensitive: bool = False) -> str:
    """
    Light-weight policy helper that picks which LLM **family** to use for a
    reasoning turn **based on the current agent phase**.

    Parameters
    ----------
    phase : Phase
        Current high-level execution phase of the agent.
    cost_sensitive : bool, optional
        If *True*, the function will bias towards the *cheaper* model even for
        phases that usually prefer the smart model.  Useful for graceful
        degradation under tight budget constraints.

    Returns
    -------
    str
        The model name constant (either `DEFAULT_SMART_MODEL` or `DEFAULT_FAST_MODEL`).

    Decision Matrix
    ---------------
    +---------------+-------------------------------+
    | Phase         | Preferred Model               |
    +===============+===============================+
    | UNDERSTAND    | FAST (cheap comprehension)    |
    | PLAN          | FAST (schema-structured)      |
    | GRAPH_MAINT   | FAST (micro-ops)              |
    | EXECUTE       | SMART (tool synthesis)        |
    | REVIEW        | SMART (detailed reasoning)    |
    | COMPLETE      | FAST (final acknowledgements) |
    +---------------+-------------------------------+

    The `cost_sensitive` flag simply forces FAST for all phases except
    **EXECUTE**, because that stage often still needs the more powerful model to
    generate correct tool calls.
    """
    # Forced downgrade if caller signals budget pressure
    if cost_sensitive and phase is not Phase.EXECUTE:
        return DEFAULT_FAST_MODEL

    if phase in {Phase.UNDERSTAND, Phase.PLAN, Phase.GRAPH_MAINT, Phase.COMPLETE}:
        return DEFAULT_FAST_MODEL
    if phase in {Phase.EXECUTE, Phase.REVIEW}:
        return DEFAULT_SMART_MODEL
    # Fallback – should never happen but errs on the side of capability
    return DEFAULT_SMART_MODEL

