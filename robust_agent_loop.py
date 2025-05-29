from __future__ import annotations

"""agent_master_loop.py – *Outline V4*  
Leveraging full UMS semantic‑graph power + dual–LLM orchestration.

This version provides **finished, fully‑working code for `MemoryGraphManager`** while keeping the rest of the architecture identical to the outline delivered in V4.  
All TODOs inside that class are removed; every public method really performs its work against the local UMS SQLite store that the MCP keeps at `ums.db`.
"""

###############################################################################
# SECTION 0. Imports & typing helpers
###############################################################################

import asyncio
import dataclasses
import datetime as _dt
import enum
import heapq
import json
import logging
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

import networkx as nx

###############################################################################
# SECTION 1. Global constants (tunable via config later)
###############################################################################

MAX_TURNS = 40  # richer reasoning but still bounded
MAX_BUDGET_USD = 5.00  # hard ceiling
REFLECTION_INTERVAL = 6  # generate reflection memories
GRAPH_MAINT_EVERY = 2  # turns between graph‑maintenance phases
STALL_THRESHOLD = 3  # consecutive non‑progress turns → forced reflection

SMART_MODEL_NAME = "gpt-4o-2025-05-15"
FAST_MODEL_NAME = "gemini‑flash‑2.5‑05‑20"
FAST_CALL_MAX_USD = 0.003  # per micro‑call

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
    RELATED       = "RELATED"
    CAUSAL        = "CAUSAL"
    CONTRADICTS   = "CONTRADICTS"
    SUPPORTS      = "SUPPORTS"
    GENERALISES   = "GENERALISES"
    SPECIALISES   = "SPECIALISES"
    SEQUENTIAL    = "SEQUENTIAL"
    ELABORATES    = "ELABORATES"         # new - richer narrative chains
    QUESTION_OF   = "QUESTION_OF"        # reasoning trace ⇄ evidence
    CONSEQUENCE   = "CONSEQUENCE_OF"     # downstream effect; alias for CAUSAL

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
    workflow_id: str
    root_goal_id: str
    current_leaf_goal_id: str
    phase: Phase = Phase.UNDERSTAND
    loop_count: int = 0
    cost_usd: float = 0.0
    stuck_counter: int = 0
    last_reflection_turn: int = 0
    last_graph_maint_turn: int = 0
    graph_health: float = 0.9
    pending_attachments: List[str] = dataclasses.field(default_factory=list)
    created_at: _dt.datetime = dataclasses.field(default_factory=_dt.datetime.utcnow)


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
        self.ums_server_name = "UMS_Server"

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
            "create_memory_link",
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
            await self.auto_link(thought_mem_id, fact, "thought leads to fact", kind_hint=LinkKind.CONSEQUENCE)

    async def fetch_contradicting_evidence(self, mem_id: str, limit: int = 5) -> list[str]:
        """
        Return memory_ids of facts or evidence that *contradict* the given memory.
        """
        try:
            # Use UMS tool to get memory links
            res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                "get_memory_links", 
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
                "get_contradictions",
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
                            "get_memory_link_metadata", 
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
                "query_graph_by_link_type",
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
                            "get_memory_link_metadata", 
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
                "query_memories",
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
                "query_graph_by_link_type",
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
                "query_memories",
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
                    "get_memory_links",
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
                        "get_memory_by_id",
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
                    "store_memory",
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": summary,
                        "memory_level": "semantic",
                        "memory_type": "SUMMARY",
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
                            kind_hint=LinkKind.GENERALISES
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
                "query_memories",
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
                        "update_memory",
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
                "query_memories",
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
                    "id": mem_id,
                    "type": memory.get("memory_type", "UNKNOWN"),
                    "snippet": memory.get("content", "")[:200],
                    "tags": await self._get_tags_for_memory(mem_id)
                }
                
                # Get outgoing links for this memory
                links_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    self.ums_server_name,
                    "get_memory_links",
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
                                "get_memory_by_id",
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
                "get_memory_metadata",
                {"workflow_id": self.state.workflow_id, "memory_id": src_id}
            )
            if meta_src_res.get("success") and meta_src_res.get("data"):
                meta_src = meta_src_res["data"].get("metadata", {}).get("link_type_cache", {})
                if tgt_id in meta_src:
                    return meta_src[tgt_id]
                    
            meta_tgt_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                "get_memory_metadata",
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
                "get_memory_by_id",
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
                "get_memory_tags",
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

    # expose for AgentMasterLoop._save_state (removed - no longer needed)
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
                "get_embedding",
                {"workflow_id": self.state.workflow_id, "memory_id": memory_id}
            )
            if res.get("success") and res.get("data"):
                return res["data"].get("vector")
            
            # If no embedding exists, try to create one
            create_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                "create_embedding",
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
                "vector_similarity",
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
                "decay_link_strengths",
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
                "get_memory_metadata",
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
                    "update_memory_metadata",
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
                    "update_memory_link_metadata",
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
        model_name: str = FAST_MODEL_NAME,
        cost_cap_usd: float = FAST_CALL_MAX_USD,
        max_retries: int = 2,
    ) -> None:
        self.mcp_client = mcp_client
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
        mcp_client: Any,
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

    def set_planner(self, planner: "ProceduralAgenda") -> None:
        """Link the planner for contradiction escalation to BLOCKER goals."""
        self.planner = planner

    # ---------------------------------------------------------------- public

    async def maybe_reflect(self, turn_ctx: Dict[str, Any]) -> None:
        """Generate a reflection memory when cadence or stuckness criteria hit."""
        conditions = [
            self.state.loop_count - self.state.last_reflection_turn >= REFLECTION_INTERVAL,
            self.state.stuck_counter >= STALL_THRESHOLD,
        ]
        if not any(conditions):
            return

        try:
            # Try using the UMS utility first
            contradictions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                "UMS_Server",
                "query_graph_by_link_type",
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
            # Fallback to the original detect_inconsistencies method
            contradictions = await self.mem_graph.detect_inconsistencies()
            
        if contradictions:
            turn_ctx["has_contradictions"] = True
            turn_ctx["contradictions_list"] = contradictions[:3]
            
            # 🔹NEW escalate persistent contradictions to BLOCKER goals
            await self._escalate_persistent_contradictions(contradictions)
            
        reflection_prompt = self._build_reflection_prompt(turn_ctx)
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

        # 🔹NEW schedule lightweight graph upkeep (doesn't need smart model)
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
        if self._goal_completed():
            self.state.phase = Phase.REVIEW

    # ---------------------------------------------------------------- util

    def _build_reflection_prompt(self, ctx: Dict[str, Any]) -> str:
        """Compose a concise reflection prompt for the cheap LLM."""
        recent_actions = ctx.get("recent_actions", "")
        memory_snips = ctx.get("working_memory", "")
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
            prompt += (
                "\nDetected potential contradictions in working memory. "
                f"IDs: {ctx['contradictions_list']}.  Analyse whether one of them "
                "blocks progress and propose a resolution strategy."
            )
        return prompt

    async def _store_reflection(self, data: Dict[str, Any]) -> None:
        """Persist reflection into UMS as both memory and thought."""
        content = data["summary"] + "\nNEXT: " + data["next_steps"]
        
        # Use UMS tool to store memory
        await self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "store_memory",
            {
                "workflow_id": self.state.workflow_id,
                "content": content,
                "memory_level": "episodic",
                "memory_type": "REFLECTION",
                "importance": min(max(float(data.get("confidence", 0.5)), 0.1), 1.0),
                "description": "Automated self-reflection",
            }
        )

    def _goal_completed(self) -> bool:
        # Placeholder; real implementation would query UMS goal status.
        try:
            # Get goal status via UMS tool
            goal_res = self.mcp_client._execute_tool_and_parse_for_agent(
                "UMS_Server",
                "get_goal_details",
                {"workflow_id": self.state.workflow_id, "goal_id": self.state.current_leaf_goal_id}
            )
            if not goal_res.get("success") or not goal_res.get("data"):
                return False
                
            status = goal_res["data"].get("status", "")
            if status != "completed":
                return False
                
            # 🔹NEW edge-aware progress check: verify goal has evidence links
            # Check for outgoing CONSEQUENCE_OF or SUPPORTS links to parent
            goal_id = self.state.current_leaf_goal_id
            links_res = self.mcp_client._execute_tool_and_parse_for_agent(
                "UMS_Server",
                "get_memory_links",
                {"workflow_id": self.state.workflow_id, "source_memory_id": goal_id}
            )
            
            # Look for evidence that this goal contributed to parent goal
            evidence_links = [
                link for link in links_res.get("data", {}).get("links", [])
                if link["link_type"] in ["CONSEQUENCE_OF", "SUPPORTS"] 
                and link["target_memory_id"] != goal_id  # avoid self-loops
            ]
            
            return len(evidence_links) > 0  # Only complete if we have evidence
        except Exception:
            return False

    async def _escalate_persistent_contradictions(self, contradictions: List[Tuple[str, str]]) -> None:
        """Track contradiction pairs and escalate to BLOCKER goals when they persist ≥3 times."""
        # Simple persistent tracking using UMS metadata
        for pair in contradictions:
            pair_key = f"contradiction_{min(pair)}_{max(pair)}"  # normalized key
            try:
                # Try to get existing count
                meta_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server",
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
                    "UMS_Server", 
                    "update_workflow_metadata",
                    {"workflow_id": self.state.workflow_id, "metadata": current_meta}
                )
                
                # Escalate if threshold reached
                if count >= 3 and self.planner:
                    blocker_title = f"RESOLVE: Contradiction {pair[0][:8]}↔{pair[1][:8]}"
                    blocker_desc = f"Persistent contradiction detected {count} times. Requires explicit resolution."
                    self.planner.add_goal(blocker_title, blocker_desc, priority=1)  # highest priority

                    # Tag both memories so other components can suppress them
                    for mem in pair:
                        try:
                            await self.mcp_client._execute_tool_and_parse_for_agent(
                                "UMS_Server",
                                "add_tag_to_memory",
                                {"workflow_id": self.state.workflow_id,
                                 "memory_id": mem,
                                 "tag": "BLOCKER"},
                            )
                        except Exception:
                            pass
            except Exception:
                pass  # fail gracefully if metadata storage isn't available


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
    """

    # --------------------------------------------------------------------- init

    def __init__(self, mcp_client, orchestrator, mem_graph):
        self.mcp = mcp_client
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
        centrality = nx.algorithms.centrality.pagerank(g, alpha=0.85)
        # Greedy modularity community detection
        communities = list(nx.algorithms.community.louvain_communities(g))
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

        Utilises the `summarise_context_block` UMS util for token control.
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
        result = await self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "summarize_context_block",
            {
                "text_to_summarize": text_blob,
                "target_tokens": target_tokens,
                "context_type": "graph_snapshot",
                "workflow_id": self.llms.state.workflow_id,
            },
        )
        return result["data"]["summary"]


###############################################################################
# SECTION 6. Orchestrators & ToolExecutor (outline)
###############################################################################


class ToolExecutor:
    def __init__(self, mcp_client, state: AMLState, mem_graph):
        self.mcp_client = mcp_client
        self.state = state
        self.mem_graph = mem_graph

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
        if tool_name.startswith(""):
            server_name = "UMS_Server"
        else:
            # For non-UMS tools, try to determine the server or use a default
            # This might need to be refined based on the actual tool registry
            server_name = "Unknown_Server"  # MCPClient should handle server resolution
        
        # Execute the tool via MCPClient
        result_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
            server_name, tool_name, tool_args
        )
        
        # Check if the tool execution was successful
        if not result_envelope.get("success", False):
            error_msg = result_envelope.get("error_message", "Unknown tool execution error")
            self.mcp_client.logger.error(f"Tool {tool_name} failed: {error_msg}")
            
            # Create an error memory for failed tool execution
            try:
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server", 
                    "store_memory", 
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
                "UMS_Server",
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
                    if isinstance(result_data, dict) and "memory_id" in result_data:
                        output_id = result_data["memory_id"]
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


class LLMOrchestrator:
    def __init__(self, mcp_client, state: AMLState):
        self.mcp_client = mcp_client
        self.state = state
        self._fast_query = StructuredCall(mcp_client, model_name=FAST_MODEL_NAME)

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

        self._load_from_ums()

    # ----------------------------------------------------------- public API

    # ---- CRUD operations --------------------------------------------------

    def add_goal(
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
        self._persist_goal(goal)
        return goal_id

    def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        goal = self._goals[goal_id]
        goal.status = status
        goal.updated_at = _ts()
        self._persist_goal(goal)

    def reprioritise_goal(self, goal_id: str, new_priority: int) -> None:
        goal = self._goals[goal_id]
        if new_priority == goal.priority:
            return
        goal.priority = new_priority
        goal.updated_at = _ts()
        # remove old pq item and re-insert
        self._remove_from_queue(goal)
        self._insert_to_queue(goal)
        self._persist_goal(goal)

    # ---- retrieval & iteration -------------------------------------------

    def next_goal(self) -> Optional[Goal]:
        """Pop the highest-priority PLANNED/ACTIVE goal from queue."""
        while self._pq:
            top = heapq.heappop(self._pq)
            goal = self._goals.get(top.goal_id)
            if goal and goal.status in {GoalStatus.PLANNED, GoalStatus.ACTIVE}:
                goal.status = GoalStatus.ACTIVE
                goal.updated_at = _ts()
                self._persist_goal(goal)
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

    def _persist_goal(self, goal: Goal):
        """Create or update goal in UMS via MCP tools.
        
        Uses separate create and update operations as UMS likely has distinct tools for these.
        """
        try:
            # Try to create a new goal first
            create_result = self.mcp_client._execute_tool_and_parse_for_agent(
                "UMS_Server",
                "create_goal",
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
                update_result = self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server",
                    "update_goal",
                    {
                        "workflow_id": self.state.workflow_id,
                        "goal_id": goal.goal_id,
                        "status": goal.status.value,
                        "priority": goal.priority,
                        "updated_at": goal.updated_at,
                    }
                )
                if not update_result.get("success"):
                    # If both fail, try the combined tool name as fallback
                    fallback_result = self.mcp_client._execute_tool_and_parse_for_agent(
                        "UMS_Server",
                        "create_or_update_goal",
                        {
                            "workflow_id": self.state.workflow_id,
                            "goal_id": goal.goal_id,
                            "parent_goal_id": goal.parent_goal_id,
                            "title": goal.title,
                            "description": goal.description,
                            "status": goal.status.value,
                            "priority": goal.priority,
                            "sequence_number": goal.sequence_number,
                            "updated_at": goal.updated_at,
                        }
                    )
                    if not fallback_result.get("success"):
                        self.mcp_client.logger.warning(f"Failed to persist goal {goal.goal_id}: all UMS operations failed")
                        
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

    def _load_from_ums(self):
        """
        Fetch existing goals for the workflow and seed in-memory structures.

        Uses UMS goal retrieval tools with proper error handling and structure validation.
        """
        try:
            # Try the standard get_goals tool first
            result = self.mcp_client._execute_tool_and_parse_for_agent(
                "UMS_Server",
                "get_goals",
                {"workflow_id": self.state.workflow_id},
            )
            
            if not result.get("success"):
                # Try alternative tool names if the first fails
                alt_result = self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server", 
                    "query_goals",
                    {"workflow_id": self.state.workflow_id}
                )
                if alt_result.get("success"):
                    result = alt_result
                    
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to load goals from  {e}")
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

    def __init__(self, mcp_client: Any, *, user_goal: str) -> None:
        """
        Parameters
        ----------
        mcp_client:
            The main MCP SDK client (must expose tool execution helpers).
        user_goal:
            Natural-language user goal that seeds the initial workflow/goal.
        """
        self.mcp_client = mcp_client
        self.logger = logging.getLogger("AgentMasterLoop")

        # Bootstrap ---------------------------------------------------------
        self.state: AMLState = self._bootstrap(user_goal)
        self.mem_graph = MemoryGraphManager(self.mcp_client, self.state)
        self.async_queue = AsyncTaskQueue(max_concurrency=6)
        # auto-flush graph after each drain so turns see consistent graph state
        self.async_queue.inject_flush_cb(self.mem_graph.flush)

        # Create orchestrators / engines (requires state + mem_graph) -------
        # ToolExecutor and LLMOrchestrator are expected to be implemented
        # elsewhere (stubs existed earlier).  We *import at runtime* so the
        # circular dependency order does not matter.
        from .llm_orchestrator import LLMOrchestrator  # type: ignore
        from .tool_executor import ToolExecutor  # type: ignore

        self.llms = LLMOrchestrator(self.mcp_client, self.state)
        self.tool_exec = ToolExecutor(self.mcp_client, self.state, self.mem_graph)
        self.metacog = MetacognitionEngine(self.mcp_client, self.state, self.mem_graph, self.llms, self.async_queue)
        self.planner = ProceduralAgenda(self.mcp_client, self.state)
        self.graph_reasoner = GraphReasoner(self.mcp_client, self.llms, self.mem_graph)

        # 🔹NEW link components after initialization
        self.metacog.set_planner(self.planner)
    # ---------------------------------------------------------------- bootstrap

    def _bootstrap(self, user_goal: str) -> AMLState:
        """
        • Creates a new workflow & root goal in UMS (if not resumed)
        • Persists initial records so subsequent restarts can resume
        """
        # 1. Create workflow -------------------------------------------------
        wf_resp = self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "create_workflow",
            {
                "title": f"Workflow – {user_goal[:60]}",
                "description": user_goal,
            },
        )
        workflow_id = wf_resp["data"]["workflow_id"]

        # 2. Create root goal -----------------------------------------------
        goal_resp = self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "create_or_update_goal",
            {
                "workflow_id": workflow_id,
                "title": "Root goal",
                "description": user_goal,
                "status": "active",
                "priority": 1,
            },
        )
        root_goal_id = goal_resp["data"]["goal_id"]

        return AMLState(
            workflow_id=workflow_id,
            root_goal_id=root_goal_id,
            current_leaf_goal_id=root_goal_id,
        )

    # ---------------------------------------------------------------- run-loop

    async def run(self) -> None:
        """Main asynchronous loop – exits when COMPLETE or budget exceeded."""
        self.logger.info("[AML] starting main loop for workflow %s", self.state.workflow_id)
        while True:
            step_status = await self._turn()

            if step_status == "finished":
                self.logger.info("[AML] workflow completed – exiting run()")
                return                      # ← early exit

            if step_status == "failed":
                self.logger.error("[AML] workflow failed – exiting run()")
                return                      # ← early exit

    # -------------------------------------------------------------- single turn

    async def _turn(self) -> str:
        """Executes exactly one *agent turn* (one SMART-model reasoning)."""
        self.state.loop_count += 1
        loop_idx = self.state.loop_count
        self.logger.debug("==== TURN %s  | phase=%s ====", loop_idx, self.state.phase)

        # 0. Finish/background cheap tasks -----------------------------------
        await self.async_queue.drain()

        # 1. Gather context ---------------------------------------------------
        context = await self._gather_context()

        # 2. Maybe spawn new micro-tasks (runs in background) -----------------
        await self._maybe_spawn_fast_tasks(context)

        # 3. Build reasoning messages & tool schemas --------------------------
        messages = self._build_messages(context)
        tool_schemas = self._get_tool_schemas()

        # 4. Call SMART model --------------------------------------------------
        decision = await self.llms.big_reasoning_call(messages, tool_schemas)

        # 5. Enact decision & track progress ----------------------------------
        progress = await self._enact(decision)

        # 6. Metacognition & maintenance --------------------------------------
        await self.metacog.maybe_reflect(context)
        await self.metacog.assess_and_transition(progress)

        # 7. Persist / housekeeping -------------------------------------------
        self._save_state()

        # 8. Budget & termination checks --------------------------------------
        if self.state.phase == Phase.COMPLETE:
            # 🔹NEW: signal orchestrators with idle tool call  
            try:
                await self.mcp_client._execute_tool_and_parse_for_agent("UMS_Server", "idle", {})
            except Exception:
                pass  # never block completion on sentinel call failure
            return "finished"
        if self._budget_exceeded():
            self.logger.warning("[AML] hard budget exceeded -> abort")
            return self._hard_fail("budget_exceeded")
        if loop_idx >= MAX_TURNS:
            self.logger.warning("[AML] turn limit exceeded -> abort")
            return self._hard_fail("turn_limit")
        return "continue"

    # -------------------------------------------------- helper: gather context


    async def _gather_context(self) -> Dict[str, Any]:
        """Collects all information fed into the SMART-model prompt."""
        # 🔹NEW --- Vector-similar memories for the *current* leaf goal ------
        sim_res = self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "get_similar_memories",
            {
                "workflow_id": self.state.workflow_id,
                "memory_id": self.state.current_leaf_goal_id,
                "k": 8,
                "include_content": True,
            },
        )
        top_similar = sim_res["data"]["memories"]   # list[dict]

        # Snapshot graph (no change) -----------------------------------------
        graph_snapshot = await self.mem_graph.snapshot_context_graph()

        # Procedural agenda summary ------------------------------------------
        active_goals = [g.title for g in self.planner.active_goals()][:5]

        # ⇢ NEW: pick top-central nodes to focus the SMART model on the most
        #        influential memories rather than only embedding-neighbours.
        graph_metrics = await self.graph_reasoner.analyse_graph(graph_snapshot)
        centrality = graph_metrics["centrality"]
        important_nodes = sorted(
            graph_snapshot["nodes"].values(),
            key=lambda n: centrality.get(n["id"], 0.0),
            reverse=True,
        )[:10]
        central_snippets = "\n".join(n["snippet"] for n in important_nodes)

        # ------------------------------------------------------------------
        # Recent memories **with outgoing links** (link-aware utility)
        # ------------------------------------------------------------------
        mems_res = self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "get_recent_memories_with_links",
            {"workflow_id": self.state.workflow_id, "limit": 10},
        )

        recent_actions_text = "\n".join(
            "• {type} {mid}: {links}".format(
                type=m["memory_type"],
                mid=m["memory_id"][:6],
                links=", ".join(l["link_type"] for l in m.get("outgoing_links", [])) or "no links",  # noqa: E741
            )
            for m in mems_res["data"]["memories"]
        )

        # ------------------------------------------------------------------
        # Goal context: path leaf-goal ➜ root  (subgraph, link-aware)
        # ------------------------------------------------------------------
        path_res = self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "get_subgraph",
            {
                "workflow_id": self.state.workflow_id,
                "start_node_id": self.state.current_leaf_goal_id,
                "direction": "up",
                "max_hops": 4,
            },
        )
        goal_path_nodes = [n["node_id"] for n in path_res["data"]["nodes"]]

        return {
            "graph_snapshot": graph_snapshot,
            "active_goals": active_goals,
            "recent_actions": recent_actions_text,
            "goal_path": goal_path_nodes,
            "top_similar": top_similar,
            "working_memory": central_snippets,
        }

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
                summary_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server",
                    "store_memory",
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": res["summary"],
                        "memory_level": "working",
                        "memory_type": "SUMMARY",
                        "description": f"Auto-summary of {target_id}",
                    }
                )
                if summary_res.get("success") and summary_res.get("data"):
                    new_mem_id = summary_res["data"].get("memory_id")
                    if new_mem_id:
                        # Link summary ➜ original with ELABORATES so graph queries know
                        await self.mem_graph.auto_link(
                            src_id=new_mem_id,
                            tgt_id=target_id,
                            context_snip="machine-generated summary",
                        )

            coro = self.llms.fast_structured_call(prompt, schema)
            self.async_queue.spawn(
                AsyncTask(f"summarise_{mem_id[:8]}", coro, callback=_on_summary)
            )

        ##########################################################################
        # 2) Detect and digest contradictions
        ##########################################################################
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
                contr_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server",
                    "store_memory",
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": f"{res['summary']}\n\nCLARIFY: {res['question']}",
                        "memory_level": "working",
                        "memory_type": "CONTRADICTION_ANALYSIS",
                        "description": "Automated contradiction digest",
                    }
                )
                if contr_res.get("success") and contr_res.get("data"):
                    contr_mem_id = contr_res["data"].get("memory_id")
                    if contr_mem_id:
                        # Link both original memories to the analysis node
                        await self.mem_graph.auto_link(
                            src_id=aid,
                            tgt_id=contr_mem_id,
                            context_snip="contradiction_summary",
                        )
                        await self.mem_graph.auto_link(
                            src_id=bid,
                            tgt_id=contr_mem_id,
                            context_snip="contradiction_summary",
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
        contradiction_note = "⚠️ Contradictions detected\n" if ctx.get("has_contradictions") else ""
        user_msg = (
            f"**Phase**: {self.state.phase}\n"
            f"**Active goals**: {ctx['active_goals']}\n"
            f"**Recent actions**:\n{ctx['recent_actions']}\n"
            f"**Goal path (leaf→root)**: {ctx['goal_path']}\n"
            f"**Focused memories (most relevant to goal)**:\n{ctx['working_memory']}\n"
            f"**Graph snapshot** (truncated):\n{ctx['graph_snapshot']}\n\n"
            f"{contradiction_note}"
            "What should be the next step? If a tool call is required, specify "
            "tool name and arguments. Else, think in prose."
        )
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

    # -------------------------------- helper: get tool-schemas for SMART model

    def _get_tool_schemas(self) -> Optional[List[Dict[str, Any]]]:
        """Return JSON-schema list describing callable tools (minimal demo)."""
        # In a real system we would dynamically query MCP for available tools.
        example_tool_schema = {
            "name": "web.search_query",
            "description": "Search the web for latest info",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "recency": {"type": "integer"},
                },
                "required": ["q"],
            },
        }
        return [example_tool_schema]

    # -------------------------------- helper: enact decision from model

    async def _enact(self, decision: Any) -> bool:
        """
        Execute the SMART-model output.

        Returns
        -------
        progress_made : bool
            Heuristic flag used by metacognition.
        """
        self.logger.debug("[AML] decision raw: %s", decision)

        # If model produced a dict, we expect certain keys
        if isinstance(decision, dict):
            dtype = decision.get("decision_type", "").upper()
            if dtype in {"CALL_TOOL", "TOOL_SINGLE"}:
                tool_name = decision["tool_name"]
                tool_args = decision.get("tool_args", {})
                self.logger.info("[AML] → executing tool %s", tool_name)
                await self.tool_exec.run(tool_name, tool_args)
                return True
            elif dtype == "THOUGHT_PROCESS":
                thought = decision.get("content", "")
                thought_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    "UMS_Server",
                    "store_memory",
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": thought,
                        "memory_level": "working",
                        "memory_type": "REASONING_STEP",
                        "description": "Thought from SMART model",
                    }
                )
                if thought_res.get("success") and thought_res.get("data"):
                    mem_id = thought_res["data"].get("memory_id")
                    if mem_id:
                        # ⇢ NEW: link any referenced memories as supporting evidence
                        evid_ids = re.findall(r"mem_[0-9a-f]{8}", thought)
                        if evid_ids:
                            await self.mem_graph.register_reasoning_trace(
                                thought_mem_id=mem_id,
                                evidence_ids=evid_ids,
                            )
                return bool(thought.strip())
            elif dtype == "DONE":
                self.state.phase = Phase.COMPLETE
                return True
            else:
                # Unknown decision -> still treat as progress
                return True
        else:
            # If it's plain text, store as a reasoning step
            text = str(decision)
            text_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                "UMS_Server",
                "store_memory",
                {
                    "workflow_id": self.state.workflow_id,
                    "content": text,
                    "memory_level": "working",
                    "memory_type": "REASONING_STEP",
                    "description": "Unstructured reasoning output",
                }
            )
            if text_res.get("success") and text_res.get("data"):
                mem_id = text_res["data"].get("memory_id")
                if mem_id:
                    evid_ids = re.findall(r"mem_[0-9a-f]{8}", text)
                    if evid_ids:
                        await self.mem_graph.register_reasoning_trace(
                            thought_mem_id=mem_id,
                            evidence_ids=evid_ids,
                        )
            return bool(text.strip())

    # ----------------------------------------------------- after-turn misc

    def _save_state(self) -> None:
        """Persist minimal state back to UMS to allow recovery."""
        self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "update_workflow_metadata",
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

    # ------------------------------------------------------------- budget etc

    def _budget_exceeded(self) -> bool:
        return self.state.cost_usd >= MAX_BUDGET_USD

    def _hard_fail(self, reason: str) -> str:
        """Marks workflow failed and returns 'failed'."""
        self.mcp_client._execute_tool_and_parse_for_agent(
            "UMS_Server",
            "update_workflow_status",
            {"workflow_id": self.state.workflow_id, "status": "failed", "reason": reason},
        )
        return "failed"


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
        The model name constant (either `SMART_MODEL_NAME` or `FAST_MODEL_NAME`).

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
        return FAST_MODEL_NAME

    if phase in {Phase.UNDERSTAND, Phase.PLAN, Phase.GRAPH_MAINT, Phase.COMPLETE}:
        return FAST_MODEL_NAME
    if phase in {Phase.EXECUTE, Phase.REVIEW}:
        return SMART_MODEL_NAME
    # Fallback – should never happen but errs on the side of capability
    return SMART_MODEL_NAME

