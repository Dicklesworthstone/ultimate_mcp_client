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
import math
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from pathlib import Path

import logging

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
        Detect contradictions using UMS get_contradictions tool.
        Returns list of (memory_id_a, memory_id_b) tuples representing contradictory pairs.
        """
        try:
            contradictions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_contradictions"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 50,
                    "include_resolved": False  # Exclude already resolved contradictions
                }
            )
            
            if contradictions_res.get("success") and contradictions_res.get("data"):
                contradictions_found = contradictions_res["data"].get("contradictions_found", [])
                
                # Convert to the expected format: List[Tuple[str, str]]
                pairs = []
                for contradiction in contradictions_found:
                    mem_a = contradiction.get("memory_id_a")
                    mem_b = contradiction.get("memory_id_b")
                    if mem_a and mem_b:
                        pairs.append((mem_a, mem_b))
                
                self.mcp_client.logger.debug(f"Found {len(pairs)} contradictions via UMS")
                return pairs
            else:
                self.mcp_client.logger.warning(f"UMS contradiction detection failed: {contradictions_res.get('error_message', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.mcp_client.logger.warning(f"Error detecting contradictions via UMS: {e}")
            return []

    async def consolidate_cluster(self, min_size: int = 6) -> None:
        """
        Consolidate related memories using UMS consolidate_memories tool.
        """
        try:
            # Use UMS consolidate_memories tool instead of manual consolidation
            consolidation_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                self.ums_server_name,
                self._get_ums_tool_name("consolidate_memories"),
                {
                    "workflow_id": self.state.workflow_id,
                    "consolidation_type": "summary",  # Could be summary, insight, procedural, question
                    "source_selection": {
                        "method": "query_filter",
                        "filters": {
                            "memory_level": "working",
                            "min_link_count": min_size,
                            "limit": 50
                        }
                    },
                    "target_memory_level": "semantic",
                    "importance_threshold": 6.0,
                    "max_source_memories": 20,
                    "enable_clustering": True
                }
            )
            
            if consolidation_res.get("success"):
                consolidated_count = consolidation_res.get("data", {}).get("consolidated_memories_count", 0)
                self.mcp_client.logger.debug(f"Consolidated {consolidated_count} memory clusters")
            else:
                self.mcp_client.logger.warning(f"Memory consolidation failed: {consolidation_res.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            self.mcp_client.logger.warning(f"Error during memory consolidation: {e}")

        # Keep the existing decay call
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
        """
        Get context graph using UMS NetworkX analysis.
        """
        try:
            graph_res = await self.get_context_graph()  # Use preset!
            
            if graph_res.get("success") and graph_res.get("data"):
                data = graph_res["data"]
                
                # Convert to expected format
                nodes = {}
                for node in data.get("nodes", []):
                    node_id = node["memory_id"]
                    nodes[node_id] = {
                        "id": node_id,
                        "type": node.get("memory_type", "UNKNOWN"),
                        "snippet": node.get("content_preview", "")[:200],
                        "tags": set()  # Could enhance this later
                    }
                
                edges = [(e["source"], e["target"], e["link_type"]) for e in data.get("edges", [])]
                
                return {"nodes": nodes, "edges": edges}
            else:
                return {"nodes": {}, "edges": []}
                
        except Exception as e:
            self.mcp_client.logger.warning(f"Error getting context graph: {e}")
            return {"nodes": {}, "edges": []}

    async def get_context_graph(self, focus_node: str = None) -> Dict[str, Any]:
        """Get lightweight graph for agent context."""
        return await self.mcp_client._execute_tool_and_parse_for_agent(
            self.ums_server_name,
            self._get_ums_tool_name("get_subgraph"),
            {
                "workflow_id": self.state.workflow_id,
                "start_node_id": focus_node,
                "algorithm": "ego_graph" if focus_node else "full_graph",
                "max_hops": 2,
                "max_nodes": 25,
                "compute_centrality": True,
                "centrality_algorithms": ["pagerank"],
                "include_node_content": False
            }
        )

    async def get_analysis_graph(self) -> Dict[str, Any]:
        """Get comprehensive graph analysis."""
        return await self.mcp_client._execute_tool_and_parse_for_agent(
            self.ums_server_name,
            self._get_ums_tool_name("get_subgraph"),
            {
                "workflow_id": self.state.workflow_id,
                "algorithm": "full_graph",
                "max_nodes": 100,
                "compute_centrality": True,
                "centrality_algorithms": ["pagerank", "betweenness"],
                "detect_communities": True,
                "community_algorithm": "louvain",
                "compute_graph_metrics": True
            }
        )

    async def get_argument_graph(self, claim_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get argument tree around a claim."""
        return await self.mcp_client._execute_tool_and_parse_for_agent(
            self.ums_server_name,
            self._get_ums_tool_name("get_subgraph"),
            {
                "workflow_id": self.state.workflow_id,
                "start_node_id": claim_id,
                "algorithm": "ego_graph",
                "max_hops": depth,
                "max_nodes": 20,
                "link_type_filter": ["SUPPORTS", "CONTRADICTS", "CAUSAL", "REFERENCES"],
                "include_node_content": False
            }
        )

    async def get_relationship_graph(self, node_a: str, node_b: str) -> Dict[str, Any]:
        """Get graph for analyzing relationships between nodes."""
        return await self.mcp_client._execute_tool_and_parse_for_agent(
            self.ums_server_name,
            self._get_ums_tool_name("get_subgraph"),
            {
                "workflow_id": self.state.workflow_id,
                "start_node_id": node_a,
                "algorithm": "bfs_tree",
                "max_hops": 3,
                "max_nodes": 20,
                "include_shortest_paths": True,
                "shortest_path_targets": [node_b]
            }
        )            

    async def get_summarization_graph(self, focus_node: str, max_nodes: int = 15) -> Dict[str, Any]:
        """Get subgraph with content for summarization."""
        return await self.mcp_client._execute_tool_and_parse_for_agent(
            self.ums_server_name,
            self._get_ums_tool_name("get_subgraph"),
            {
                "workflow_id": self.state.workflow_id,
                "start_node_id": focus_node,
                "algorithm": "ego_graph",
                "max_hops": 2,
                "max_nodes": max_nodes,
                "include_node_content": True  # Key difference - includes content
            }
        )
        
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
            reflection_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("generate_reflection"),
                {
                    "workflow_id": self.state.workflow_id,
                    "reflection_type": reflection_type,
                    "recent_ops_limit": 30,
                    "max_tokens": 800
                }
            )
            
            if reflection_res.get("success"):
                reflection_id = reflection_res.get("reflection_id")
                self.mcp_client.logger.debug(f"Generated {reflection_type} reflection: {reflection_id}")
            else:
                self.mcp_client.logger.warning(f"Reflection generation failed: {reflection_res.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to generate reflection: {e}")

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
    Simplified graph reasoning that leverages UMS NetworkX analysis.
    
    All heavy lifting is now done server-side by UMS get_subgraph tool.
    """

    def __init__(self, mcp_client, orchestrator, mem_graph):
        self.mcp_client = mcp_client
        self.llms = orchestrator
        self.mem_graph = mem_graph

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Convert a base tool name to the UMS-prefixed version."""
        return f"{UMS_SERVER_NAME}:{base_tool_name}"

    async def build_argument_tree(self, claim_id: str, depth: int = 2) -> str:
        try:
            subgraph_res = await self.mem_graph.get_argument_graph(claim_id, depth)  # Use preset!
            
            if not subgraph_res.get("success") or not subgraph_res.get("data"):
                return "@startmindmap\n* Error: Could not retrieve argument tree\n@endmindmap"
                
            data = subgraph_res["data"]
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            
            # Build plantUML from UMS data
            lines = ["@startmindmap"]
            
            # Find center node
            center_node = next((n for n in nodes if n.get("memory_id") == claim_id), None)
            if center_node:
                label = center_node.get("content_preview", "")[:60].replace("\n", " ")
                lines.append(f"* {claim_id[:6]}: {label}")
                
                # Add connected nodes
                for edge in edges:
                    if edge.get("source") == claim_id:
                        target_id = edge.get("target")
                        target_node = next((n for n in nodes if n.get("memory_id") == target_id), None)
                        if target_node:
                            target_label = target_node.get("content_preview", "")[:60].replace("\n", " ")
                            edge_type = edge.get("link_type", "RELATED")
                            lines.append(f"  ** ({edge_type}) ➜ {target_id[:6]}: {target_label}")
                            
            lines.append("@endmindmap")
            return "\n".join(lines)
            
        except Exception as e:
            return f"@startmindmap\n* Error: {str(e)}\n@endmindmap"

    async def analyse_graph(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Use preset for comprehensive analysis
            analysis_res = await self.mem_graph.get_analysis_graph()  # Use preset!
                
            if analysis_res.get("success") and analysis_res.get("data"):
                data = analysis_res["data"]
                
                # Extract pre-computed analysis
                centrality = data.get("centrality", {})
                communities = data.get("communities", {})
                metrics = data.get("graph_metrics", {})
                
                # Convert to expected format
                centrality_scores = {}
                if "pagerank" in centrality:
                    centrality_scores = centrality["pagerank"].get("top_nodes", {})
                
                community_list = []
                if communities.get("communities"):
                    community_list = [c["members"] for c in communities["communities"]]
                
                return {
                    "centrality": centrality_scores,
                    "communities": community_list,
                    "summary_stats": {
                        "node_count": metrics.get("node_count", 0),
                        "edge_count": metrics.get("edge_count", 0),
                        "density": metrics.get("density", 0.0),
                        "avg_degree": metrics.get("degree_stats", {}).get("mean", 0.0)
                    }
                }
            else:
                return {
                    "centrality": {},
                    "communities": [],
                    "summary_stats": {"node_count": 0, "edge_count": 0, "density": 0.0, "avg_degree": 0.0}
                }
                
        except Exception as e:
            self.mcp_client.logger.warning(f"UMS graph analysis failed: {e}")
            return {
                "centrality": {},
                "communities": [],
                "summary_stats": {"node_count": 0, "edge_count": 0, "density": 0.0, "avg_degree": 0.0}
            }

    async def evaluate_plan_progress(self, snapshot: Dict[str, Any], current_goal_id: str, max_tokens: int = 350) -> Dict[str, Any]:
        """
        Evaluate plan progress using UMS analysis + fast LLM.
        """
        # Get analysis from UMS instead of computing manually
        graph_info = await self.analyse_graph(snapshot)
        
        prompt = (
            "You are a meta-reasoning assistant. Using the following graph analysis "
            "evaluate how close the agent is to achieving the goal and list up "
            "to 3 key blockers or missing steps.\n\n"
            f"GOAL_ID: {current_goal_id}\n"
            f"GRAPH_ANALYSIS:\n```json\n{graph_info}\n```\n"
            "Respond as JSON: {'progress_level': 'low|medium|high', 'blockers': [...]}."
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

    async def suggest_next_actions(self, snapshot: Dict[str, Any], current_goal_id: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Suggest actions using UMS graph analysis.
        """
        graph_info = await self.analyse_graph(snapshot)
        
        prompt = (
            "You are an autonomous agent strategist. Using the graph analysis below, "
            f"suggest up to {limit} next concrete actions for goal `{current_goal_id}`. "
            "Focus on high-centrality nodes and community patterns.\n\n"
            f"GRAPH_ANALYSIS:\n```json\n{graph_info}\n```\n"
            f"Respond as JSON list of max {limit} objects with keys: title, tool_hint, rationale."
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
            "maxItems": limit,
        }
        
        return await self.llms.fast_structured_call(prompt, schema)

    async def explain_relationship(self, snapshot: Dict[str, Any], node_a: str, node_b: str) -> str:
        """
        Explain relationship using UMS path analysis.
        """
        try:
            path_res = await self.mem_graph.get_relationship_graph(node_a, node_b)  # Use preset!
            
            if path_res.get("success") and path_res.get("data"):
                paths = path_res["data"].get("shortest_paths", {})
                if node_b in paths:
                    path_info = paths[node_b]
                    return f"Distance: {path_info.get('avg_distance_to', 'unknown')} hops. Reachable via graph connections."
                else:
                    return "No direct path found between these memories."
            else:
                return "Could not analyze relationship."
                
        except Exception as e:
            return f"Error analyzing relationship: {e}"

    async def summarise_subgraph(self, snapshot: Dict[str, Any], focus_nodes: List[str], target_tokens: int = 250) -> str:
        if not focus_nodes:
            return "No focus nodes provided"
            
        try:
            # Use specialized preset for summarization
            subgraph_res = await self.mem_graph.get_summarization_graph(focus_nodes[0])  # Use new preset!
            
            if not subgraph_res.get("success"):
                return "Could not retrieve subgraph for summarization"
                
            nodes = subgraph_res["data"].get("nodes", [])
            content_texts = [node.get("content", node.get("content_preview", ""))[:200] for node in nodes]
            combined_text = "\n".join(content_texts)
            
            # Use UMS summarization
            summary_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("summarize_text"),
                {
                    "text": combined_text,
                    "target_tokens": target_tokens,
                    "context": "subgraph_summary",
                    "workflow_id": self.mem_graph.state.workflow_id
                }
            )
            
            if summary_res.get("success") and summary_res.get("data"):
                return summary_res["data"].get("summary", combined_text[:target_tokens])
            else:
                return combined_text[:target_tokens] + ("..." if len(combined_text) > target_tokens else "")
                
        except Exception as e:
            return f"Summarization failed: {e}"

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
        Execute a tool with proper UMS action lifecycle tracking.
        """
        # Determine the server name for the tool
        ums_tools = {
            "store_memory", "get_memory_by_id", "update_memory", "query_memories",
            "create_memory_link", "get_linked_memories", "update_memory_link_metadata", 
            "get_memory_metadata", "update_memory_metadata", "get_memory_tags",
            "create_embedding", "get_embedding", "vector_similarity",
            "create_goal", "update_goal_status", "get_goals", "query_goals", "get_goal_details",
            "create_workflow", "update_workflow_metadata", "get_workflow_metadata",
            "query_graph_by_link_type", "get_contradictions", "decay_link_strengths",
            "add_tag_to_memory", "summarize_context_block", "get_recent_memories_with_links",
            "get_similar_memories", "update_workflow_status"
        }
        
        if tool_name in ums_tools:
            server_name = UMS_SERVER_NAME
            actual_tool_name = tool_name
        else:
            server_name = "Unknown_Server"
            actual_tool_name = tool_name
        
        # Start action tracking
        action_id = None
        try:
            action_start_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("record_action_start"),
                {
                    "workflow_id": self.state.workflow_id,
                    "action_type": "tool_use",
                    "reasoning": f"Executing tool {tool_name} to advance current goal",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "title": f"Execute {tool_name}",
                    "related_thought_id": None,  # Could link to current reasoning if available
                    "idempotency_key": f"{tool_name}_{hash(str(tool_args))}_{self.state.loop_count}"
                }
            )
            
            if action_start_res.get("success") and action_start_res.get("data"):
                action_id = action_start_res["data"].get("action_id")
                self.mcp_client.logger.debug(f"Started action tracking for {tool_name}: {action_id}")
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to start action tracking for {tool_name}: {e}")
        
        # Execute the actual tool
        try:
            result_envelope = await self.mcp_client._execute_tool_and_parse_for_agent(
                server_name, actual_tool_name, tool_args
            )
            
            success = result_envelope.get("success", False)
            
            # Complete action tracking
            if action_id:
                try:
                    completion_status = "completed" if success else "failed"
                    summary = f"Tool {tool_name} {'succeeded' if success else 'failed'}"
                    if not success:
                        error_msg = result_envelope.get("error_message", "Unknown error")
                        summary += f": {error_msg}"
                    
                    await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("record_action_completion"),
                        {
                            "action_id": action_id,
                            "status": completion_status,
                            "tool_result": result_envelope if success else None,
                            "summary": summary,
                            "conclusion_thought": None  # Could add reasoning about the result
                        }
                    )
                    self.mcp_client.logger.debug(f"Completed action tracking for {tool_name}: {completion_status}")
                except Exception as e:
                    self.mcp_client.logger.warning(f"Failed to complete action tracking for {tool_name}: {e}")
            
            # Handle tool execution results
            if not success:
                error_msg = result_envelope.get("error_message", "Unknown tool execution error")
                self.mcp_client.logger.error(f"Tool {tool_name} failed: {error_msg}")
            
            return result_envelope
            
        except Exception as e:
            # Complete action tracking with error status
            if action_id:
                try:
                    await self.mcp_client._execute_tool_and_parse_for_agent(
                        UMS_SERVER_NAME,
                        self._get_ums_tool_name("record_action_completion"),
                        {
                            "action_id": action_id,
                            "status": "failed",
                            "tool_result": None,
                            "summary": f"Tool {tool_name} failed with exception: {str(e)}",
                            "conclusion_thought": None
                        }
                    )
                except Exception:
                    pass  # Don't let action tracking errors mask the original error
            
            self.mcp_client.logger.error(f"Tool {tool_name} execution failed: {e}")
            return {"success": False, "error_message": str(e)}

    async def run_parallel(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple independent tools in parallel with UMS action tracking.
        """
        if not tool_calls:
            return {"success": True, "results": [], "timing": {}, "batch_memory_id": None}
        
        start_time = time.time()
        
        # Start batch action tracking
        batch_action_id = None
        try:
            batch_action_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("record_action_start"),
                {
                    "workflow_id": self.state.workflow_id,
                    "action_type": "parallel_tools",
                    "reasoning": f"Executing {len(tool_calls)} tools in parallel",
                    "tool_name": "parallel_execution",
                    "tool_args": {"tool_count": len(tool_calls), "tools": [tc.get("tool_name") for tc in tool_calls]},
                    "title": f"Parallel execution of {len(tool_calls)} tools"
                }
            )
            
            if batch_action_res.get("success") and batch_action_res.get("data"):
                batch_action_id = batch_action_res["data"].get("action_id")
        except Exception as e:
            self.mcp_client.logger.warning(f"Failed to start batch action tracking: {e}")
        
        # Create tasks with proper error handling
        tasks = []
        tool_identifiers = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call['tool_name']
            tool_args = call.get('tool_args', {})
            tool_id = call.get('tool_id', f"{tool_name}_{i}")
            
            # Create coroutine for this tool (using the updated run method)
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
        
        if batch_action_id:
            try:
                batch_summary = f"Parallel execution completed: {successful_count}/{len(tool_calls)} tools succeeded in {total_time:.2f}s"
                await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("record_action_completion"),
                    {
                        "action_id": batch_action_id,
                        "status": "completed" if successful_count > 0 else "failed",
                        "tool_result": {
                            "successful_count": successful_count,
                            "total_count": len(tool_calls),
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
        
        # Define the EXACT response schema we expect
        response_schema = {
            "type": "object",
            "properties": {
                "decision_type": {
                    "type": "string",
                    "enum": ["TOOL_SINGLE", "TOOL_MULTIPLE", "THOUGHT_PROCESS", "DONE"]
                },
                "tool_name": {
                    "type": "string",
                    "description": "Required if decision_type is TOOL_SINGLE"
                },
                "tool_args": {
                    "type": "object",
                    "description": "Required if decision_type is TOOL_SINGLE"
                },
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "tool_args": {"type": "object"},
                            "tool_id": {"type": "string", "description": "Unique identifier for this tool call"}
                        },
                        "required": ["tool_name", "tool_args", "tool_id"]
                    },
                    "description": "Required if decision_type is TOOL_MULTIPLE"
                },
                "content": {
                    "type": "string", 
                    "description": "Required if decision_type is THOUGHT_PROCESS or DONE"
                }
            },
            "required": ["decision_type"],
            "additionalProperties": False
        }
        
        try:
            # Use MCPClient's structured query method to enforce the schema
            resp = await self.mcp_client.query_llm_structured(
                prompt_messages=messages,
                response_schema=response_schema,
                model_override=target_model,
                use_cheap_model=False  # This is a big reasoning call
            )
            
            # The response should now be exactly what we expect
            self.mcp_client.logger.debug(f"[LLMOrch] Structured response: {resp}")
            return resp
            
        except Exception as e:
            self.mcp_client.logger.error(f"[LLMOrch] Structured call failed: {e}")
            # If structured call fails, try regular call as fallback
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
                    max_tokens=2000
                )
                
                # Try to parse as JSON, fallback to thought process
                try:
                    import json
                    parsed = json.loads(resp)
                    if "decision_type" in parsed:
                        return parsed
                except Exception:
                    pass
                
                # Return as thought process if parsing fails
                return {
                    "decision_type": "THOUGHT_PROCESS",
                    "content": resp
                }
                
            except Exception as fallback_e:
                raise RuntimeError(f"Both structured and fallback LLM calls failed: {e}, fallback: {fallback_e}") from e

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
    Simplified goal management using UMS goal tools instead of manual priority queues.
    
    This maintains the same API as the original but delegates all operations to UMS.
    """

    def __init__(self, mcp_client, state: AMLState):
        self.mcp_client = mcp_client
        self.state = state
        self.logger = mcp_client.logger

    def _get_ums_tool_name(self, base_tool_name: str) -> str:
        """Convert a base tool name to the UMS-prefixed version."""
        return f"{UMS_SERVER_NAME}:{base_tool_name}"

    # ---- CRUD operations --------------------------------------------------

    async def add_goal(
        self,
        title: str,
        description: str,
        priority: int = 3,
        parent_goal_id: Optional[str] = None,
    ) -> str:
        """Create a new goal using UMS create_goal tool."""
        try:
            goal_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("create_goal"),
                {
                    "workflow_id": self.state.workflow_id,
                    "description": description,
                    "title": title,
                    "priority": priority,
                    "parent_goal_id": parent_goal_id,
                    "initial_status": "active",
                    "reasoning": f"Added via ProceduralAgenda at loop {self.state.loop_count}"
                }
            )
            
            if goal_res.get("success") and goal_res.get("data"):
                goal_id = goal_res["data"]["goal"]["goal_id"]
                self.logger.debug(f"Created goal {goal_id}: {title}")
                return goal_id
            else:
                self.logger.error(f"Failed to create goal: {goal_res.get('error_message', 'Unknown error')}")
                return str(uuid.uuid4())  # Fallback to prevent crashes
                
        except Exception as e:
            self.logger.error(f"Error creating goal: {e}")
            return str(uuid.uuid4())  # Fallback to prevent crashes

    async def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        """Update goal status using UMS update_goal_status tool."""
        try:
            await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("update_goal_status"),
                {
                    "goal_id": goal_id,
                    "status": status.value,
                    "reason": f"Status change via ProceduralAgenda at loop {self.state.loop_count}"
                }
            )
            self.logger.debug(f"Updated goal {goal_id} status to {status.value}")
        except Exception as e:
            self.logger.warning(f"Failed to update goal {goal_id} status: {e}")

    async def reprioritise_goal(self, goal_id: str, new_priority: int) -> None:
        """Update goal priority (UMS doesn't have direct priority update, so we log it)."""
        try:
            # UMS doesn't expose priority updates directly, so we add a note
            await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("update_goal_status"),
                {
                    "goal_id": goal_id,
                    "status": "active",  # Keep current status
                    "reason": f"Priority change to {new_priority} at loop {self.state.loop_count}"
                }
            )
            self.logger.debug(f"Logged priority change for goal {goal_id} to {new_priority}")
        except Exception as e:
            self.logger.warning(f"Failed to update goal {goal_id} priority: {e}")

    # ---- retrieval & iteration -------------------------------------------

    async def next_goal(self) -> Optional[Goal]:
        """Get the highest-priority active goal using UMS query_goals."""
        try:
            goals_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("query_goals"),
                {
                    "workflow_id": self.state.workflow_id,
                    "status": "active",
                    "sort_by": "priority,sequence_number",
                    "sort_order": "ASC",
                    "limit": 1
                }
            )
            
            if goals_res.get("success") and goals_res.get("data"):
                goals = goals_res["data"].get("goals", [])
                if goals:
                    ums_goal = goals[0]
                    # Convert UMS goal to our Goal dataclass
                    goal = Goal(
                        goal_id=ums_goal["goal_id"],
                        parent_goal_id=ums_goal.get("parent_goal_id"),
                        title=ums_goal.get("title", ""),
                        description=ums_goal["description"],
                        status=GoalStatus(ums_goal["status"]),
                        priority=ums_goal["priority"],
                        sequence_number=ums_goal["sequence_number"],
                        created_at=ums_goal["created_at"],
                        updated_at=ums_goal["updated_at"]
                    )
                    return goal
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get next goal: {e}")
            return None

    def active_goals(self) -> List[Goal]:
        """Return active goals - this is now async, so we return empty list."""
        # Note: This method was sync in the original, but UMS calls are async
        # We'll need to update callers to use an async version instead
        self.logger.warning("active_goals() called - use async_active_goals() instead")
        return []

    async def async_active_goals(self) -> List[Goal]:
        """Return a list of all active goals ordered by priority."""
        try:
            goals_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_goals"),
                {
                    "workflow_id": self.state.workflow_id,
                    "status": "active",
                    "limit": 50
                }
            )
            
            if goals_res.get("success") and goals_res.get("data"):
                ums_goals = goals_res["data"].get("goals", [])
                goals = []
                for ums_goal in ums_goals:
                    goal = Goal(
                        goal_id=ums_goal["goal_id"],
                        parent_goal_id=ums_goal.get("parent_goal_id"),
                        title=ums_goal.get("title", ""),
                        description=ums_goal["description"],
                        status=GoalStatus(ums_goal["status"]),
                        priority=ums_goal["priority"],
                        sequence_number=ums_goal["sequence_number"],
                        created_at=ums_goal["created_at"],
                        updated_at=ums_goal["updated_at"]
                    )
                    goals.append(goal)
                return sorted(goals, key=lambda g: (g.priority, g.sequence_number))
            return []
        except Exception as e:
            self.logger.warning(f"Failed to get active goals: {e}")
            return []

    def all_goals(self) -> List[Goal]:
        """Return all goals - this is now async, so we return empty list."""
        self.logger.warning("all_goals() called - use async_all_goals() instead")
        return []

    async def async_all_goals(self) -> List[Goal]:
        """Return all goals for the workflow."""
        try:
            goals_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_goals"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 100
                }
            )
            
            if goals_res.get("success") and goals_res.get("data"):
                ums_goals = goals_res["data"].get("goals", [])
                goals = []
                for ums_goal in ums_goals:
                    goal = Goal(
                        goal_id=ums_goal["goal_id"],
                        parent_goal_id=ums_goal.get("parent_goal_id"),
                        title=ums_goal.get("title", ""),
                        description=ums_goal["description"],
                        status=GoalStatus(ums_goal["status"]),
                        priority=ums_goal["priority"],
                        sequence_number=ums_goal["sequence_number"],
                        created_at=ums_goal["created_at"],
                        updated_at=ums_goal["updated_at"]
                    )
                    goals.append(goal)
                return goals
            return []
        except Exception as e:
            self.logger.warning(f"Failed to get all goals: {e}")
            return []

    # ---- compatibility methods -------------------------------------------

    def __len__(self):
        """Return 0 - actual count requires async call."""
        return 0

    def __repr__(self) -> str:
        return f"<ProceduralAgenda using UMS goal tools for workflow {self.state.workflow_id}>"

    async def _load_from_ums(self):
        """No-op since we don't cache goals locally anymore."""
        pass

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
            load_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("load_cognitive_state"),
                {
                    "workflow_id": self.state.workflow_id,  # Remove the conditional here since we checked above
                    "state_id": None  # Get the latest state
                }
            )
            if load_res.get("success") and load_res.get("state_id"):
                # We have actual state data - update our state from it
                self.state.workflow_id = load_res.get("workflow_id")
                
                # Update state fields that we can restore from cognitive state
                focus_areas = load_res.get("focus_areas", [])
                if focus_areas and len(focus_areas) > 0:
                    # Assume first focus area is current goal
                    self.state.current_leaf_goal_id = focus_areas[0]
                
                # Could also restore other fields if the cognitive state format expands
                # For now, just keep the current agent state but update the restorable parts
                
                self.logger.info(f"Loaded cognitive state from UMS: {load_res.get('title', 'Unknown')}")
            else:
                self.logger.info("No cognitive state data found, using default state")
                
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

        # Create orchestrators / engines
        self.llms = LLMOrchestrator(self.mcp_client, self.state)
        self.tool_exec = ToolExecutor(self.mcp_client, self.state, self.mem_graph)
        self.metacog = MetacognitionEngine(self.mcp_client, self.state, self.mem_graph, self.llms, self.async_queue)
        self.planner = ProceduralAgenda(self.mcp_client, self.state)
        self.graph_reasoner = GraphReasoner(self.mcp_client, self.llms, self.mem_graph)

        # Link components after initialization
        self.metacog.set_planner(self.planner)
        self.metacog.set_agent(self)

    async def _create_bootstrap_memories(self, overall_goal: str) -> None:
        """Create initial memories to bootstrap the context system."""
        if not self.state.workflow_id:
            return
            
        try:
            # Create initial context memory with valid memory_type
            initial_memory = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("store_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "content": f"Starting work on goal: {overall_goal}\n\nThis is the beginning of the agent's reasoning process. The goal will be approached systematically through research, planning, and execution phases.",
                    "memory_type": "observation",  # Changed to valid type
                    "memory_level": "working",
                    "importance": 6.0,
                    "description": "Initial context memory to bootstrap working memory",
                    "suggest_links": False,
                    "generate_embedding": True
                }
            )
            
            if initial_memory.get("success"):
                self.logger.info("Created initial bootstrap memory")
                
            # Create a planning memory instead of using non-existent thought chain tool
            planning_content = f"I need to complete this goal: {overall_goal}\n\nMy approach will be:\n1. Understand the requirements thoroughly\n2. Research relevant information\n3. Plan the execution strategy\n4. Create the required outputs\n\nI should start by analyzing what this goal requires and gathering necessary information."
            
            planning_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("store_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "content": planning_content,
                    "memory_type": "plan",  # Use valid type
                    "memory_level": "working",
                    "importance": 7.0,
                    "description": "Initial planning thought to establish approach",
                    "suggest_links": False,
                    "generate_embedding": True
                }
            )
            
            if planning_res.get("success"):
                self.logger.info("Created initial planning memory")
                
        except Exception as e:
            self.logger.warning(f"Failed to create bootstrap memories: {e}")
            
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
        """Filter tools based on current phase and goal keywords."""
        if not self.tool_schemas:
            return []
        
        goal_lower = goal_desc.lower()
        
        # Phase-specific tool priorities
        phase_tools = {
            Phase.UNDERSTAND: [
                # Research & Information Gathering
                "browse", "search", "download", "autopilot", "collect_documentation",
                # File Analysis
                "read_file", "read_multiple_files", "search_files", "directory_tree", "get_file_info",
                # Document Processing
                "convert_document", "analyze_pdf_structure", "extract_tables", "ocr_image", "detect_content_type",
                # Memory & Context
                "query_memories", "search_semantic_memories", "get_recent_memories_with_links", "hybrid_search_memories",
                # Text Analysis
                "run_ripgrep", "run_awk", "run_jq", "analyze_business_sentiment", "extract_entities",
                # AI Analysis
                "generate_completion", "chat_completion", "analyze_business_text_batch"
            ],
            Phase.PLAN: [
                # Goal Management
                "create_goal", "update_goal_status", "get_goals", "query_goals", "get_goal_details", "get_goal_stack",
                # Memory & Planning
                "store_memory", "create_memory_link", "get_working_memory", "focus_memory", "create_thought_chain",
                # Workflow Management
                "create_workflow", "update_workflow_metadata", "get_workflow_details", "record_action_start",
                # Analysis & Planning
                "estimate_cost", "recommend_model", "get_subgraph"
            ],
            Phase.EXECUTE: [
                # File Creation & Editing
                "write_file", "edit_file", "create_directory", "move_file", "get_unique_filepath",
                # Code Execution
                "execute_python", "repl_python",
                # Document Creation & Processing
                "convert_document", "clean_and_format_text_as_markdown", "batch_format_texts", "chunk_document",
                # Web Actions
                "browse", "click", "type_text", "download", "parallel", "run_macro",
                # Memory & Artifacts
                "store_memory", "record_artifact", "record_action_completion", "record_thought",
                # AI Generation
                "generate_completion", "chat_completion", "single_shot_synthesis", "multi_completion",
                # Text Processing
                "run_sed", "run_awk", "optimize_markdown_formatting", "enhance_ocr_text"
            ],
            Phase.REVIEW: [
                # Artifact Review
                "get_artifacts", "get_artifact_by_id", "record_artifact",
                # Action Review
                "get_recent_actions", "get_action_details", "generate_workflow_report",
                # Quality Check
                "analyze_business_sentiment", "extract_metrics", "flag_risks", "analyze_pdf_structure",
                # Memory & Reflection
                "generate_reflection", "consolidate_memories", "visualize_reasoning_chain", "visualize_memory_network",
                # File Validation
                "read_file", "get_file_info", "directory_tree", "search_files",
                # Final Processing
                "summarize_document", "compute_memory_statistics", "get_contradictions"
            ],
        }
        
        # Keywords to tool mapping
        keyword_tools = {
            "search": ["browse", "search", "search_files", "run_ripgrep", "search_semantic_memories"],
            "web": ["browse", "search", "download", "autopilot", "click", "type_text"],
            "browse": ["browse", "search", "download", "collect_documentation", "autopilot"],
            "write": ["write_file", "edit_file", "create_directory", "record_artifact"],
            "create": ["write_file", "create_directory", "record_artifact", "create_goal"],
            "file": ["read_file", "write_file", "edit_file", "move_file", "search_files"],
            "read": ["read_file", "read_multiple_files", "get_memory_by_id", "query_memories"],
            "analyze": ["analyze_business_sentiment", "extract_metrics", "get_subgraph", "get_contradictions"],
            "document": ["convert_document", "analyze_pdf_structure", "extract_tables", "chunk_document"],
            "pdf": ["convert_document", "analyze_pdf_structure", "extract_tables", "ocr_image"],
            "code": ["execute_python", "repl_python", "write_file", "edit_file"],
            "python": ["execute_python", "repl_python"],
            "text": ["run_ripgrep", "run_awk", "run_sed", "run_jq", "clean_and_format_text_as_markdown"],
            "memory": ["store_memory", "query_memories", "get_working_memory", "focus_memory"],
            "goal": ["create_goal", "update_goal_status", "get_goals", "query_goals"],
            "download": ["download", "download_site_pdfs", "browse"],
            "ai": ["generate_completion", "chat_completion", "single_shot_synthesis", "analyze_business_sentiment"],
            "llm": ["generate_completion", "chat_completion", "multi_completion", "single_shot_synthesis"],
            "report": ["write_file", "record_artifact", "convert_document", "generate_completion"],
            "html": ["write_file", "record_artifact", "browse", "click", "type_text"],
            "interactive": ["browse", "click", "type_text", "parallel", "run_macro"],
            "data": ["extract_tables", "run_jq", "run_awk", "extract_entities"],
            "image": ["ocr_image", "enhance_ocr_text", "convert_document"],
            "research": ["browse", "search", "query_memories", "store_memory", "generate_completion"],
        }
        
        relevant_tool_names = set()
        
        # Add phase-specific tools
        if phase in phase_tools:
            relevant_tool_names.update(phase_tools[phase])
        
        # Add keyword-based tools
        for keyword, tools in keyword_tools.items():
            if keyword in goal_lower:
                relevant_tool_names.update(tools)
        
        # Filter schemas
        filtered_schemas = []
        for schema in self.tool_schemas:
            tool_name = schema.get("function", {}).get("name", "") if "function" in schema else schema.get("name", "")
            if tool_name in relevant_tool_names or not relevant_tool_names:  # Include if relevant or no filter
                filtered_schemas.append(schema)
        
        # Always include essential UMS tools
        essential_ums = ["store_memory", "get_working_memory", "create_goal"]
        for schema in self.tool_schemas:
            tool_name = schema.get("function", {}).get("name", "") if "function" in schema else schema.get("name", "")
            if tool_name in essential_ums and schema not in filtered_schemas:
                filtered_schemas.append(schema)
        
        # Add complementary tools that work well together in parallel
        complementary_pairs = {
            # Research workflows
            "browse": ["store_memory", "record_artifact", "download"],
            "search": ["store_memory", "create_goal", "record_thought"],
            "download": ["convert_document", "store_memory", "record_artifact"],
            
            # File operations
            "read_file": ["store_memory", "analyze_business_sentiment", "extract_entities"],
            "write_file": ["record_artifact", "store_memory", "get_file_info"],
            "edit_file": ["record_artifact", "store_memory"],
            
            # Document processing
            "convert_document": ["store_memory", "record_artifact", "extract_tables"],
            "analyze_pdf_structure": ["extract_tables", "store_memory", "ocr_image"],
            "ocr_image": ["enhance_ocr_text", "store_memory", "extract_entities"],
            
            # Code execution
            "execute_python": ["store_memory", "record_artifact", "write_file"],
            "repl_python": ["store_memory", "record_thought"],
            
            # Goal and memory management
            "create_goal": ["store_memory", "record_action_start"],
            "store_memory": ["create_memory_link", "focus_memory"],
            "query_memories": ["store_memory", "create_memory_link"],
            
            # AI generation
            "generate_completion": ["store_memory", "record_artifact", "write_file"],
            "chat_completion": ["store_memory", "record_thought"],
            "single_shot_synthesis": ["store_memory", "record_artifact"],
            
            # Analysis and extraction
            "extract_tables": ["store_memory", "record_artifact", "run_jq"],
            "extract_entities": ["store_memory", "create_memory_link"],
            "analyze_business_sentiment": ["store_memory", "record_thought"],
            
            # Review and reflection
            "get_artifacts": ["store_memory", "generate_reflection"],
            "get_recent_actions": ["generate_reflection", "store_memory"],
            "generate_reflection": ["store_memory", "consolidate_memories"],
            
            # Web automation
            "click": ["type_text", "browse", "download"],
            "type_text": ["click", "browse"],
            "parallel": ["store_memory", "record_artifact"],
        }
        
        # If we have any tool from a complementary pair, include its partners
        current_tool_names = {
            schema.get("function", {}).get("name", "") if "function" in schema else schema.get("name", "")
            for schema in filtered_schemas
        }
        
        for tool_name, partners in complementary_pairs.items():
            if tool_name in current_tool_names:
                for partner in partners:
                    # Find and add partner tool schema if not already included
                    for schema in self.tool_schemas:
                        schema_name = schema.get("function", {}).get("name", "") if "function" in schema else schema.get("name", "")
                        if schema_name == partner and schema not in filtered_schemas:
                            filtered_schemas.append(schema)
                            break
        
        return filtered_schemas[:limit]

    async def _rank_tools_with_llm_scoring(self, goal_desc: str, phase: Phase, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Alternative implementation using cheap LLM to score tool relevance.
        Not used yet - keeping simple for now.
        """
        if not self.tool_schemas:
            return []
            
        # Use a class-level flag to prevent recursion
        if getattr(self, '_currently_ranking_tools', False):
            return self.tool_schemas[:limit]
        
        try:
            self._currently_ranking_tools = True
            
            # Build tool list for LLM scoring
            tool_names = []
            tool_descriptions = []
            for tool in self.tool_schemas:
                if "function" in tool:
                    name = tool["function"].get("name", "unknown")
                    desc = tool["function"].get("description", "")
                else:
                    name = tool.get("name", "unknown") 
                    desc = tool.get("description", "")
                tool_names.append(name)
                tool_descriptions.append(f"{name}: {desc[:100]}")
            
            # Ask cheap LLM to score tools
            prompt = f"""Rate each tool's relevance for this goal and phase.
Goal: {goal_desc}
Phase: {phase.value}

Tools:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(tool_descriptions))}

Rate each tool 0-100 for relevance. Return JSON array of scores in same order."""
            
            schema = {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 100},
                        "minItems": len(tool_names),
                        "maxItems": len(tool_names)
                    }
                },
                "required": ["scores"]
            }
            
            result = await self.llms.fast_structured_call(prompt, schema)
            scores = result.get("scores", [])
            
            if len(scores) == len(self.tool_schemas):
                # Score and sort tools
                scored_tools = list(zip(scores, self.tool_schemas, strict=False))
                scored_tools.sort(key=lambda x: x[0], reverse=True)
                
                # Return tools above threshold (75.0) or top N
                good_tools = [tool for score, tool in scored_tools if score >= 75.0]
                return good_tools[:limit] if good_tools else [tool for _, tool in scored_tools[:limit]]
            else:
                # Fallback if scoring failed
                return self.tool_schemas[:limit]
                
        except Exception as e:
            self.logger.warning(f"LLM tool scoring failed: {e}")
            return self.tool_schemas[:limit]
        finally:
            self._currently_ranking_tools = False

    async def record_tool_effectiveness(self, goal_desc: str, tool_name: str, success: bool):
        """Track tool effectiveness for future ranking."""
        # Early return if no workflow_id
        if not self.state.workflow_id:
            self.logger.debug("No workflow_id available, skipping tool effectiveness recording")
            return
                    
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
        
    async def _save_cognitive_state(self) -> None:
        """Save current agent state using UMS cognitive state tools."""
        # Early return if no workflow_id
        if not self.state.workflow_id:
            self.logger.debug("No workflow_id available, skipping cognitive state saving")
            return
            
        try:
            # Get current working memory IDs from UMS
            working_memory_ids = []
            try:
                working_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_working_memory"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "focus_goal_id": self.state.current_leaf_goal_id,
                        "limit": 20
                    }
                )
                
                if working_memory_res.get("success") and working_memory_res.get("data"):
                    memories = working_memory_res["data"].get("memories", [])
                    working_memory_ids = [mem.get("memory_id") for mem in memories if mem.get("memory_id")]
            except Exception as e:
                self.logger.debug(f"Could not get working memory IDs: {e}")
                # Continue with empty list
            
            # Get recent action IDs  
            context_action_ids = []
            try:
                actions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                    UMS_SERVER_NAME,
                    self._get_ums_tool_name("get_recent_actions"),
                    {
                        "workflow_id": self.state.workflow_id,
                        "limit": 10
                    }
                )
                
                if actions_res.get("success") and actions_res.get("data"):
                    actions = actions_res["data"].get("actions", [])
                    context_action_ids = [action.get("action_id") for action in actions if action.get("action_id")]
            except Exception as e:
                self.logger.debug(f"Could not get recent action IDs: {e}")
                # Continue with empty list

            # Create title for the cognitive state
            title = f"Agent state at loop {self.state.loop_count} - Phase: {self.state.phase.value}"
            
            # Call UMS save_cognitive_state tool with correct parameters
            save_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("save_cognitive_state"),
                {
                    "workflow_id": self.state.workflow_id,
                    "title": title,
                    "working_memory_ids": working_memory_ids,
                    "focus_area_ids": [self.state.current_leaf_goal_id] if self.state.current_leaf_goal_id else [],
                    "context_action_ids": context_action_ids,
                    "current_goal_thought_ids": []  # Could be populated from thought chains if available
                }
            )
            
            if save_res.get("success"):
                self.logger.debug(f"Saved cognitive state for loop {self.state.loop_count}")
            else:
                self.logger.warning(f"Failed to save cognitive state: {save_res.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            self.logger.warning(f"Failed to save cognitive state: {e}")

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

            # 4. Return parameters for MCPClient to handle
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
            # Create bootstrap memories only after workflow creation
            await self._create_bootstrap_memories(overall_goal)
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
            # DON'T call bootstrap memories again here - remove this line

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
            await self._save_cognitive_state()

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
        
        if not self.state.workflow_id:
            self.logger.error("No workflow_id - cannot gather context")
            raise RuntimeError("Cannot gather context without workflow_id")
        
        try:
            # Build proper parameters for the rich context package
            params = {
                "workflow_id": self.state.workflow_id,
                "context_id": self.state.workflow_id,  # Use workflow_id as context_id if not set separately
                "current_plan_step_description": (
                    self.state.current_plan[0].description 
                    if self.state.current_plan 
                    else "Working on primary goal"
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
                "compression_token_threshold": 4000,
                "compression_target_tokens": 2500
            }
            
            self.logger.debug(f"Calling get_rich_context_package with params: {params}")
            
            rich_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_rich_context_package"),
                params
            )
            
            if not rich_res.get("success"):
                error_msg = rich_res.get("error_message", "Unknown error")
                self.logger.error(f"Rich context package failed: {error_msg}")
                raise RuntimeError(f"Rich context package failed: {error_msg}")
            
            raw_data = rich_res.get("data", {})
            context_package = raw_data.get("context_package", {})
            
            if not context_package:
                self.logger.error(f"Rich context package returned empty context_package. Raw data: {raw_data}")
                raise RuntimeError("Rich context package returned empty context_package")
            
            # Log what we actually got for debugging
            working_memory = context_package.get("current_working_memory", {})
            recent_actions = context_package.get("recent_actions", [])
            core_context = context_package.get("core_context", {})
            
            self.logger.info(
                f"Rich context retrieved: "
                f"working_memory={len(working_memory.get('working_memories', []))} items, "
                f"recent_actions={len(recent_actions)} items, "
                f"core_context={bool(core_context)}"
            )
            
            # Extract contradictions for metacognition
            contradictions = context_package.get("contradictions", {}).get("contradictions_found", [])
            
            # Return the rich context with minimal transformation
            return {
                "rich_context_package": context_package,
                "contradictions": contradictions,
                "has_contradictions": len(contradictions) > 0,
                "context_retrieval_timestamp": context_package.get("retrieval_timestamp_ums_package"),
                "context_sources": {
                    "rich_package": True,
                    "compression_applied": "ums_compression_details" in context_package
                }
            }
                
        except Exception as e:
            self.logger.error(f"Failed to get rich context package: {e}")
            raise RuntimeError(f"Context gathering failed: {e}") from e

    async def _ensure_working_memory_exists(self) -> None:
        """
        Ensure we have some working memory by creating contextual memories if needed.
        """
        if not self.state.workflow_id:
            return
            
        try:
            # Check if we have any working memory
            working_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_working_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 5
                }
            )
            
            has_working_memory = (
                working_memory_res.get("success") and 
                working_memory_res.get("data", {}).get("memories")
            )
            
            if not has_working_memory:
                self.logger.info("No working memory found, creating initial context memories")
                
                # Create initial context memory
                current_goal_desc = await self._get_current_goal_description()
                initial_context = (
                    f"Starting work on goal: {current_goal_desc}\n"
                    f"Current phase: {self.state.phase.value}\n"
                    f"Loop count: {self.state.loop_count}\n"
                    f"This is the beginning of the agent's reasoning process."
                )
                
                await self._store_memory_with_auto_linking(
                    content=initial_context,
                    memory_type="context_initialization",
                    memory_level="working",
                    importance=6.0,
                    description="Initial context memory created to bootstrap working memory"
                )
                
                # Create current state memory
                state_memory = (
                    f"Agent state at loop {self.state.loop_count}:\n"
                    f"- Phase: {self.state.phase.value}\n"
                    f"- Workflow ID: {self.state.workflow_id}\n"
                    f"- Current goal: {self.state.current_leaf_goal_id}\n"
                    f"- Last action: {self.state.last_action_summary or 'None'}\n"
                    f"Agent is actively working toward the goal."
                )
                
                await self._store_memory_with_auto_linking(
                    content=state_memory,
                    memory_type="state_snapshot", 
                    memory_level="working",
                    importance=5.5,
                    description="Current agent state for context"
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to ensure working memory exists: {e}")

    async def _get_working_memory_manually(self) -> str:
        """Get working memory through individual UMS calls."""
        try:
            working_memory_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_working_memory"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 10
                }
            )
            
            if working_memory_res.get("success") and working_memory_res.get("data"):
                memories = working_memory_res["data"].get("memories", [])
                if memories:
                    summaries = []
                    for mem in memories[-5:]:  # Last 5 memories
                        content = mem.get("content", "")[:150]
                        mem_type = mem.get("memory_type", "unknown")
                        importance = mem.get("importance", 0.0)
                        summaries.append(f"[{mem_type}, {importance:.1f}] {content}")
                    return f"{len(memories)} working memories:\n" + "\n".join(summaries)
                else:
                    return "Working memory exists but is empty"
            else:
                return "Could not retrieve working memory"
                
        except Exception as e:
            self.logger.warning(f"Failed to get working memory manually: {e}")
            return f"Working memory retrieval failed: {e}"

    async def _get_recent_actions_manually(self) -> str:
        """Get recent actions through individual UMS calls."""
        try:
            actions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_recent_actions"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 5
                }
            )
            
            if actions_res.get("success") and actions_res.get("data"):
                actions = actions_res["data"].get("actions", [])
                if actions:
                    summaries = []
                    for action in actions:
                        action_type = action.get("action_type", "unknown")
                        status = action.get("status", "unknown")
                        title = action.get("title", "No title")[:50]
                        summaries.append(f"• {action_type}: {title} ({status})")
                    return f"{len(actions)} recent actions:\n" + "\n".join(summaries)
                else:
                    return "No recent actions found"
            else:
                return "Could not retrieve recent actions"
                
        except Exception as e:
            self.logger.warning(f"Failed to get recent actions manually: {e}")
            return f"Recent actions retrieval failed: {e}"

    async def _get_contradictions_manually(self) -> List[Tuple[str, str]]:
        """Get contradictions through individual UMS calls."""
        try:
            contradictions_res = await self.mcp_client._execute_tool_and_parse_for_agent(
                UMS_SERVER_NAME,
                self._get_ums_tool_name("get_contradictions"),
                {
                    "workflow_id": self.state.workflow_id,
                    "limit": 10
                }
            )
            
            if contradictions_res.get("success") and contradictions_res.get("data"):
                contradictions_found = contradictions_res["data"].get("contradictions_found", [])
                pairs = []
                for contradiction in contradictions_found:
                    mem_a = contradiction.get("memory_id_a")
                    mem_b = contradiction.get("memory_id_b")
                    if mem_a and mem_b:
                        pairs.append((mem_a, mem_b))
                return pairs
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"Failed to get contradictions manually: {e}")
            return []

    # -------------------------------- helper: spawn background fast tasks

    async def _maybe_spawn_fast_tasks(self, ctx: Dict[str, Any]) -> None:
        """
        Fire-and-forget cheap-LLM micro-tasks using rich context package data.
        """
        # Extract rich context package
        rich_package = ctx.get("rich_context_package")
        if not rich_package:
            return  # No context to work with
        
        ##########################################################################
        # 1) Handle contradictions if detected
        ##########################################################################
        contradictions = ctx.get('contradictions', [])
        for pair in contradictions[:2]:  # Process up to 2 per turn
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
                }

                async def _on_contradiction(res: Dict[str, str], aid: str = a_id, bid: str = b_id) -> None:
                    # Use simplified auto-linking storage
                    await self._store_memory_with_auto_linking(
                        content=f"{res['summary']}\n\nCLARIFY: {res['question']}",
                        memory_type="contradiction_analysis",
                        memory_level="working",
                        importance=7.0,
                        description="Automated contradiction analysis",
                        link_to_goal=True
                    )
                
                coro = self.llms.fast_structured_call(prompt, schema)
                task_name = f"contradict_{a_id[:4]}_{b_id[:4]}"
                self.async_queue.spawn(AsyncTask(task_name, coro, callback=_on_contradiction))

        ##########################################################################
        # 2) Proactive insight generation from working memory
        ##########################################################################
        working_memory = rich_package.get("current_working_memory", {})
        working_memories = working_memory.get("working_memories", [])
        
        if len(working_memories) >= 3:  # Only if we have enough context
            # Create insight from recent working memories
            memory_contents = [
                mem.get("content", mem.get("content_preview", ""))[:300] 
                for mem in working_memories[-5:]  # Last 5 memories
            ]
            
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
            }

            async def _on_insight(res: Dict[str, str]) -> None:
                await self._store_memory_with_auto_linking(
                    content=res["insight"],
                    memory_type="strategic_insight",
                    memory_level="working",
                    importance=6.5,
                    description="Proactive insight from working memory analysis",
                    link_to_goal=True
                )
            
            coro = self.llms.fast_structured_call(prompt, schema)
            self.async_queue.spawn(AsyncTask("working_memory_insight", coro, callback=_on_insight))

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
1. Using tools to gather information (browse, search, read_file, query_memories)
2. Creating tangible outputs (write_file, record_artifact, execute_python)
3. Storing key findings in memory (store_memory, record_thought)
4. Breaking down complex goals (create_goal)

EFFICIENCY: Use TOOL_MULTIPLE when possible! Examples:
- Research: [browse, store_memory] or [search, download, store_memory]
- File analysis: [read_file, analyze_business_sentiment, store_memory]
- Document processing: [convert_document, extract_tables, record_artifact]
- Code work: [execute_python, write_file, record_artifact]
- Web automation: [click, type_text, download]
- Data extraction: [extract_tables, run_jq, store_memory]

Response Format - EXACTLY this JSON structure:

For SINGLE tool:
{{
    "decision_type": "TOOL_SINGLE",
    "tool_name": "tool_name_here",
    "tool_args": {{...}}
}}

For MULTIPLE tools (PREFERRED when possible):
{{
    "decision_type": "TOOL_MULTIPLE", 
    "tool_calls": [
        {{"tool_name": "web_search", "tool_args": {{"query": "..."}}, "tool_id": "search_1"}},
        {{"tool_name": "store_memory", "tool_args": {{"content": "..."}}, "tool_id": "store_1"}}
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
            
            # Add phase-specific guidance
            if self.state.phase == Phase.UNDERSTAND:
                user_msg += "Focus: Research and gather information about your goal. EFFICIENT: Use [browse, store_memory] or [search, download, store_memory] together."
            elif self.state.phase == Phase.PLAN:
                user_msg += "Focus: Break down your goal into concrete, actionable sub-tasks. EFFICIENT: Use [create_goal, store_memory] or [create_goal] together."
            elif self.state.phase == Phase.EXECUTE:
                user_msg += "Focus: Create the required outputs and deliverables. EFFICIENT: Use [write_file, record_artifact] or [execute_python, store_memory] together."
            elif self.state.phase == Phase.REVIEW:
                user_msg += "Focus: Verify all outputs exist and meet requirements. EFFICIENT: Use [get_artifacts, store_memory] or [analyze_business_sentiment, generate_reflection] together."
            
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

    async def _enact(self, decision: Any) -> bool:
        """
        Execute the SMART-model output with guaranteed structure.

        Returns
        -------
        progress_made : bool
            Heuristic flag used by metacognition.
        """
        self.logger.debug("[AML] decision raw: %s", decision)

        # With structured output, we guarantee this is a dict with decision_type
        if not isinstance(decision, dict) or "decision_type" not in decision:
            self.logger.error(f"[AML] Invalid decision format: {decision}")
            return False

        decision_type = decision["decision_type"]
        
        if decision_type == "TOOL_SINGLE":
            # Execute the specified tool
            tool_name = decision["tool_name"]
            tool_args = decision["tool_args"]
            
            self.logger.info("[AML] → executing tool %s with args %s", tool_name, tool_args)
            result = await self.tool_exec.run(tool_name, tool_args)
            success = result.get("success", False)
            
            # Record tool effectiveness for learning
            if self.state.current_leaf_goal_id:
                goal_desc = await self._get_current_goal_description()
                await self.record_tool_effectiveness(goal_desc, tool_name, success)
            
            if success:
                self.state.last_action_summary = f"Successfully executed {tool_name}"
                return True
            else:
                error_msg = result.get("error_message", "Unknown error")
                self.state.last_action_summary = f"Failed to execute {tool_name}: {error_msg}"
                return False
                
        elif decision_type == "TOOL_MULTIPLE":
            # Execute multiple tools in parallel - much more efficient!
            tool_calls = decision["tool_calls"]
            
            self.logger.info("[AML] → executing %d tools in parallel", len(tool_calls))
            
            # Use the existing run_parallel method from ToolExecutor
            parallel_result = await self.tool_exec.run_parallel(tool_calls)
            
            success = parallel_result.get("success", False)
            results = parallel_result.get("results", [])
            timing_info = parallel_result.get("timing", {})
            
            # Count successes and failures
            successful_count = sum(1 for r in results if r.get("success", False))
            total_count = len(results)
            
            # Record effectiveness for each tool
            if self.state.current_leaf_goal_id:
                goal_desc = await self._get_current_goal_description()
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call["tool_name"]
                    if i < len(results):
                        tool_success = results[i].get("success", False)
                        await self.record_tool_effectiveness(goal_desc, tool_name, tool_success)
            
            # Log timing information
            if timing_info:
                timing_summary = ", ".join([f"{tid}: {time:.2f}s" for tid, time in timing_info.items()])
                self.logger.debug(f"[AML] Parallel execution timing: {timing_summary}")
            
            if successful_count > 0:
                self.state.last_action_summary = f"Parallel execution: {successful_count}/{total_count} tools succeeded"
                return True
            else:
                self.state.last_action_summary = f"Parallel execution failed: 0/{total_count} tools succeeded"
                return False
                
        elif decision_type == "THOUGHT_PROCESS":
            # Store reasoning as memory
            thought = decision["content"]
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
            # This should never happen with structured output
            self.logger.error(f"[AML] Unexpected decision type: {decision_type}")
            return False

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
            asyncio.create_task(self._save_cognitive_state())
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
            await self._save_cognitive_state()
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
        Store a memory with automatic linking via UMS store_memory tool.
        """
        try:
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
                    "suggest_links": True,
                    "max_suggested_links": 5,
                    "link_suggestion_threshold": 0.7,
                    "action_id": None,  # Could link to current action if available
                    "generate_embedding": True
                }
            )
            
            if store_res.get("success") and store_res.get("data"):
                memory_id = store_res["data"].get("memory_id")
                
                # If requested, associate memory with current goal using proper UMS goal tools
                if link_to_goal and self.state.current_leaf_goal_id and memory_id:
                    try:
                        # Use a goal-specific tool or metadata to associate memory with goal
                        # For now, we'll store this relationship in memory metadata
                        current_meta = await self.mem_graph._get_metadata(memory_id)
                        current_meta["associated_goal_id"] = self.state.current_leaf_goal_id
                        
                        await self.mcp_client._execute_tool_and_parse_for_agent(
                            UMS_SERVER_NAME,
                            self._get_ums_tool_name("update_memory_metadata"),
                            {
                                "workflow_id": self.state.workflow_id,
                                "memory_id": memory_id,
                                "metadata": current_meta
                            }
                        )
                    except Exception as e:
                        self.logger.debug(f"Goal association failed (non-critical): {e}")
                        
                return memory_id
            else:
                self.logger.warning(f"Memory storage failed: {store_res.get('error_message', 'Unknown error')}")
                return None
                
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

