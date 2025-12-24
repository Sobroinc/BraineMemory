"""
Router for BraineMemory - Intent Detection & Pipeline Selection.

Determines the optimal retrieval pipeline based on query characteristics.
Inspired by GAM (General Agentic Memory) and GraphRAG routing patterns.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Query intent categories."""

    FACTUAL = "factual"           # "где написано", "цитата", exact quote lookup
    ENTITY_CENTRIC = "entity"     # "кто", "что связано с X", entity-focused
    CORPUS_WIDE = "corpus"        # "по всему корпусу", "обзор", broad summary
    RESEARCH = "research"         # "исследуй", "противоречия", deep analysis
    TIMELINE = "timeline"         # "когда", "хронология", temporal queries
    COMPARE = "compare"           # "сравни", "различия", diff between items
    EVIDENCE = "evidence"         # "докажи", "откуда известно", provenance chain


class RetrievalMode(Enum):
    """Retrieval strategies available."""

    FTS_ONLY = "fts"              # Full-text search only (BM25)
    VECTOR_ONLY = "vector"        # Vector similarity only (HNSW)
    HYBRID = "hybrid"             # FTS + Vector with fusion
    LOCAL_GRAPH = "local_graph"   # Entity → neighbors expansion
    GLOBAL_GRAPH = "global_graph" # Community summaries (GraphRAG global)
    DRIFT = "drift"               # Local + community context (GraphRAG DRIFT)


@dataclass
class Pipeline:
    """Execution pipeline configuration."""

    retrieval_modes: list[RetrievalMode]
    use_user_context: bool = True
    use_graph: bool = False
    max_iterations: int = 1       # >1 enables GAM-style iterative refinement
    token_budget: int = 4000
    rerank: bool = True           # Apply reranking after fusion
    include_evidence: bool = True

    # Weights for hybrid fusion
    fts_weight: float = 0.4
    vector_weight: float = 0.6
    graph_weight: float = 0.3

    # Thresholds
    min_relevance: float = 0.005  # Filter results below this (RRF scores are small)
    confidence_threshold: float = 0.02  # For research mode: continue if below


@dataclass
class IntentPattern:
    """Pattern for intent detection."""

    intent: Intent
    keywords_ru: list[str] = field(default_factory=list)
    keywords_en: list[str] = field(default_factory=list)
    keywords_fr: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = checked first


class Router:
    """
    Query Router - determines optimal retrieval pipeline.

    Responsibilities:
    1. Detect query intent (factual, entity-centric, research, etc.)
    2. Select appropriate retrieval modes
    3. Configure pipeline parameters
    4. Handle mode overrides
    """

    # Intent patterns for detection
    INTENT_PATTERNS = [
        IntentPattern(
            intent=Intent.RESEARCH,
            keywords_ru=["исследуй", "проанализируй", "противоречия", "все аспекты", "глубокий анализ"],
            keywords_en=["research", "analyze", "contradictions", "deep analysis", "investigate"],
            keywords_fr=["recherche", "analyse", "contradictions", "approfondi"],
            priority=10,  # Highest priority - explicit research request
        ),
        IntentPattern(
            intent=Intent.COMPARE,
            keywords_ru=["сравни", "различия", "отличия", "vs", "против"],
            keywords_en=["compare", "difference", "vs", "versus", "contrast"],
            keywords_fr=["compare", "différence", "contre"],
            regex_patterns=[r"(.+)\s+(?:vs|против|или)\s+(.+)"],
            priority=9,
        ),
        IntentPattern(
            intent=Intent.TIMELINE,
            keywords_ru=["когда", "хронология", "история", "даты", "временная"],
            keywords_en=["when", "timeline", "history", "dates", "chronology"],
            keywords_fr=["quand", "chronologie", "histoire", "dates"],
            priority=8,
        ),
        IntentPattern(
            intent=Intent.EVIDENCE,
            keywords_ru=["докажи", "откуда известно", "источник", "цепочка доказательств"],
            keywords_en=["prove", "evidence", "source", "how do you know"],
            keywords_fr=["prouve", "source", "comment sais-tu"],
            priority=7,
        ),
        IntentPattern(
            intent=Intent.FACTUAL,
            keywords_ru=["где написано", "цитата", "страница", "точная формулировка", "дословно"],
            keywords_en=["where is it written", "quote", "exact", "page", "verbatim"],
            keywords_fr=["où est écrit", "citation", "exacte", "page"],
            priority=6,
        ),
        IntentPattern(
            intent=Intent.ENTITY_CENTRIC,
            keywords_ru=["кто такой", "что такое", "связано с", "относится к", "о ком", "о чём"],
            keywords_en=["who is", "what is", "related to", "about whom", "about what"],
            keywords_fr=["qui est", "qu'est-ce que", "lié à", "à propos de"],
            priority=5,
        ),
        IntentPattern(
            intent=Intent.CORPUS_WIDE,
            keywords_ru=["все", "обзор", "резюме", "по всему", "общая картина", "сводка"],
            keywords_en=["all", "overview", "summary", "across all", "big picture"],
            keywords_fr=["tout", "aperçu", "résumé", "vue d'ensemble"],
            priority=4,
        ),
    ]

    # Pipeline configurations per intent
    ROUTING_TABLE: dict[Intent, Pipeline] = {
        Intent.FACTUAL: Pipeline(
            retrieval_modes=[RetrievalMode.FTS_ONLY, RetrievalMode.VECTOR_ONLY],
            use_graph=False,
            max_iterations=1,
            fts_weight=0.6,  # Favor exact matches
            vector_weight=0.4,
        ),
        Intent.ENTITY_CENTRIC: Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID, RetrievalMode.LOCAL_GRAPH],
            use_graph=True,
            max_iterations=1,
            graph_weight=0.4,
        ),
        Intent.CORPUS_WIDE: Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID, RetrievalMode.GLOBAL_GRAPH],
            use_graph=True,
            max_iterations=1,
            token_budget=8000,  # Larger budget for summaries
        ),
        Intent.RESEARCH: Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID, RetrievalMode.LOCAL_GRAPH, RetrievalMode.GLOBAL_GRAPH],
            use_graph=True,
            max_iterations=3,  # GAM-style iterative refinement
            token_budget=12000,
            confidence_threshold=0.7,
        ),
        Intent.TIMELINE: Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID],
            use_graph=True,
            max_iterations=1,
            include_evidence=True,
        ),
        Intent.COMPARE: Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID, RetrievalMode.LOCAL_GRAPH],
            use_graph=True,
            max_iterations=1,
        ),
        Intent.EVIDENCE: Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID, RetrievalMode.LOCAL_GRAPH],
            use_graph=True,
            max_iterations=1,
            include_evidence=True,
        ),
    }

    # Mode overrides (explicit user request)
    MODE_MAP: dict[str, Pipeline] = {
        "auto": None,  # Use intent detection
        "fts": Pipeline(retrieval_modes=[RetrievalMode.FTS_ONLY], use_graph=False),
        "vector": Pipeline(retrieval_modes=[RetrievalMode.VECTOR_ONLY], use_graph=False),
        "hybrid": Pipeline(retrieval_modes=[RetrievalMode.HYBRID], use_graph=False),
        "local": Pipeline(retrieval_modes=[RetrievalMode.LOCAL_GRAPH], use_graph=True),
        "global": Pipeline(retrieval_modes=[RetrievalMode.GLOBAL_GRAPH], use_graph=True),
        "drift": Pipeline(retrieval_modes=[RetrievalMode.DRIFT], use_graph=True),
        "research": Pipeline(
            retrieval_modes=[RetrievalMode.HYBRID, RetrievalMode.LOCAL_GRAPH],
            use_graph=True,
            max_iterations=3,
        ),
    }

    def __init__(self) -> None:
        # Sort patterns by priority (descending)
        self._patterns = sorted(
            self.INTENT_PATTERNS,
            key=lambda p: p.priority,
            reverse=True,
        )

    def detect_intent(self, query: str) -> Intent:
        """
        Detect query intent using keyword matching and patterns.

        This is a fast, rule-based approach. For production,
        consider adding LLM-based intent classification as fallback.
        """
        query_lower = query.lower()

        for pattern in self._patterns:
            # Check keywords (all languages)
            all_keywords = (
                pattern.keywords_ru +
                pattern.keywords_en +
                pattern.keywords_fr
            )

            for keyword in all_keywords:
                if keyword.lower() in query_lower:
                    logger.debug(f"Intent detected: {pattern.intent.value} (keyword: {keyword})")
                    return pattern.intent

            # Check regex patterns
            for regex in pattern.regex_patterns:
                if re.search(regex, query, re.IGNORECASE):
                    logger.debug(f"Intent detected: {pattern.intent.value} (regex: {regex})")
                    return pattern.intent

        # Default to FACTUAL (most common case)
        logger.debug("Intent defaulted to FACTUAL")
        return Intent.FACTUAL

    def get_pipeline(
        self,
        query: str,
        mode: str = "auto",
        **overrides: Any,
    ) -> Pipeline:
        """
        Get execution pipeline for query.

        Args:
            query: The search query
            mode: Explicit mode override ("auto", "fts", "vector", etc.)
            **overrides: Override specific pipeline parameters

        Returns:
            Pipeline configuration
        """
        # Check for explicit mode
        if mode != "auto" and mode in self.MODE_MAP:
            base_pipeline = self.MODE_MAP[mode]
            if base_pipeline:
                pipeline = self._apply_overrides(base_pipeline, overrides)
                logger.info(f"Using explicit mode: {mode}")
                return pipeline

        # Auto-detect intent
        intent = self.detect_intent(query)
        base_pipeline = self.ROUTING_TABLE.get(intent, self.ROUTING_TABLE[Intent.FACTUAL])

        # Apply overrides
        pipeline = self._apply_overrides(base_pipeline, overrides)

        logger.info(f"Router: intent={intent.value}, modes={[m.value for m in pipeline.retrieval_modes]}")
        return pipeline

    def _apply_overrides(self, pipeline: Pipeline, overrides: dict[str, Any]) -> Pipeline:
        """Apply parameter overrides to pipeline."""
        if not overrides:
            return pipeline

        # Create a copy with overrides
        return Pipeline(
            retrieval_modes=overrides.get("retrieval_modes", pipeline.retrieval_modes),
            use_user_context=overrides.get("use_user_context", pipeline.use_user_context),
            use_graph=overrides.get("use_graph", pipeline.use_graph),
            max_iterations=overrides.get("max_iterations", pipeline.max_iterations),
            token_budget=overrides.get("token_budget", pipeline.token_budget),
            rerank=overrides.get("rerank", pipeline.rerank),
            include_evidence=overrides.get("include_evidence", pipeline.include_evidence),
            fts_weight=overrides.get("fts_weight", pipeline.fts_weight),
            vector_weight=overrides.get("vector_weight", pipeline.vector_weight),
            graph_weight=overrides.get("graph_weight", pipeline.graph_weight),
            min_relevance=overrides.get("min_relevance", pipeline.min_relevance),
            confidence_threshold=overrides.get("confidence_threshold", pipeline.confidence_threshold),
        )

    def explain_routing(self, query: str, mode: str = "auto") -> dict[str, Any]:
        """
        Explain why a particular pipeline was chosen.
        Useful for debugging and transparency.
        """
        intent = self.detect_intent(query)
        pipeline = self.get_pipeline(query, mode)

        # Find matching pattern
        matched_pattern = None
        matched_keyword = None
        query_lower = query.lower()

        for pattern in self._patterns:
            if pattern.intent == intent:
                for kw in pattern.keywords_ru + pattern.keywords_en + pattern.keywords_fr:
                    if kw.lower() in query_lower:
                        matched_pattern = pattern
                        matched_keyword = kw
                        break
                break

        return {
            "query": query,
            "detected_intent": intent.value,
            "matched_keyword": matched_keyword,
            "explicit_mode": mode if mode != "auto" else None,
            "pipeline": {
                "retrieval_modes": [m.value for m in pipeline.retrieval_modes],
                "use_graph": pipeline.use_graph,
                "max_iterations": pipeline.max_iterations,
                "token_budget": pipeline.token_budget,
                "weights": {
                    "fts": pipeline.fts_weight,
                    "vector": pipeline.vector_weight,
                    "graph": pipeline.graph_weight,
                },
            },
        }


# Global router instance
router = Router()
