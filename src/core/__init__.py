"""Core modules for BraineMemory."""

from src.core.router import Router, Intent, RetrievalMode, Pipeline, router
from src.core.fusion import (
    ScoredItem,
    reciprocal_rank_fusion,
    linear_combination,
    max_score_fusion,
    borda_count_fusion,
)
from src.core.user_memory import UserMemoryManager, MemoryCategory, user_memory
from src.core.recall import (
    RecallEngine,
    RecallResult,
    recall_engine,
    hybrid_recall,
    smart_recall,
    research_recall,
    claim_recall_with_conflicts,
)
from src.core.community import (
    CommunityDetector,
    Community,
    community_detector,
    build_community_index,
)
from src.core.conflict import (
    ConflictDetector,
    ConflictType,
    ConflictSeverity,
    DetectedConflict,
    conflict_detector,
    detect_conflicts_for_new_claim,
    scan_all_conflicts,
)

__all__ = [
    # Router
    "Router",
    "Intent",
    "RetrievalMode",
    "Pipeline",
    "router",
    # Fusion
    "ScoredItem",
    "reciprocal_rank_fusion",
    "linear_combination",
    "max_score_fusion",
    "borda_count_fusion",
    # User Memory
    "UserMemoryManager",
    "MemoryCategory",
    "user_memory",
    # Recall Engine
    "RecallEngine",
    "RecallResult",
    "recall_engine",
    "hybrid_recall",
    "smart_recall",
    "research_recall",
    "claim_recall_with_conflicts",
    # Community Detection
    "CommunityDetector",
    "Community",
    "community_detector",
    "build_community_index",
    # Conflict Detection
    "ConflictDetector",
    "ConflictType",
    "ConflictSeverity",
    "DetectedConflict",
    "conflict_detector",
    "detect_conflicts_for_new_claim",
    "scan_all_conflicts",
]
