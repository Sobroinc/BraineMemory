"""
Fusion algorithms for combining multiple retrieval results.

Implements:
- Reciprocal Rank Fusion (RRF)
- Linear Combination
- Max Score
- Borda Count
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ScoredItem:
    """Item with score for fusion."""

    id: str
    content: str
    score: float
    source: str  # Which retriever produced this
    metadata: dict[str, Any] | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ScoredItem):
            return self.id == other.id
        return False


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredItem]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[ScoredItem]:
    """
    Reciprocal Rank Fusion (RRF) - combines multiple ranked lists.

    RRF score = Σ (weight_i / (k + rank_i))

    This is the most robust fusion method for combining different retrieval strategies.
    It handles score incompatibility well (FTS scores vs cosine similarity).

    Args:
        result_lists: List of ranked result lists from different retrievers
        k: RRF constant (default 60, from original paper)
        weights: Optional weights for each result list

    Returns:
        Fused and re-ranked results
    """
    if not result_lists:
        return []

    # Default equal weights
    if weights is None:
        weights = [1.0] * len(result_lists)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Calculate RRF scores
    rrf_scores: dict[str, float] = defaultdict(float)
    items_by_id: dict[str, ScoredItem] = {}
    sources_by_id: dict[str, list[str]] = defaultdict(list)

    for list_idx, result_list in enumerate(result_lists):
        weight = weights[list_idx]

        for rank, item in enumerate(result_list, start=1):
            # RRF formula
            rrf_score = weight / (k + rank)
            rrf_scores[item.id] += rrf_score

            # Keep the item with highest original score
            if item.id not in items_by_id or item.score > items_by_id[item.id].score:
                items_by_id[item.id] = item

            sources_by_id[item.id].append(item.source)

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build result with fused scores
    results = []
    for item_id in sorted_ids:
        item = items_by_id[item_id]
        fused_item = ScoredItem(
            id=item.id,
            content=item.content,
            score=rrf_scores[item_id],
            source="+".join(sorted(set(sources_by_id[item_id]))),
            metadata={
                **(item.metadata or {}),
                "rrf_score": rrf_scores[item_id],
                "sources": sources_by_id[item_id],
                "original_score": item.score,
            },
        )
        results.append(fused_item)

    logger.debug(f"RRF fusion: {sum(len(r) for r in result_lists)} items → {len(results)} unique")
    return results


def linear_combination(
    result_lists: list[list[ScoredItem]],
    weights: list[float] | None = None,
    normalize_scores: bool = True,
) -> list[ScoredItem]:
    """
    Linear combination of scores from multiple retrievers.

    final_score = Σ (weight_i * normalized_score_i)

    Best when scores are comparable (e.g., all cosine similarities).

    Args:
        result_lists: List of ranked result lists
        weights: Weights for each list (default: equal)
        normalize_scores: Normalize scores to [0, 1] range

    Returns:
        Fused results sorted by combined score
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Collect scores
    combined_scores: dict[str, float] = defaultdict(float)
    items_by_id: dict[str, ScoredItem] = {}

    for list_idx, result_list in enumerate(result_lists):
        if not result_list:
            continue

        weight = weights[list_idx]

        # Normalize scores if requested
        if normalize_scores and result_list:
            max_score = max(item.score for item in result_list)
            min_score = min(item.score for item in result_list)
            score_range = max_score - min_score if max_score != min_score else 1.0
        else:
            min_score = 0.0
            score_range = 1.0

        for item in result_list:
            if normalize_scores:
                normalized_score = (item.score - min_score) / score_range
            else:
                normalized_score = item.score

            combined_scores[item.id] += weight * normalized_score

            if item.id not in items_by_id:
                items_by_id[item.id] = item

    # Sort by combined score
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    results = []
    for item_id in sorted_ids:
        item = items_by_id[item_id]
        results.append(ScoredItem(
            id=item.id,
            content=item.content,
            score=combined_scores[item_id],
            source=item.source,
            metadata={
                **(item.metadata or {}),
                "combined_score": combined_scores[item_id],
            },
        ))

    return results


def max_score_fusion(
    result_lists: list[list[ScoredItem]],
) -> list[ScoredItem]:
    """
    Max score fusion - take the highest score for each item.

    Useful when you want any high-confidence match to surface.
    """
    if not result_lists:
        return []

    max_scores: dict[str, float] = {}
    items_by_id: dict[str, ScoredItem] = {}

    for result_list in result_lists:
        for item in result_list:
            if item.id not in max_scores or item.score > max_scores[item.id]:
                max_scores[item.id] = item.score
                items_by_id[item.id] = item

    sorted_ids = sorted(max_scores.keys(), key=lambda x: max_scores[x], reverse=True)

    return [items_by_id[item_id] for item_id in sorted_ids]


def borda_count_fusion(
    result_lists: list[list[ScoredItem]],
    weights: list[float] | None = None,
) -> list[ScoredItem]:
    """
    Borda count fusion - points based on rank position.

    Each item gets (n - rank + 1) points where n is list length.
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    borda_scores: dict[str, float] = defaultdict(float)
    items_by_id: dict[str, ScoredItem] = {}

    for list_idx, result_list in enumerate(result_lists):
        weight = weights[list_idx]
        n = len(result_list)

        for rank, item in enumerate(result_list):
            points = (n - rank) * weight
            borda_scores[item.id] += points

            if item.id not in items_by_id:
                items_by_id[item.id] = item

    sorted_ids = sorted(borda_scores.keys(), key=lambda x: borda_scores[x], reverse=True)

    results = []
    for item_id in sorted_ids:
        item = items_by_id[item_id]
        results.append(ScoredItem(
            id=item.id,
            content=item.content,
            score=borda_scores[item_id],
            source=item.source,
            metadata={
                **(item.metadata or {}),
                "borda_score": borda_scores[item_id],
            },
        ))

    return results


def filter_by_threshold(
    items: list[ScoredItem],
    min_score: float = 0.0,
) -> list[ScoredItem]:
    """Filter items below minimum score threshold."""
    return [item for item in items if item.score >= min_score]


def deduplicate_by_content(
    items: list[ScoredItem],
    similarity_threshold: float = 0.95,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> list[ScoredItem]:
    """
    Remove near-duplicate content.

    By default uses simple character overlap ratio.
    Can provide custom similarity function.
    """
    if not items:
        return []

    def default_similarity(a: str, b: str) -> float:
        """Simple character overlap ratio."""
        if not a or not b:
            return 0.0
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return intersection / union if union > 0 else 0.0

    sim_fn = similarity_fn or default_similarity

    deduplicated = []
    for item in items:
        is_duplicate = False
        for existing in deduplicated:
            if sim_fn(item.content, existing.content) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(item)

    logger.debug(f"Deduplication: {len(items)} → {len(deduplicated)} items")
    return deduplicated


def rerank_by_query_relevance(
    items: list[ScoredItem],
    query: str,
    boost_keywords: bool = True,
) -> list[ScoredItem]:
    """
    Simple reranking based on query keyword presence.

    For production, consider using a cross-encoder model.
    """
    if not items or not query:
        return items

    query_words = set(query.lower().split())

    def keyword_boost(item: ScoredItem) -> float:
        content_words = set(item.content.lower().split())
        overlap = len(query_words & content_words)
        boost = overlap / len(query_words) if query_words else 0
        return item.score * (1 + boost * 0.5) if boost_keywords else item.score

    # Sort by boosted score
    reranked = sorted(items, key=keyword_boost, reverse=True)

    # Update scores
    for i, item in enumerate(reranked):
        item.metadata = item.metadata or {}
        item.metadata["rerank_position"] = i + 1

    return reranked
