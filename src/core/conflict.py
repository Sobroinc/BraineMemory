"""
Conflict Detection for Claims.

Detects contradictions between claims using:
- Semantic similarity to find potentially conflicting claims
- LLM analysis to verify actual contradictions
- Severity classification
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from src.adapters.embeddings import embeddings
from src.config import settings
from src.db.client import db

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """Types of conflicts between claims."""

    CONTRADICTION = "contradiction"  # Direct logical contradiction
    INCONSISTENCY = "inconsistency"  # Factual inconsistency
    TEMPORAL = "temporal"  # Time-based conflict (was X, now Y)
    NUMERIC = "numeric"  # Conflicting numbers/amounts
    ATTRIBUTION = "attribution"  # Different sources say different things


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""

    CRITICAL = "critical"  # Must be resolved immediately
    HIGH = "high"  # Important to resolve
    MEDIUM = "medium"  # Should be reviewed
    LOW = "low"  # Minor discrepancy


@dataclass
class DetectedConflict:
    """A detected conflict between claims."""

    claim1_id: str
    claim1_statement: str
    claim2_id: str
    claim2_statement: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    confidence: float  # 0-1 confidence in conflict detection


class ConflictDetector:
    """
    Detects conflicts between claims using semantic similarity and LLM analysis.

    Process:
    1. Find semantically similar claims (potential conflicts)
    2. Use LLM to analyze if they actually contradict
    3. Classify conflict type and severity
    4. Store in database
    """

    def __init__(self):
        self._http_client: httpx.AsyncClient | None = None
        self.similarity_threshold = 0.6  # Min similarity to consider as potential conflict

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._http_client

    async def detect_conflicts_for_claim(
        self,
        claim_id: str,
        claim_statement: str,
        claim_embedding: list[float] | None = None,
    ) -> list[DetectedConflict]:
        """
        Detect conflicts for a specific claim against existing claims.

        Args:
            claim_id: ID of the claim to check
            claim_statement: Statement text
            claim_embedding: Pre-computed embedding (optional)

        Returns:
            List of detected conflicts
        """
        # Get embedding if not provided
        if claim_embedding is None:
            claim_embedding = await embeddings.embed(claim_statement)

        # Find similar claims via vector search
        similar_claims = await db.vector_search(
            table="claim",
            vector_field="embedding",
            query_vector=claim_embedding,
            limit=10,
            where="status = 'active'",
        )

        # Filter out self and low similarity
        candidates = []
        for row in similar_claims:
            row_id = str(row.get("id", ""))
            similarity = float(row.get("relevance", 0))

            # Skip self
            if row_id == claim_id:
                continue

            # Skip low similarity
            if similarity < self.similarity_threshold:
                continue

            candidates.append({
                "id": row_id,
                "statement": row.get("statement", ""),
                "similarity": similarity,
            })

        if not candidates:
            logger.debug(f"No similar claims found for {claim_id}")
            return []

        logger.info(f"Found {len(candidates)} similar claims to analyze for conflicts")

        # Analyze each candidate pair for contradiction
        conflicts = []
        for candidate in candidates:
            conflict = await self._analyze_conflict(
                claim1_id=claim_id,
                claim1_statement=claim_statement,
                claim2_id=candidate["id"],
                claim2_statement=candidate["statement"],
            )

            if conflict:
                conflicts.append(conflict)

        return conflicts

    async def detect_all_conflicts(
        self,
        batch_size: int = 50,
    ) -> list[DetectedConflict]:
        """
        Scan all claims for conflicts.

        Args:
            batch_size: Number of claims to process at once

        Returns:
            List of all detected conflicts
        """
        # Get all active claims (note: embedding is a large array, fetched separately)
        # Note: ORDER BY may fail in some SurrealDB versions, so we skip it
        claims_result = await db.query(
            "SELECT id, statement FROM claim WHERE status = 'active'"
        )

        claims = claims_result[0].get("result", []) if claims_result else []
        logger.info(f"Scanning {len(claims)} claims for conflicts")

        all_conflicts = []
        checked_pairs: set[tuple[str, str]] = set()

        for claim in claims:
            claim_id = str(claim.get("id", ""))
            statement = claim.get("statement", "")

            if not statement:
                continue

            # Detect conflicts for this claim (embedding will be computed inside)
            conflicts = await self.detect_conflicts_for_claim(
                claim_id=claim_id,
                claim_statement=statement,
                claim_embedding=None,  # Will be computed
            )

            # Deduplicate (don't report A-B and B-A)
            for conflict in conflicts:
                pair = tuple(sorted([conflict.claim1_id, conflict.claim2_id]))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    all_conflicts.append(conflict)

        logger.info(f"Detected {len(all_conflicts)} conflicts")
        return all_conflicts

    async def _analyze_conflict(
        self,
        claim1_id: str,
        claim1_statement: str,
        claim2_id: str,
        claim2_statement: str,
    ) -> DetectedConflict | None:
        """
        Use LLM to analyze if two claims contradict each other.

        Returns:
            DetectedConflict if contradiction found, None otherwise
        """
        client = await self._ensure_client()

        prompt = f"""Analyze these two claims and determine if they contradict each other.

CLAIM 1:
{claim1_statement}

CLAIM 2:
{claim2_statement}

Analyze carefully:
1. Do these claims directly contradict each other?
2. Are they merely different but compatible perspectives?
3. Could both be true in different contexts/times?

If there IS a conflict, respond with JSON:
{{
    "is_conflict": true,
    "conflict_type": "contradiction|inconsistency|temporal|numeric|attribution",
    "severity": "critical|high|medium|low",
    "description": "Brief explanation of the conflict",
    "confidence": 0.0-1.0
}}

If there is NO conflict, respond with:
{{
    "is_conflict": false,
    "reason": "Why these claims don't conflict"
}}

Respond with valid JSON only."""

        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert at detecting logical contradictions and factual inconsistencies. Be precise and conservative - only flag clear conflicts.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]

            # Parse JSON
            import json
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            result = json.loads(content)

            if not result.get("is_conflict"):
                return None

            return DetectedConflict(
                claim1_id=claim1_id,
                claim1_statement=claim1_statement,
                claim2_id=claim2_id,
                claim2_statement=claim2_statement,
                conflict_type=ConflictType(result.get("conflict_type", "inconsistency")),
                severity=ConflictSeverity(result.get("severity", "medium")),
                description=result.get("description", ""),
                confidence=float(result.get("confidence", 0.5)),
            )

        except Exception as e:
            logger.error(f"Conflict analysis failed: {e}")
            return None

    async def store_conflict(self, conflict: DetectedConflict) -> str | None:
        """
        Store a detected conflict in the database.

        Args:
            conflict: The conflict to store

        Returns:
            Conflict ID if stored, None if duplicate
        """
        # Check if conflict already exists
        existing = await db.query(f"""
            SELECT id FROM conflict_side
            WHERE claim IN [{conflict.claim1_id}, {conflict.claim2_id}]
        """)

        existing_conflicts = existing[0].get("result", []) if existing else []
        if len(existing_conflicts) >= 2:
            # Both claims already in a conflict
            logger.debug(f"Conflict already exists for {conflict.claim1_id} and {conflict.claim2_id}")
            return None

        # Create conflict record
        conflict_data = {
            "conflict_type": conflict.conflict_type.value,
            "description": conflict.description,
            "severity": conflict.severity.value,
            "status": "open",
        }

        created = await db.create("conflict", conflict_data)
        conflict_id = created["id"]

        # Create conflict_side records
        await db.create("conflict_side", {
            "conflict": conflict_id,
            "claim": conflict.claim1_id,
            "role": "claim_a",
            "weight": conflict.confidence,
        })

        await db.create("conflict_side", {
            "conflict": conflict_id,
            "claim": conflict.claim2_id,
            "role": "claim_b",
            "weight": conflict.confidence,
        })

        logger.info(f"Stored conflict {conflict_id}: {conflict.description[:50]}...")
        return conflict_id

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        resolved_by: str = "system",
        winning_claim_id: str | None = None,
    ) -> bool:
        """
        Resolve a conflict.

        Args:
            conflict_id: Conflict to resolve
            resolution: Resolution description
            resolved_by: Who resolved it
            winning_claim_id: If one claim "wins", hide the other

        Returns:
            True if resolved successfully
        """
        try:
            # Update conflict status
            await db.merge(conflict_id, {
                "status": "resolved",
                "resolution": resolution,
                "resolved_by": resolved_by,
                "resolved_at": "time::now()",
            })

            # If a winning claim is specified, hide the losing one
            if winning_claim_id:
                sides = await db.query(f"""
                    SELECT claim FROM conflict_side
                    WHERE conflict = {conflict_id}
                """)

                for side in sides[0].get("result", []) if sides else []:
                    claim_id = str(side.get("claim", ""))
                    if claim_id and claim_id != winning_claim_id:
                        await db.merge(claim_id, {"status": "superseded"})
                        logger.info(f"Marked claim {claim_id} as superseded")

            logger.info(f"Resolved conflict {conflict_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}")
            return False


# Global instance
conflict_detector = ConflictDetector()


async def detect_conflicts_for_new_claim(
    claim_id: str,
    claim_statement: str,
    claim_embedding: list[float] | None = None,
    auto_store: bool = True,
) -> list[dict[str, Any]]:
    """
    Convenience function to detect and optionally store conflicts for a new claim.

    Args:
        claim_id: ID of the new claim
        claim_statement: Statement text
        claim_embedding: Pre-computed embedding
        auto_store: Whether to store detected conflicts

    Returns:
        List of conflict info dicts
    """
    conflicts = await conflict_detector.detect_conflicts_for_claim(
        claim_id=claim_id,
        claim_statement=claim_statement,
        claim_embedding=claim_embedding,
    )

    results = []
    for conflict in conflicts:
        conflict_id = None
        if auto_store:
            conflict_id = await conflict_detector.store_conflict(conflict)

        results.append({
            "conflict_id": conflict_id,
            "claim1_id": conflict.claim1_id,
            "claim2_id": conflict.claim2_id,
            "type": conflict.conflict_type.value,
            "severity": conflict.severity.value,
            "description": conflict.description,
            "confidence": conflict.confidence,
        })

    return results


async def scan_all_conflicts(auto_store: bool = True) -> dict[str, Any]:
    """
    Scan all claims for conflicts.

    Args:
        auto_store: Whether to store detected conflicts

    Returns:
        Summary of scan results
    """
    conflicts = await conflict_detector.detect_all_conflicts()

    stored_count = 0
    if auto_store:
        for conflict in conflicts:
            conflict_id = await conflict_detector.store_conflict(conflict)
            if conflict_id:
                stored_count += 1

    return {
        "conflicts_detected": len(conflicts),
        "conflicts_stored": stored_count,
        "conflicts": [
            {
                "claim1": c.claim1_statement[:100],
                "claim2": c.claim2_statement[:100],
                "type": c.conflict_type.value,
                "severity": c.severity.value,
            }
            for c in conflicts
        ],
    }
