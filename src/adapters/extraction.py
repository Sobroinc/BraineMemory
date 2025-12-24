"""
Entity and Relation Extraction Adapter.

Uses LLM to extract:
- Named entities (people, organizations, places, etc.)
- Relations between entities
- Claims/facts with evidence

Features:
- Retry logic with exponential backoff
- Proper HTTP client lifecycle
- Structured JSON output parsing
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Extracted entity from text."""
    name: str
    entity_type: str  # person, organization, location, document, concept, event
    description: str
    importance: float  # 0-1
    aliases: list[str] | None = None


@dataclass
class ExtractedRelation:
    """Extracted relation between entities."""
    source: str  # entity name
    target: str  # entity name
    relation_type: str  # works_for, located_in, owns, mentions, etc.
    description: str
    weight: float  # 0-1


@dataclass
class ExtractedClaim:
    """Extracted claim/fact from text."""
    statement: str
    subject: str  # entity name
    confidence: float
    evidence_quote: str


@dataclass
class ExtractionResult:
    """Result of extraction."""
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    claims: list[ExtractedClaim]
    language: str


EXTRACTION_PROMPT = """Analyze the following text and extract structured information.

TEXT:
{text}

Extract the following in JSON format:
{{
  "entities": [
    {{
      "name": "exact name as appears in text",
      "entity_type": "person|organization|location|document|concept|event|date|money",
      "description": "brief description based on context",
      "importance": 0.0-1.0,
      "aliases": ["alternative names if any"]
    }}
  ],
  "relations": [
    {{
      "source": "entity name",
      "target": "entity name",
      "relation_type": "works_for|located_in|owns|created|signed|mentions|related_to|part_of|dated",
      "description": "description of relation",
      "weight": 0.0-1.0
    }}
  ],
  "claims": [
    {{
      "statement": "factual claim from the text",
      "subject": "main entity this claim is about",
      "confidence": 0.0-1.0,
      "evidence_quote": "exact quote supporting this claim"
    }}
  ],
  "language": "detected language code (fr/en/ru)"
}}

Rules:
- Extract ALL named entities (people, companies, addresses, dates, amounts)
- Create relations between entities that are connected in the text
- Extract factual claims that can be verified
- Use exact names as they appear in the text
- Importance: 1.0 for main subjects, 0.5 for mentioned entities, 0.3 for context
- Return valid JSON only, no markdown"""


class ExtractionAdapter:
    """
    Adapter for entity/relation extraction using LLM.

    Features proper retry logic and HTTP client lifecycle management.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self.model = settings.llm_model
        self.api_key = settings.openai_api_key
        logger.info(f"ExtractionAdapter initialized: {self.model}")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=90.0,  # Increased timeout for extraction
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Call LLM with retry logic.

        Retries on:
        - HTTP errors (5xx, rate limits)
        - Timeout errors
        """
        client = await self._ensure_client()

        response = await client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from documents. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def extract(
        self,
        text: str,
        max_entities: int = 20,
        min_importance: float = 0.3,
    ) -> ExtractionResult:
        """
        Extract entities, relations, and claims from text.

        Args:
            text: Text to analyze
            max_entities: Maximum entities to extract
            min_importance: Minimum importance threshold

        Returns:
            ExtractionResult with entities, relations, claims
        """
        # Truncate if too long (keep ~4000 tokens worth)
        if len(text) > 12000:
            text = text[:12000] + "..."

        prompt = EXTRACTION_PROMPT.format(text=text)

        try:
            content = await self._call_llm(prompt)

            # Clean up response (remove markdown if present)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            # Parse JSON
            result = json.loads(content)

            # Convert to dataclasses
            entities = []
            for e in result.get("entities", []):
                if e.get("importance", 0) >= min_importance:
                    entities.append(ExtractedEntity(
                        name=e["name"],
                        entity_type=e.get("entity_type", "concept"),
                        description=e.get("description", ""),
                        importance=float(e.get("importance", 0.5)),
                        aliases=e.get("aliases"),
                    ))

            # Limit entities
            entities = sorted(entities, key=lambda x: x.importance, reverse=True)[:max_entities]

            relations = []
            entity_names = {e.name.lower() for e in entities}
            for r in result.get("relations", []):
                # Only include relations where both entities are in our list
                if r.get("source", "").lower() in entity_names or r.get("target", "").lower() in entity_names:
                    relations.append(ExtractedRelation(
                        source=r["source"],
                        target=r["target"],
                        relation_type=r.get("relation_type", "related_to"),
                        description=r.get("description", ""),
                        weight=float(r.get("weight", 0.5)),
                    ))

            claims = []
            for c in result.get("claims", []):
                claims.append(ExtractedClaim(
                    statement=c["statement"],
                    subject=c.get("subject", ""),
                    confidence=float(c.get("confidence", 0.5)),
                    evidence_quote=c.get("evidence_quote", ""),
                ))

            logger.info(
                f"Extracted: {len(entities)} entities, "
                f"{len(relations)} relations, {len(claims)} claims"
            )

            return ExtractionResult(
                entities=entities,
                relations=relations,
                claims=claims,
                language=result.get("language", "unknown"),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction result: {e}")
            return ExtractionResult(entities=[], relations=[], claims=[], language="unknown")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(entities=[], relations=[], claims=[], language="unknown")

    async def extract_entities_only(
        self,
        text: str,
        max_entities: int = 15,
    ) -> list[ExtractedEntity]:
        """Quick extraction of entities only (faster, cheaper)."""
        result = await self.extract(text, max_entities=max_entities)
        return result.entities


# Global instance (for backward compatibility)
# Prefer using container.extraction instead
extraction = ExtractionAdapter()
