"""Pydantic models for BraineMemory database entities."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class AssetType(str, Enum):
    DOCUMENT = "document"
    IMAGE = "image"
    CAD = "cad"
    AUDIO = "audio"
    VIDEO = "video"
    MODEL3D = "model3d"


class AssetStatus(str, Enum):
    ACTIVE = "active"
    HIDDEN = "hidden"
    DELETED = "deleted"


class EntityType(str, Enum):
    PERSON = "person"
    ORG = "org"
    CONCEPT = "concept"
    OBJECT = "object"
    LOCATION = "location"
    EVENT = "event"
    DOCUMENT = "document"


class ClaimType(str, Enum):
    FACT = "fact"
    OPINION = "opinion"
    RULE = "rule"
    DEFINITION = "definition"


class ConflictStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class PolicyAction(str, Enum):
    REMEMBER = "remember"
    FORGET = "forget"
    UPDATE = "update"
    MERGE = "merge"
    HIDE = "hide"
    RESTORE = "restore"


# ═══════════════════════════════════════════════════════════════════════════
# CORE MODELS
# ═══════════════════════════════════════════════════════════════════════════


class Asset(BaseModel):
    """Any artifact (document, photo, CAD, audio, video)."""

    id: str | None = None
    type: AssetType
    mime: str
    source_url: str | None = None
    hash: str
    size_bytes: int | None = None
    lang: str | None = None
    version_of: str | None = None
    version_tag: str | None = None
    supersedes: str | None = None
    status: AssetStatus = AssetStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Universal locator (where exactly in the artifact)."""

    id: str | None = None
    asset: str  # asset:xxx
    locator: dict[str, Any]  # {type, page, bbox, time_range, entity_id, ...}
    note: str | None = None
    created_at: datetime | None = None


class Chunk(BaseModel):
    """Atomic content piece (for retrieval)."""

    id: str | None = None
    asset: str  # asset:xxx
    prov: str | None = None  # provenance:xxx
    content: str
    lang: str = "multi"
    chunk_index: int = 0
    vector: list[float] | None = None
    vector_model: str = "text-embedding-3-large"
    vector_dim: int = 3072
    created_at: datetime | None = None


class Entity(BaseModel):
    """Extracted entity (person, org, concept, object, etc.)."""

    id: str | None = None
    name: str
    normalized_name: str
    type: EntityType
    lang: str | None = None
    description: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    status: AssetStatus = AssetStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Claim(BaseModel):
    """Statement/fact."""

    id: str | None = None
    statement: str
    lang: str = "multi"
    confidence: float = 1.0
    claim_type: ClaimType = ClaimType.FACT
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    embedding: list[float] | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    status: AssetStatus = AssetStatus.ACTIVE
    created_at: datetime | None = None


class Evidence(BaseModel):
    """Link claim to source (provenance)."""

    id: str | None = None
    claim: str  # claim:xxx
    prov: str  # provenance:xxx
    chunk: str | None = None  # chunk:xxx
    quote: str | None = None  # max 500 chars
    extraction_method: str = "llm"  # llm|regex|manual|ocr|vision
    confidence: float = 1.0
    created_at: datetime | None = None


class Conflict(BaseModel):
    """Conflict between claims."""

    id: str | None = None
    conflict_type: str = "contradiction"  # contradiction|version_diff|ambiguity
    description: str
    severity: str = "medium"  # low|medium|high|critical
    status: ConflictStatus = ConflictStatus.OPEN
    resolution: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    created_at: datetime | None = None


class ConflictSide(BaseModel):
    """Side of a conflict (edge conflict → claim)."""

    id: str | None = None
    conflict: str  # conflict:xxx
    claim: str  # claim:xxx
    role: str = "a"  # a|b|third_party|context
    weight: float = 1.0


class PolicyDecision(BaseModel):
    """Audit trail for memory operations."""

    id: str | None = None
    action: PolicyAction
    target_table: str
    target_id: str
    reason: str
    decided_by: str = "system"  # system|user|llm|admin
    user_id: str | None = None
    decided_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserMemory(BaseModel):
    """User-specific memory (Mem0 style)."""

    id: str | None = None
    user: str  # user:xxx
    content: str
    category: str = "fact"  # preference|fact|instruction|correction
    importance: float = 0.5
    embedding: list[float] | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    status: AssetStatus = AssetStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ═══════════════════════════════════════════════════════════════════════════
# RESULT MODELS (for tool responses)
# ═══════════════════════════════════════════════════════════════════════════


class IngestResult(BaseModel):
    """Result of memory.ingest operation."""

    asset_id: str
    chunks_created: int
    entities_extracted: int
    claims_extracted: int
    conflicts_detected: int
    processing_status: str = "complete"  # complete|partial|queued


class RecallItem(BaseModel):
    """Single item in recall results."""

    type: str  # chunk|claim|entity|user_memory
    id: str
    content: str
    relevance: float
    provenance: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] | None = None


class RecallResult(BaseModel):
    """Result of memory.recall operation."""

    items: list[RecallItem]
    conflicts: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackResult(BaseModel):
    """Result of memory.context_pack operation."""

    context: str
    tokens_used: int
    sources: list[dict[str, Any]]
    user_context: dict[str, Any] | None = None
