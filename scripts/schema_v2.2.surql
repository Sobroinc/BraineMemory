-- ═══════════════════════════════════════════════════════════════════════════
-- BRAINEMEMORY v2.2 COMPLETE SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════
-- Embeddings: text-embedding-3-large (native dim=3072)
-- Vision: gpt-5.2-2025-12-11
-- Vector indexes: HNSW (MTREE deprecated)
-- ═══════════════════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────────────────
-- FTS ANALYZER (required for full-text search indexes)
-- ───────────────────────────────────────────────────────────────────────────

DEFINE ANALYZER IF NOT EXISTS simple TOKENIZERS blank, class FILTERS lowercase;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 0: CONFIG (global constants)
-- Note: "settings" and "value" are reserved words in SurrealQL 2.x
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS config SCHEMAFULL;
DEFINE FIELD key ON config TYPE string;
DEFINE FIELD `value` ON config FLEXIBLE TYPE any;
DEFINE FIELD description ON config TYPE option<string>;
DEFINE FIELD updated_at ON config TYPE datetime DEFAULT time::now();
DEFINE INDEX config_key ON config FIELDS key UNIQUE;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 0: CORE INFRASTRUCTURE
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS asset SCHEMAFULL;
DEFINE FIELD type ON asset TYPE string;
DEFINE FIELD mime ON asset TYPE string;
DEFINE FIELD source_url ON asset TYPE option<string>;
DEFINE FIELD hash ON asset TYPE string;
DEFINE FIELD size_bytes ON asset TYPE option<int>;
DEFINE FIELD lang ON asset TYPE option<string>;
DEFINE FIELD version_of ON asset TYPE option<record<asset>>;
DEFINE FIELD version_tag ON asset TYPE option<string>;
DEFINE FIELD supersedes ON asset TYPE option<record<asset>>;
DEFINE FIELD status ON asset TYPE string DEFAULT "active";
DEFINE FIELD created_at ON asset TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON asset TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON asset FLEXIBLE TYPE object;

DEFINE INDEX asset_hash ON asset FIELDS hash;
DEFINE INDEX asset_status ON asset FIELDS status;
DEFINE INDEX asset_type ON asset FIELDS type;

DEFINE TABLE IF NOT EXISTS provenance SCHEMAFULL;
DEFINE FIELD asset ON provenance TYPE record<asset>;
DEFINE FIELD locator ON provenance FLEXIBLE TYPE object;
DEFINE FIELD note ON provenance TYPE option<string>;
DEFINE FIELD created_at ON provenance TYPE datetime DEFAULT time::now();

DEFINE INDEX prov_asset ON provenance FIELDS asset;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 1: CONTENT & RETRIEVAL
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
DEFINE FIELD asset ON chunk TYPE record<asset>;
DEFINE FIELD prov ON chunk TYPE option<record<provenance>>;
DEFINE FIELD content ON chunk TYPE string;
DEFINE FIELD lang ON chunk TYPE string DEFAULT "multi";
DEFINE FIELD chunk_index ON chunk TYPE int;
DEFINE FIELD vector ON chunk TYPE option<array<float>>;
DEFINE FIELD vector_model ON chunk TYPE option<string> DEFAULT "text-embedding-3-large";
DEFINE FIELD vector_dim ON chunk TYPE option<int> DEFAULT 3072;
DEFINE FIELD created_at ON chunk TYPE datetime DEFAULT time::now();

DEFINE INDEX OVERWRITE chunk_vector ON chunk FIELDS vector HNSW DIMENSION 3072 DIST COSINE;
DEFINE INDEX chunk_fts ON chunk FIELDS content SEARCH ANALYZER simple;
DEFINE INDEX chunk_asset ON chunk FIELDS asset;
DEFINE INDEX chunk_lang ON chunk FIELDS lang;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 2: KNOWLEDGE GRAPH (ENTITIES)
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD normalized_name ON entity TYPE string;
DEFINE FIELD type ON entity TYPE string;
DEFINE FIELD lang ON entity TYPE option<string>;
DEFINE FIELD description ON entity TYPE option<string>;
DEFINE FIELD external_ids ON entity FLEXIBLE TYPE object;
DEFINE FIELD embedding ON entity TYPE option<array<float>>;
DEFINE FIELD embedding_model ON entity TYPE option<string> DEFAULT "text-embedding-3-large";
DEFINE FIELD embedding_dim ON entity TYPE option<int> DEFAULT 3072;
DEFINE FIELD status ON entity TYPE string DEFAULT "active";
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON entity TYPE datetime DEFAULT time::now();

DEFINE INDEX OVERWRITE entity_vector ON entity FIELDS embedding HNSW DIMENSION 3072 DIST COSINE;
DEFINE INDEX entity_name ON entity FIELDS name SEARCH ANALYZER simple;
DEFINE INDEX entity_normalized ON entity FIELDS normalized_name;
DEFINE INDEX entity_type ON entity FIELDS type;

DEFINE TABLE IF NOT EXISTS entity_alias SCHEMAFULL;
DEFINE FIELD entity ON entity_alias TYPE record<entity>;
DEFINE FIELD alias ON entity_alias TYPE string;
DEFINE FIELD lang ON entity_alias TYPE option<string>;
DEFINE FIELD source ON entity_alias TYPE string;
DEFINE FIELD created_at ON entity_alias TYPE datetime DEFAULT time::now();

DEFINE INDEX alias_text ON entity_alias FIELDS alias SEARCH ANALYZER simple;
DEFINE INDEX alias_entity ON entity_alias FIELDS entity;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 3: CLAIMS & EVIDENCE
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS claim SCHEMAFULL;
DEFINE FIELD statement ON claim TYPE string;
DEFINE FIELD lang ON claim TYPE string DEFAULT "multi";
DEFINE FIELD confidence ON claim TYPE float;
DEFINE FIELD claim_type ON claim TYPE string DEFAULT "fact";
DEFINE FIELD valid_from ON claim TYPE option<datetime>;
DEFINE FIELD valid_until ON claim TYPE option<datetime>;
DEFINE FIELD embedding ON claim TYPE option<array<float>>;
DEFINE FIELD embedding_model ON claim TYPE option<string> DEFAULT "text-embedding-3-large";
DEFINE FIELD embedding_dim ON claim TYPE option<int> DEFAULT 3072;
DEFINE FIELD status ON claim TYPE string DEFAULT "active";
DEFINE FIELD created_at ON claim TYPE datetime DEFAULT time::now();

DEFINE INDEX OVERWRITE claim_vector ON claim FIELDS embedding HNSW DIMENSION 3072 DIST COSINE;
DEFINE INDEX claim_status ON claim FIELDS status;

DEFINE TABLE IF NOT EXISTS evidence SCHEMAFULL;
DEFINE FIELD claim ON evidence TYPE record<claim>;
DEFINE FIELD prov ON evidence TYPE record<provenance>;
DEFINE FIELD chunk ON evidence TYPE option<record<chunk>>;
DEFINE FIELD quote ON evidence TYPE option<string>;
DEFINE FIELD extraction_method ON evidence TYPE string;
DEFINE FIELD confidence ON evidence TYPE float;
DEFINE FIELD created_at ON evidence TYPE datetime DEFAULT time::now();

DEFINE INDEX evidence_claim ON evidence FIELDS claim;
DEFINE INDEX evidence_prov ON evidence FIELDS prov;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 4: CONFLICTS
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS conflict SCHEMAFULL;
DEFINE FIELD conflict_type ON conflict TYPE string;
DEFINE FIELD description ON conflict TYPE string;
DEFINE FIELD severity ON conflict TYPE string DEFAULT "medium";
DEFINE FIELD status ON conflict TYPE string DEFAULT "open";
DEFINE FIELD resolution ON conflict TYPE option<string>;
DEFINE FIELD resolved_at ON conflict TYPE option<datetime>;
DEFINE FIELD resolved_by ON conflict TYPE option<string>;
DEFINE FIELD created_at ON conflict TYPE datetime DEFAULT time::now();

DEFINE INDEX conflict_status ON conflict FIELDS status;
DEFINE INDEX conflict_severity ON conflict FIELDS severity;

DEFINE TABLE IF NOT EXISTS conflict_side SCHEMAFULL;
DEFINE FIELD conflict ON conflict_side TYPE record<conflict>;
DEFINE FIELD claim ON conflict_side TYPE record<claim>;
DEFINE FIELD role ON conflict_side TYPE string;
DEFINE FIELD weight ON conflict_side TYPE float DEFAULT 1.0;

DEFINE INDEX cside_conflict ON conflict_side FIELDS conflict;
DEFINE INDEX cside_claim ON conflict_side FIELDS claim;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 5: TIMELINE
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS timeline_event SCHEMAFULL;
DEFINE FIELD event_type ON timeline_event TYPE string;
DEFINE FIELD title ON timeline_event TYPE string;
DEFINE FIELD description ON timeline_event TYPE option<string>;
DEFINE FIELD occurred_at ON timeline_event TYPE option<datetime>;
DEFINE FIELD time_start ON timeline_event TYPE option<datetime>;
DEFINE FIELD time_end ON timeline_event TYPE option<datetime>;
DEFINE FIELD time_precision ON timeline_event TYPE string DEFAULT "exact";
DEFINE FIELD confidence ON timeline_event TYPE float DEFAULT 1.0;
DEFINE FIELD created_at ON timeline_event TYPE datetime DEFAULT time::now();

DEFINE INDEX timeline_occurred ON timeline_event FIELDS occurred_at;
DEFINE INDEX timeline_range ON timeline_event FIELDS time_start, time_end;

DEFINE TABLE IF NOT EXISTS event_participant SCHEMAFULL;
DEFINE FIELD event ON event_participant TYPE record<timeline_event>;
DEFINE FIELD entity ON event_participant TYPE record<entity>;
DEFINE FIELD role ON event_participant TYPE string;

DEFINE INDEX eparticipant_event ON event_participant FIELDS event;
DEFINE INDEX eparticipant_entity ON event_participant FIELDS entity;

DEFINE TABLE IF NOT EXISTS event_source SCHEMAFULL;
DEFINE FIELD event ON event_source TYPE record<timeline_event>;
DEFINE FIELD evidence ON event_source TYPE record<evidence>;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 6: POLICY & AUDIT
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS policy_decision SCHEMAFULL;
DEFINE FIELD action ON policy_decision TYPE string;
DEFINE FIELD target_table ON policy_decision TYPE string;
DEFINE FIELD target_id ON policy_decision TYPE string;
DEFINE FIELD reason ON policy_decision TYPE string;
DEFINE FIELD decided_by ON policy_decision TYPE string;
DEFINE FIELD user_id ON policy_decision TYPE option<string>;
DEFINE FIELD decided_at ON policy_decision TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON policy_decision FLEXIBLE TYPE object;

DEFINE INDEX policy_target ON policy_decision FIELDS target_table, target_id;
DEFINE INDEX policy_action ON policy_decision FIELDS action;
DEFINE INDEX policy_time ON policy_decision FIELDS decided_at;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 7: GRAPHRAG COMMUNITIES
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS community SCHEMAFULL;
DEFINE FIELD level ON community TYPE int;
DEFINE FIELD title ON community TYPE string;
DEFINE FIELD summary ON community TYPE string;
DEFINE FIELD parent ON community TYPE option<record<community>>;
DEFINE FIELD entity_count ON community TYPE int DEFAULT 0;
DEFINE FIELD embedding ON community TYPE option<array<float>>;
DEFINE FIELD embedding_model ON community TYPE option<string> DEFAULT "text-embedding-3-large";
DEFINE FIELD embedding_dim ON community TYPE option<int> DEFAULT 3072;
DEFINE FIELD status ON community TYPE string DEFAULT "active";
DEFINE FIELD created_at ON community TYPE datetime DEFAULT time::now();

DEFINE INDEX OVERWRITE community_vector ON community FIELDS embedding HNSW DIMENSION 3072 DIST COSINE;
DEFINE INDEX community_level ON community FIELDS level;
DEFINE INDEX community_status ON community FIELDS status;

-- Community membership edge
DEFINE TABLE IF NOT EXISTS belongs_to_community SCHEMALESS;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 8: USER MEMORY (MEM0)
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS user SCHEMAFULL;
DEFINE FIELD external_id ON user TYPE string;
DEFINE FIELD name ON user TYPE option<string>;
DEFINE FIELD preferences ON user FLEXIBLE TYPE object;
DEFINE FIELD status ON user TYPE string DEFAULT "active";
DEFINE FIELD created_at ON user TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON user TYPE datetime DEFAULT time::now();

DEFINE INDEX user_external ON user FIELDS external_id UNIQUE;

DEFINE TABLE IF NOT EXISTS session SCHEMAFULL;
DEFINE FIELD user ON session TYPE record<user>;
DEFINE FIELD started_at ON session TYPE datetime DEFAULT time::now();
DEFINE FIELD ended_at ON session TYPE option<datetime>;
DEFINE FIELD context ON session FLEXIBLE TYPE object;

DEFINE INDEX session_user ON session FIELDS user;

DEFINE TABLE IF NOT EXISTS user_memory SCHEMAFULL;
DEFINE FIELD user ON user_memory TYPE record<user>;
DEFINE FIELD content ON user_memory TYPE string;
DEFINE FIELD category ON user_memory TYPE string;
DEFINE FIELD importance ON user_memory TYPE float DEFAULT 0.5;
DEFINE FIELD embedding ON user_memory TYPE option<array<float>>;
DEFINE FIELD embedding_model ON user_memory TYPE option<string> DEFAULT "text-embedding-3-large";
DEFINE FIELD embedding_dim ON user_memory TYPE option<int> DEFAULT 3072;
DEFINE FIELD status ON user_memory TYPE string DEFAULT "active";
DEFINE FIELD created_at ON user_memory TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON user_memory TYPE datetime DEFAULT time::now();

DEFINE INDEX OVERWRITE umem_vector ON user_memory FIELDS embedding HNSW DIMENSION 3072 DIST COSINE;
DEFINE INDEX umem_user ON user_memory FIELDS user;
DEFINE INDEX umem_category ON user_memory FIELDS category;

-- ───────────────────────────────────────────────────────────────────────────
-- LAYER 9: GRAPH EDGES
-- ───────────────────────────────────────────────────────────────────────────

DEFINE TABLE IF NOT EXISTS relates_to SCHEMALESS;
DEFINE TABLE IF NOT EXISTS belongs_to SCHEMALESS;
DEFINE TABLE IF NOT EXISTS mentions SCHEMALESS;
DEFINE TABLE IF NOT EXISTS same_as SCHEMALESS;
DEFINE TABLE IF NOT EXISTS remembers SCHEMALESS;

-- ═══════════════════════════════════════════════════════════════════════════
-- EXTENSION: SPATIAL MEMORY (DRAWINGS)
-- ═══════════════════════════════════════════════════════════════════════════

DEFINE TABLE IF NOT EXISTS drawing SCHEMAFULL;
DEFINE FIELD asset ON drawing TYPE record<asset>;
DEFINE FIELD title ON drawing TYPE option<string>;
DEFINE FIELD drawing_type ON drawing TYPE string DEFAULT "unknown";
DEFINE FIELD version_tag ON drawing TYPE option<string>;
DEFINE FIELD status ON drawing TYPE string DEFAULT "active";
DEFINE FIELD created_at ON drawing TYPE datetime DEFAULT time::now();

DEFINE INDEX drawing_asset ON drawing FIELDS asset;

DEFINE TABLE IF NOT EXISTS drawing_page SCHEMAFULL;
DEFINE FIELD drawing ON drawing_page TYPE record<drawing>;
DEFINE FIELD page_no ON drawing_page TYPE int;
DEFINE FIELD layout_name ON drawing_page TYPE option<string>;
DEFINE FIELD preview_asset ON drawing_page TYPE option<record<asset>>;
DEFINE FIELD width_px ON drawing_page TYPE option<int>;
DEFINE FIELD height_px ON drawing_page TYPE option<int>;
DEFINE FIELD created_at ON drawing_page TYPE datetime DEFAULT time::now();

DEFINE INDEX dpage_drawing ON drawing_page FIELDS drawing;

DEFINE TABLE IF NOT EXISTS calibration SCHEMAFULL;
DEFINE FIELD page ON calibration TYPE record<drawing_page>;
DEFINE FIELD method ON calibration TYPE string;
DEFINE FIELD unit ON calibration TYPE string;
DEFINE FIELD scale ON calibration TYPE float;
DEFINE FIELD confidence ON calibration TYPE float;
DEFINE FIELD prov ON calibration TYPE option<record<provenance>>;
DEFINE FIELD is_active ON calibration TYPE bool DEFAULT true;
DEFINE FIELD created_at ON calibration TYPE datetime DEFAULT time::now();

DEFINE INDEX calib_page ON calibration FIELDS page;

DEFINE TABLE IF NOT EXISTS drawing_object SCHEMAFULL;
DEFINE FIELD page ON drawing_object TYPE record<drawing_page>;
DEFINE FIELD obj_type ON drawing_object TYPE string;
DEFINE FIELD label ON drawing_object TYPE option<string>;
DEFINE FIELD layer ON drawing_object TYPE option<string>;
DEFINE FIELD geometry ON drawing_object FLEXIBLE TYPE object;
DEFINE FIELD prov ON drawing_object TYPE option<record<provenance>>;
DEFINE FIELD confidence ON drawing_object TYPE float;
DEFINE FIELD status ON drawing_object TYPE string DEFAULT "active";
DEFINE FIELD created_at ON drawing_object TYPE datetime DEFAULT time::now();

DEFINE INDEX dobj_page ON drawing_object FIELDS page;
DEFINE INDEX dobj_type ON drawing_object FIELDS obj_type;
DEFINE INDEX dobj_label ON drawing_object FIELDS label;

DEFINE TABLE IF NOT EXISTS dimension SCHEMAFULL;
DEFINE FIELD page ON dimension TYPE record<drawing_page>;
DEFINE FIELD dim_kind ON dimension TYPE string;
DEFINE FIELD text_raw ON dimension TYPE option<string>;
DEFINE FIELD value ON dimension TYPE option<float>;
DEFINE FIELD unit ON dimension TYPE option<string>;
DEFINE FIELD tol_plus ON dimension TYPE option<float>;
DEFINE FIELD tol_minus ON dimension TYPE option<float>;
DEFINE FIELD p1 ON dimension FLEXIBLE TYPE object;
DEFINE FIELD p2 ON dimension FLEXIBLE TYPE object;
DEFINE FIELD estimated ON dimension TYPE bool DEFAULT true;
DEFINE FIELD source ON dimension TYPE string;
DEFINE FIELD prov ON dimension TYPE option<record<provenance>>;
DEFINE FIELD confidence ON dimension TYPE float;
DEFINE FIELD created_at ON dimension TYPE datetime DEFAULT time::now();

DEFINE INDEX dim_page ON dimension FIELDS page;
DEFINE INDEX dim_kind ON dimension FIELDS dim_kind;
DEFINE INDEX dim_value ON dimension FIELDS value;

DEFINE TABLE IF NOT EXISTS dim_applies_to SCHEMALESS;
DEFINE TABLE IF NOT EXISTS callout_targets SCHEMALESS;
DEFINE TABLE IF NOT EXISTS drawing_same_as SCHEMALESS;

-- ═══════════════════════════════════════════════════════════════════════════
-- EXTENSION: PHOTO MEMORY
-- ═══════════════════════════════════════════════════════════════════════════

DEFINE TABLE IF NOT EXISTS photo_observation SCHEMAFULL;
DEFINE FIELD asset ON photo_observation TYPE record<asset>;
DEFINE FIELD obs_type ON photo_observation TYPE string;
DEFINE FIELD severity ON photo_observation TYPE option<string>;
DEFINE FIELD description ON photo_observation TYPE option<string>;
DEFINE FIELD prov ON photo_observation TYPE option<record<provenance>>;
DEFINE FIELD confidence ON photo_observation TYPE float;
DEFINE FIELD verified ON photo_observation TYPE bool DEFAULT false;
DEFINE FIELD verified_by ON photo_observation TYPE option<string>;
DEFINE FIELD created_at ON photo_observation TYPE datetime DEFAULT time::now();

DEFINE INDEX pobs_asset ON photo_observation FIELDS asset;
DEFINE INDEX pobs_type ON photo_observation FIELDS obs_type;
DEFINE INDEX pobs_severity ON photo_observation FIELDS severity;

DEFINE TABLE IF NOT EXISTS photo_measurement SCHEMAFULL;
DEFINE FIELD asset ON photo_measurement TYPE record<asset>;
DEFINE FIELD method ON photo_measurement TYPE string;
DEFINE FIELD unit ON photo_measurement TYPE string;
DEFINE FIELD value ON photo_measurement TYPE float;
DEFINE FIELD p1 ON photo_measurement FLEXIBLE TYPE object;
DEFINE FIELD p2 ON photo_measurement FLEXIBLE TYPE object;
DEFINE FIELD estimated ON photo_measurement TYPE bool DEFAULT true;
DEFINE FIELD confidence ON photo_measurement TYPE float;
DEFINE FIELD prov ON photo_measurement TYPE option<record<provenance>>;
DEFINE FIELD note ON photo_measurement TYPE option<string>;
DEFINE FIELD created_at ON photo_measurement TYPE datetime DEFAULT time::now();

DEFINE INDEX pmeas_asset ON photo_measurement FIELDS asset;
DEFINE INDEX pmeas_method ON photo_measurement FIELDS method;

DEFINE TABLE IF NOT EXISTS evidence_of SCHEMALESS;

-- ═══════════════════════════════════════════════════════════════════════════
-- INITIAL CONFIG DATA
-- ═══════════════════════════════════════════════════════════════════════════

-- Note: This reset is intentional to ensure consistent config state
DELETE config;

CREATE config SET key = 'embedding.model', `value` = 'text-embedding-3-large', description = 'Only embedding model allowed';
CREATE config SET key = 'embedding.dim', `value` = 3072, description = 'Native dimension for text-embedding-3-large';
CREATE config SET key = 'embedding.cost_per_1m', `value` = 0.13, description = 'USD per 1M tokens';
CREATE config SET key = 'embedding.cost_per_1m_batch', `value` = 0.065, description = 'USD per 1M tokens (Batch API)';
CREATE config SET key = 'embedding.max_tokens', `value` = 8191, description = 'Max input tokens per request';

CREATE config SET key = 'vision.model', `value` = 'gpt-5.2-2025-12-11', description = 'Vision/extraction model snapshot';
CREATE config SET key = 'vision.cost_input_per_1m', `value` = 1.75, description = 'USD per 1M input tokens';
CREATE config SET key = 'vision.cost_output_per_1m', `value` = 14.0, description = 'USD per 1M output tokens';
CREATE config SET key = 'vision.cost_cached_per_1m', `value` = 0.175, description = 'USD per 1M cached input tokens';
CREATE config SET key = 'vision.context_window', `value` = 400000, description = 'Max context tokens';
CREATE config SET key = 'vision.max_output', `value` = 128000, description = 'Max output tokens';

CREATE config SET key = 'system.version', `value` = '2.2.0', description = 'BraineMemory schema version';
CREATE config SET key = 'system.index_type', `value` = 'HNSW', description = 'Vector index algorithm';
CREATE config SET key = 'system.created_at', `value` = time::now(), description = 'Schema creation timestamp';
