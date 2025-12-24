-- ═══════════════════════════════════════════════════════════════════════════
-- BRAINEMEMORY v2.2 HEALTH CHECKS
-- ═══════════════════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────────────────
-- 1. SCHEMA VERSION CHECK
-- ───────────────────────────────────────────────────────────────────────────

LET $version = (SELECT value FROM settings WHERE key = 'system.version');
RETURN {
  check: "schema_version",
  status: IF $version[0].value = '2.2' THEN "OK" ELSE "WARN" END,
  version: $version[0].value ?? "unknown"
};

-- ───────────────────────────────────────────────────────────────────────────
-- 2. VECTOR DIMENSION CONSISTENCY
-- ───────────────────────────────────────────────────────────────────────────

LET $wrong_dim_chunks = (
  SELECT id, array::len(vector) as actual_dim
  FROM chunk
  WHERE vector IS NOT NONE AND array::len(vector) != 3072
);
RETURN {
  check: "chunk_dimension",
  status: IF array::len($wrong_dim_chunks) = 0 THEN "OK" ELSE "FAIL" END,
  wrong_count: array::len($wrong_dim_chunks),
  samples: array::slice($wrong_dim_chunks, 0, 5)
};

LET $wrong_dim_entities = (
  SELECT id, array::len(embedding) as actual_dim
  FROM entity
  WHERE embedding IS NOT NONE AND array::len(embedding) != 3072
);
RETURN {
  check: "entity_dimension",
  status: IF array::len($wrong_dim_entities) = 0 THEN "OK" ELSE "FAIL" END,
  wrong_count: array::len($wrong_dim_entities)
};

-- ───────────────────────────────────────────────────────────────────────────
-- 3. MODEL CONSISTENCY
-- ───────────────────────────────────────────────────────────────────────────

LET $models = array::distinct(
  (SELECT vector_model FROM chunk WHERE vector_model IS NOT NONE).vector_model
);
RETURN {
  check: "embedding_model_consistency",
  status: IF array::len($models) <= 1 AND ($models[0] = 'text-embedding-3-large' OR $models[0] IS NONE)
          THEN "OK" ELSE "WARN" END,
  models_found: $models
};

-- ───────────────────────────────────────────────────────────────────────────
-- 4. VECTOR COVERAGE
-- ───────────────────────────────────────────────────────────────────────────

LET $total = (SELECT count() FROM chunk GROUP ALL)[0].count ?? 0;
LET $with_vector = (SELECT count() FROM chunk WHERE vector IS NOT NONE GROUP ALL)[0].count ?? 0;
RETURN {
  check: "vector_coverage",
  total_chunks: $total,
  with_vector: $with_vector,
  coverage_pct: IF $total > 0 THEN math::round(100 * $with_vector / $total) ELSE 100 END
};

-- ───────────────────────────────────────────────────────────────────────────
-- 5. FTS COVERAGE
-- ───────────────────────────────────────────────────────────────────────────

LET $empty_content = (SELECT count() FROM chunk WHERE content IS NONE OR content = "" GROUP ALL)[0].count ?? 0;
RETURN {
  check: "fts_coverage",
  empty_content_chunks: $empty_content,
  status: IF $empty_content = 0 THEN "OK" ELSE "WARN" END
};

-- ───────────────────────────────────────────────────────────────────────────
-- 6. PROVENANCE INTEGRITY
-- ───────────────────────────────────────────────────────────────────────────

LET $orphan_chunks = (SELECT count() FROM chunk WHERE prov IS NONE GROUP ALL)[0].count ?? 0;
RETURN {
  check: "provenance_integrity",
  chunks_without_prov: $orphan_chunks,
  status: IF $orphan_chunks = 0 THEN "OK" ELSE "INFO" END
};

-- ───────────────────────────────────────────────────────────────────────────
-- 7. SETTINGS COMPLETENESS
-- ───────────────────────────────────────────────────────────────────────────

LET $required = ['embedding.model', 'embedding.dim', 'vision.model', 'system.version'];
LET $existing = (SELECT key FROM settings).key;
LET $missing = array::filter($required, |$k| !array::contains($existing, $k));
RETURN {
  check: "settings_completeness",
  status: IF array::len($missing) = 0 THEN "OK" ELSE "FAIL" END,
  missing: $missing
};

-- ───────────────────────────────────────────────────────────────────────────
-- 8. CONFLICT STATUS
-- ───────────────────────────────────────────────────────────────────────────

LET $open = (SELECT count() FROM conflict WHERE status = 'open' GROUP ALL)[0].count ?? 0;
LET $critical = (SELECT count() FROM conflict WHERE status = 'open' AND severity = 'critical' GROUP ALL)[0].count ?? 0;
RETURN {
  check: "conflicts",
  open_count: $open,
  critical_count: $critical,
  status: IF $critical > 0 THEN "CRITICAL" ELSE IF $open > 0 THEN "WARN" ELSE "OK" END END
};

-- ───────────────────────────────────────────────────────────────────────────
-- 9. TABLE COUNTS SUMMARY
-- ───────────────────────────────────────────────────────────────────────────

RETURN {
  check: "summary",
  timestamp: time::now(),
  tables: {
    assets: (SELECT count() FROM asset GROUP ALL)[0].count ?? 0,
    chunks: (SELECT count() FROM chunk GROUP ALL)[0].count ?? 0,
    entities: (SELECT count() FROM entity GROUP ALL)[0].count ?? 0,
    claims: (SELECT count() FROM claim GROUP ALL)[0].count ?? 0,
    provenance: (SELECT count() FROM provenance GROUP ALL)[0].count ?? 0,
    conflicts: (SELECT count() FROM conflict GROUP ALL)[0].count ?? 0,
    users: (SELECT count() FROM user GROUP ALL)[0].count ?? 0,
    drawings: (SELECT count() FROM drawing GROUP ALL)[0].count ?? 0,
    photos: (SELECT count() FROM photo_observation GROUP ALL)[0].count ?? 0
  }
};

-- ───────────────────────────────────────────────────────────────────────────
-- 10. INDEX STATUS (run separately)
-- ───────────────────────────────────────────────────────────────────────────

INFO FOR INDEX chunk_vector ON chunk;
INFO FOR INDEX entity_vector ON entity;
INFO FOR INDEX claim_vector ON claim;
INFO FOR INDEX umem_vector ON user_memory;
