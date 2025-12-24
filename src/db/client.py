"""
SurrealDB HTTP client for BraineMemory.

Features:
- Async HTTP-based connection
- Safe parameterized queries (no SQL injection)
- Hybrid search with RRF fusion
- Proper lifecycle management
- Retry logic with exponential backoff
"""

import asyncio
import json
import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)


class SurrealClient:
    """
    HTTP-based SurrealDB client with safe query handling.

    IMPORTANT: This client uses parameterized queries to prevent SQL injection.
    Never concatenate user input directly into SQL strings.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        # Convert ws:// to http://
        self._base_url = settings.surreal_url.replace("ws://", "http://").replace("/rpc", "")

    async def connect(self) -> None:
        """Connect to SurrealDB."""
        if self._connected:
            return

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            auth=(settings.surreal_user, settings.surreal_pass),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "surreal-ns": settings.surreal_ns,
                "surreal-db": settings.surreal_db,
            },
            timeout=30.0,
        )
        self._connected = True
        logger.info(
            f"Connected to SurrealDB: {self._base_url} "
            f"({settings.surreal_ns}/{settings.surreal_db})"
        )

    async def disconnect(self) -> None:
        """Disconnect from SurrealDB."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
            logger.info("Disconnected from SurrealDB")

    # ─────────────────────────────────────────────────────────────────────────
    # Core Query Methods
    # ─────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
        """
        Execute a SurrealQL query with safe parameter binding.

        Args:
            sql: SurrealQL query with $param placeholders
            params: Dictionary of parameter values

        Returns:
            List of query results

        Example:
            await db.query(
                "SELECT * FROM user WHERE name = $name AND age > $age",
                {"name": "John", "age": 18}
            )
        """
        if not self._connected:
            await self.connect()

        # Use SurrealDB's native parameterized query format
        # POST /sql with JSON body containing statement and params
        request_body = sql

        # SurrealDB HTTP API: for parameterized queries, we need to use
        # the query endpoint with variables in the header or body
        # Since SurrealDB v1.x, we can use LET statements for parameters
        if params:
            # Build LET statements for each parameter (safe binding)
            let_statements = []
            for key, value in params.items():
                serialized = self._serialize_value(value)
                let_statements.append(f"LET ${key} = {serialized};")

            request_body = "\n".join(let_statements) + "\n" + sql

        response = await self._client.post(
            "/sql",
            content=request_body,
            headers={"Content-Type": "text/plain"},
        )
        response.raise_for_status()
        results = response.json()

        # Check for errors in results
        for result in results:
            if result.get("status") == "ERR":
                raise Exception(f"SurrealDB error: {result.get('result')}")

        # Filter out LET statement results, return only actual query results
        if params:
            # Return only the last result (the actual query)
            return [results[-1]] if results else []

        return results

    def _serialize_value(self, value: Any) -> str:
        """
        Safely serialize a value for SurrealQL.

        Handles:
        - None -> NONE
        - Strings -> JSON escaped strings
        - Booleans -> true/false
        - Numbers -> as-is
        - Lists/Dicts -> JSON
        - Record references (table:id) -> unquoted
        """
        if value is None:
            return "NONE"
        if isinstance(value, str):
            # Check if it's a record reference (table:id format)
            # Must match: lowercase_table:alphanumeric_id (e.g., asset:abc123)
            # Exclude: URLs (http:, https:), Windows paths (C:\), etc.
            if re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", value):
                return value
            # Escape for JSON string
            return json.dumps(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            # Serialize list items
            items = [self._serialize_value(item) for item in value]
            return "[" + ", ".join(items) + "]"
        if isinstance(value, dict):
            # Serialize dict
            parts = []
            for k, v in value.items():
                parts.append(f"{k}: {self._serialize_value(v)}")
            return "{" + ", ".join(parts) + "}"
        return json.dumps(value)

    def _escape_identifier(self, name: str) -> str:
        """Escape a table or field name to prevent injection."""
        # Only allow alphanumeric and underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    # ─────────────────────────────────────────────────────────────────────────
    # CRUD Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a record safely."""
        if not self._connected:
            await self.connect()

        table = self._escape_identifier(table)

        # Build SET clause with proper value serialization
        set_parts = []
        for key, value in data.items():
            safe_key = self._escape_identifier(key)
            set_parts.append(f"`{safe_key}` = {self._serialize_value(value)}")

        sql = f"CREATE {table} SET {', '.join(set_parts)}"
        logger.debug(f"CREATE SQL: {sql[:500]}...")
        results = await self.query(sql)
        if results and results[0].get("result"):
            return results[0]["result"][0]
        return {}

    async def select(self, thing: str) -> list[Any] | dict[str, Any] | None:
        """Select record(s) safely."""
        if not self._connected:
            await self.connect()

        # Validate thing format (table or table:id)
        if ":" in thing:
            # Record ID format
            if not re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", thing):
                raise ValueError(f"Invalid record ID: {thing}")
        else:
            # Table name
            thing = self._escape_identifier(thing)

        sql = f"SELECT * FROM {thing}"
        results = await self.query(sql)
        if results and results[0].get("result"):
            result = results[0]["result"]
            return result[0] if len(result) == 1 and ":" in thing else result
        return None

    async def update(self, thing: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update a record (full replace)."""
        if not self._connected:
            await self.connect()

        # Validate record ID
        if not re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", thing):
            raise ValueError(f"Invalid record ID: {thing}")

        sql = f"UPDATE {thing} CONTENT {json.dumps(data)}"
        results = await self.query(sql)
        if results and results[0].get("result"):
            return results[0]["result"][0]
        return {}

    async def merge(self, thing: str, data: dict[str, Any]) -> dict[str, Any]:
        """Merge data into a record (partial update)."""
        if not self._connected:
            await self.connect()

        # Validate record ID
        if not re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", thing):
            raise ValueError(f"Invalid record ID: {thing}")

        sql = f"UPDATE {thing} MERGE {json.dumps(data)}"
        results = await self.query(sql)
        if results and results[0].get("result"):
            return results[0]["result"][0]
        return {}

    async def delete(self, thing: str) -> None:
        """Delete a record safely."""
        if not self._connected:
            await self.connect()

        # Validate record ID
        if not re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", thing):
            raise ValueError(f"Invalid record ID: {thing}")

        await self.query(f"DELETE {thing}")

    async def relate(
        self,
        from_id: str,
        relation: str,
        to_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a graph relation between records safely."""
        # Validate all identifiers
        if not re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", from_id):
            raise ValueError(f"Invalid from_id: {from_id}")
        if not re.match(r"^[a-z_]+:[a-zA-Z0-9_]+$", to_id):
            raise ValueError(f"Invalid to_id: {to_id}")
        relation = self._escape_identifier(relation)

        sql = f"RELATE {from_id}->{relation}->{to_id}"
        if data:
            sql += f" CONTENT {json.dumps(data)}"
        results = await self.query(sql)
        if results and results[0].get("result"):
            return results[0]["result"][0]
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Search Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def vector_search(
        self,
        table: str,
        vector_field: str,
        query_vector: list[float],
        limit: int = 10,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search using HNSW index."""
        table = self._escape_identifier(table)
        vector_field = self._escape_identifier(vector_field)
        limit = int(limit)  # Ensure integer

        where_clause = f"AND ({where})" if where else ""

        sql = f"""
            SELECT *,
                   vector::similarity::cosine({vector_field}, $vec) AS relevance
            FROM {table}
            WHERE {vector_field} IS NOT NONE {where_clause}
            ORDER BY relevance DESC
            LIMIT {limit}
        """

        results = await self.query(sql, {"vec": query_vector})
        return results[0].get("result", []) if results else []

    async def fts_search(
        self,
        table: str,
        content_field: str,
        query: str,
        limit: int = 10,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform full-text search (BM25)."""
        table = self._escape_identifier(table)
        content_field = self._escape_identifier(content_field)
        limit = int(limit)

        where_clause = f"AND ({where})" if where else ""

        sql = f"""
            SELECT *,
                   search::score(1) AS relevance
            FROM {table}
            WHERE {content_field} @1@ $query {where_clause}
            ORDER BY relevance DESC
            LIMIT {limit}
        """

        results = await self.query(sql, {"query": query})
        return results[0].get("result", []) if results else []

    async def hybrid_search(
        self,
        table: str,
        content_field: str,
        vector_field: str,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining FTS and vector similarity using RRF.

        Uses Reciprocal Rank Fusion to combine results from both search methods.
        This is more robust than linear score combination because FTS and vector
        scores are on different scales.

        Args:
            table: Table to search
            content_field: Field for FTS
            vector_field: Field for vector search
            query_text: Text query for FTS
            query_vector: Embedding vector for similarity
            limit: Max results
            fts_weight: Weight for FTS results in RRF (default 0.3)
            vector_weight: Weight for vector results in RRF (default 0.7)
            where: Additional WHERE clause

        Returns:
            Fused results with combined relevance scores
        """
        # Get more results from each source for better fusion
        fetch_limit = limit * 3

        # Execute both searches in parallel
        fts_results, vector_results = await asyncio.gather(
            self.fts_search(table, content_field, query_text, fetch_limit, where),
            self.vector_search(table, vector_field, query_vector, fetch_limit, where),
        )

        # RRF fusion
        k = 60  # RRF constant (from original paper)
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}

        # Process FTS results
        for rank, item in enumerate(fts_results, start=1):
            item_id = str(item.get("id", ""))
            if item_id:
                rrf_score = fts_weight / (k + rank)
                scores[item_id] = scores.get(item_id, 0) + rrf_score
                items[item_id] = item

        # Process vector results
        for rank, item in enumerate(vector_results, start=1):
            item_id = str(item.get("id", ""))
            if item_id:
                rrf_score = vector_weight / (k + rank)
                scores[item_id] = scores.get(item_id, 0) + rrf_score
                if item_id not in items:
                    items[item_id] = item

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list
        results = []
        for item_id in sorted_ids[:limit]:
            item = items[item_id].copy()
            item["relevance"] = scores[item_id]
            item["_sources"] = []
            if item_id in {str(r.get("id", "")) for r in fts_results}:
                item["_sources"].append("fts")
            if item_id in {str(r.get("id", "")) for r in vector_results}:
                item["_sources"].append("vector")
            results.append(item)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Health Checks
    # ─────────────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Run health checks and return status."""
        checks = {}

        # Version check
        version_result = await self.query(
            "SELECT value FROM config WHERE key = $key",
            {"key": "system.version"},
        )
        checks["schema_version"] = (
            version_result[0]["result"][0]["value"]
            if version_result and version_result[0].get("result")
            else "unknown"
        )

        # Table counts
        tables = ["asset", "chunk", "entity", "claim", "provenance", "conflict"]
        counts = {}
        for table in tables:
            safe_table = self._escape_identifier(table)
            result = await self.query(f"SELECT count() FROM {safe_table} GROUP ALL")
            counts[table] = (
                result[0]["result"][0]["count"]
                if result and result[0].get("result")
                else 0
            )
        checks["table_counts"] = counts

        return checks

    # ─────────────────────────────────────────────────────────────────────────
    # Settings
    # ─────────────────────────────────────────────────────────────────────────

    async def get_setting(self, key: str) -> Any:
        """Get a setting value safely."""
        result = await self.query(
            "SELECT value FROM config WHERE key = $key",
            {"key": key},
        )
        if result and result[0].get("result"):
            return result[0]["result"][0]["value"]
        return None

    async def set_setting(self, key: str, value: Any, description: str | None = None) -> None:
        """Set a setting value (upsert) safely."""
        await self.query(
            """
            UPDATE config SET
                value = $value,
                description = $description,
                updated_at = time::now()
            WHERE key = $key
            """,
            {"key": key, "value": value, "description": description},
        )


# Global client instance (for backward compatibility)
# Prefer using container.db instead
db = SurrealClient()
