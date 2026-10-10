"""Embedded Graph Database Storage Engine for HeteroKG.

Provides a persistent, SQL-indexed graph database backend based on SQLite (and optional DuckDB).
Supports:
1. High-concurrency ACID graph persistence with node & edge tables.
2. Fast indexed neighbor and relation lookups.
3. Recursive CTE (Common Table Expressions) for multi-hop graph path traversal directly in SQL.
4. Neighborhood subgraph extraction via SQL queries.
5. In-memory and persistent on-disk graph database synchronization.
"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .hetero_kg import HeteroEdge, HeteroNode, HeteroPath, HeteroSubgraph, HeterogeneousKnowledgeGraph


class SQLiteGraphStore:
    """Embedded SQLite Graph Database Storage Engine."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize the graph database connection.

        Args:
            db_path: Path to the SQLite database file, or ':memory:' for in-memory DB.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize graph schema and high-performance indexes."""
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA journal_mode = WAL;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                attributes TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON kg_nodes(node_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_nodes_name ON kg_nodes(name);")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source_db TEXT DEFAULT 'curated',
                effect_sign INTEGER DEFAULT 0,
                attributes TEXT,
                FOREIGN KEY (source) REFERENCES kg_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES kg_nodes(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(source, relation);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_edges_dst ON kg_edges(target, relation);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_edges_rel ON kg_edges(relation);")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_edges_uniq ON kg_edges(source, target, relation, source_db);")
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()

    def __enter__(self) -> SQLiteGraphStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def count_nodes(self) -> int:
        """Return total number of nodes in graph database."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM kg_nodes;")
        return cur.fetchone()[0]

    def count_edges(self) -> int:
        """Return total number of edges in graph database."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM kg_edges;")
        return cur.fetchone()[0]

    def insert_node(self, node: HeteroNode) -> None:
        """Insert or replace a single node."""
        from .hetero_kg import HeteroNode
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO kg_nodes (id, name, node_type, attributes) VALUES (?, ?, ?, ?);",
            (node.id, node.name, node.node_type, json.dumps(node.attributes)),
        )
        self._conn.commit()

    def insert_nodes(self, nodes: Iterable[HeteroNode]) -> None:
        """Batch insert or replace nodes."""
        cur = self._conn.cursor()
        payload = [
            (n.id, n.name, n.node_type, json.dumps(n.attributes))
            for n in nodes
        ]
        cur.executemany(
            "INSERT OR REPLACE INTO kg_nodes (id, name, node_type, attributes) VALUES (?, ?, ?, ?);",
            payload,
        )
        self._conn.commit()

    def insert_edge(self, edge: HeteroEdge) -> None:
        """Insert or replace a single edge."""
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO kg_edges (source, target, relation, weight, source_db, effect_sign, attributes)
               VALUES (?, ?, ?, ?, ?, ?, ?);""",
            (edge.source, edge.target, edge.relation, edge.weight, edge.source_db, edge.effect_sign, json.dumps(edge.attributes)),
        )
        self._conn.commit()

    def insert_edges(self, edges: Iterable[HeteroEdge]) -> None:
        """Batch insert or replace edges."""
        cur = self._conn.cursor()
        payload = [
            (e.source, e.target, e.relation, e.weight, e.source_db, e.effect_sign, json.dumps(e.attributes))
            for e in edges
        ]
        cur.executemany(
            """INSERT OR REPLACE INTO kg_edges (source, target, relation, weight, source_db, effect_sign, attributes)
               VALUES (?, ?, ?, ?, ?, ?, ?);""",
            payload,
        )
        self._conn.commit()

    def save_graph(self, graph: HeterogeneousKnowledgeGraph) -> dict[str, int]:
        """Persist an in-memory HeterogeneousKnowledgeGraph into the database.

        Args:
            graph: Source HeterogeneousKnowledgeGraph instance.

        Returns:
            Dictionary with saved node and edge counts.
        """
        self.insert_nodes(graph.nodes.values())
        all_edges: list[HeteroEdge] = []
        for edge_list in graph.adj.values():
            for _, edge in edge_list:
                all_edges.append(edge)
        self.insert_edges(all_edges)
        return {"nodes_saved": len(graph.nodes), "edges_saved": len(all_edges)}

    def load_graph(self, target_graph: HeterogeneousKnowledgeGraph) -> dict[str, int]:
        """Load nodes and edges from database into an in-memory graph.

        Args:
            target_graph: Target HeterogeneousKnowledgeGraph to populate.

        Returns:
            Dictionary with loaded node and edge counts.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT id, name, node_type, attributes FROM kg_nodes;")
        n_count = 0
        for row in cur.fetchall():
            attrs = json.loads(row["attributes"]) if row["attributes"] else {}
            target_graph.add_node(row["id"], row["name"], row["node_type"], **attrs)
            n_count += 1

        cur.execute("SELECT source, target, relation, weight, source_db, effect_sign, attributes FROM kg_edges;")
        e_count = 0
        for row in cur.fetchall():
            attrs = json.loads(row["attributes"]) if row["attributes"] else {}
            target_graph.add_edge(
                source=row["source"],
                target=row["target"],
                relation=row["relation"],
                weight=row["weight"],
                source_db=row["source_db"],
                effect_sign=row["effect_sign"],
                **attrs,
            )
            e_count += 1

        return {"nodes_loaded": n_count, "edges_loaded": e_count}

    def query_nodes(
        self,
        node_type: str | None = None,
        name_contains: str | None = None,
        limit: int = 100,
    ) -> list[HeteroNode]:
        """Query nodes from database with filters."""
        from .hetero_kg import HeteroNode
        query = "SELECT id, name, node_type, attributes FROM kg_nodes WHERE 1=1"
        params: list[Any] = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)
        if name_contains:
            query += " AND name LIKE ?"
            params.append(f"%{name_contains}%")
        query += " LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        nodes = []
        for row in cur.fetchall():
            attrs = json.loads(row["attributes"]) if row["attributes"] else {}
            nodes.append(HeteroNode(id=row["id"], name=row["name"], node_type=row["node_type"], attributes=attrs))
        return nodes

    def query_edges(
        self,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
        limit: int = 100,
    ) -> list[HeteroEdge]:
        """Query edges from database with filters."""
        from .hetero_kg import HeteroEdge
        query = "SELECT source, target, relation, weight, source_db, effect_sign, attributes FROM kg_edges WHERE 1=1"
        params: list[Any] = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if target:
            query += " AND target = ?"
            params.append(target)
        if relation:
            query += " AND relation = ?"
            params.append(relation)
        query += " LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        cur.execute(query, params)
        edges = []
        for row in cur.fetchall():
            attrs = json.loads(row["attributes"]) if row["attributes"] else {}
            edges.append(
                HeteroEdge(
                    source=row["source"],
                    target=row["target"],
                    relation=row["relation"],
                    weight=row["weight"],
                    source_db=row["source_db"],
                    effect_sign=row["effect_sign"],
                    attributes=attrs,
                )
            )
        return edges

    def find_paths_sql(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> list[HeteroPath]:
        """Find multi-hop causal paths using SQLite Recursive CTE graph traversal.

        Performs pure database-level graph search with cycle avoidance.
        """
        from .hetero_kg import HeteroEdge, HeteroNode, HeteroPath

        sql = """
        WITH RECURSIVE path_search(
            src, current, hops, node_path, edge_id_path, total_weight, effect_prod
        ) AS (
            -- Base anchor query: direct 1-hop edges
            SELECT
                source,
                target,
                1,
                source || '|' || target,
                CAST(id AS TEXT),
                weight,
                CASE WHEN effect_sign = 0 THEN 1 ELSE effect_sign END
            FROM kg_edges
            WHERE source = ? AND weight >= ?

            UNION ALL

            -- Recursive step: join next-hop edge
            SELECT
                p.src,
                e.target,
                p.hops + 1,
                p.node_path || '|' || e.target,
                p.edge_id_path || '|' || CAST(e.id AS TEXT),
                p.total_weight * e.weight,
                p.effect_prod * (CASE WHEN e.effect_sign = 0 THEN 1 ELSE e.effect_sign END)
            FROM path_search p
            JOIN kg_edges e ON p.current = e.source
            WHERE p.hops < ?
              AND instr(p.node_path, '|' || e.target) = 0  -- prevent loops
              AND e.weight >= ?
        )
        SELECT hops, node_path, edge_id_path, total_weight, effect_prod
        FROM path_search
        WHERE current = ?
        ORDER BY total_weight DESC, hops ASC
        LIMIT ?;
        """
        cur = self._conn.cursor()
        cur.execute(sql, (source_id, min_weight, max_hops, min_weight, target_id, limit))
        rows = cur.fetchall()

        paths: list[HeteroPath] = []
        for row in rows:
            node_ids = row["node_path"].split("|")
            edge_ids = [int(eid) for eid in row["edge_id_path"].split("|")]

            # Retrieve full node details
            placeholders = ",".join("?" for _ in node_ids)
            cur.execute(f"SELECT id, name, node_type, attributes FROM kg_nodes WHERE id IN ({placeholders})", node_ids)
            node_map = {
                r["id"]: HeteroNode(r["id"], r["name"], r["node_type"], json.loads(r["attributes"] or "{}"))
                for r in cur.fetchall()
            }
            ordered_nodes = [node_map[nid] for nid in node_ids if nid in node_map]

            # Retrieve full edge details
            edge_placeholders = ",".join("?" for _ in edge_ids)
            cur.execute(
                f"""SELECT id, source, target, relation, weight, source_db, effect_sign, attributes
                    FROM kg_edges WHERE id IN ({edge_placeholders})""",
                edge_ids,
            )
            edge_map = {
                r["id"]: HeteroEdge(
                    r["source"], r["target"], r["relation"], r["weight"], r["source_db"], r["effect_sign"], json.loads(r["attributes"] or "{}")
                )
                for r in cur.fetchall()
            }
            ordered_edges = [edge_map[eid] for eid in edge_ids if eid in edge_map]

            paths.append(
                HeteroPath(
                    source_id=source_id,
                    target_id=target_id,
                    hops=row["hops"],
                    nodes=ordered_nodes,
                    edges=ordered_edges,
                    confidence_score=float(row["total_weight"]),
                )
            )

        return paths

    def get_neighborhood_sql(self, center_id: str, radius: int = 1) -> HeteroSubgraph:
        """Extract neighborhood subgraph surrounding a node using SQLite Recursive CTE."""
        from .hetero_kg import HeteroEdge, HeteroNode, HeteroSubgraph

        sql = """
        WITH RECURSIVE neighborhood(node_id, depth) AS (
            SELECT ?, 0
            UNION
            SELECT CASE WHEN e.source = n.node_id THEN e.target ELSE e.source END, n.depth + 1
            FROM neighborhood n
            JOIN kg_edges e ON (e.source = n.node_id OR e.target = n.node_id)
            WHERE n.depth < ?
        )
        SELECT DISTINCT node_id FROM neighborhood;
        """
        cur = self._conn.cursor()
        cur.execute(sql, (center_id, radius))
        sub_node_ids = [r["node_id"] for r in cur.fetchall()]

        if not sub_node_ids:
            return HeteroSubgraph(center_id=center_id, radius=radius, nodes={}, edges=[])

        # Fetch nodes
        placeholders = ",".join("?" for _ in sub_node_ids)
        cur.execute(f"SELECT id, name, node_type, attributes FROM kg_nodes WHERE id IN ({placeholders})", sub_node_ids)
        nodes = {
            r["id"]: HeteroNode(r["id"], r["name"], r["node_type"], json.loads(r["attributes"] or "{}"))
            for r in cur.fetchall()
        }

        # Fetch edges between nodes in subgraph
        cur.execute(
            f"""SELECT source, target, relation, weight, source_db, effect_sign, attributes
                FROM kg_edges
                WHERE source IN ({placeholders}) AND target IN ({placeholders})""",
            sub_node_ids + sub_node_ids,
        )
        edges = [
            HeteroEdge(
                r["source"], r["target"], r["relation"], r["weight"], r["source_db"], r["effect_sign"], json.loads(r["attributes"] or "{}")
            )
            for r in cur.fetchall()
        ]

        return HeteroSubgraph(center_id=center_id, radius=radius, nodes=nodes, edges=edges)
