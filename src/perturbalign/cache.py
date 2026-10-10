"""Local caching layer (persistence layer for external API data and fuzzy matching corpus).

SQLite resolve_cache: key_text -> result_json + created_at + ttl + vocab_version.
Incremental caching for online API queries; version-aware invalidation when vocabulary upgrades.
Cache as corpus: previously resolved name -> entity pairs serve as offline corpus for fuzzy matching (iter_names).
"""
from __future__ import annotations

import abc
import hashlib
import json
import os
import sqlite3
import threading
import time

_DEFAULT_CACHE = os.path.join(
    os.environ.get("PERTURBALIGN_CACHE_DIR")
    or os.path.expanduser(os.path.join("~", ".cache", "perturbalign")),
    "resolve_cache.sqlite",
)
TTL_FOREVER = None  # Exact match results never expire

_INITIALIZED_PATHS: set[str] = set()
_GLOBAL_CACHE_LOCK = threading.RLock()
_DEFAULT_CACHE_INSTANCE: SQLiteCache | None = None


class BaseCache(abc.ABC):
    """Cache abstract base class supporting multiple storage backends (SQLite, Memory, Redis, etc.)."""

    @abc.abstractmethod
    def get(self, key: str, vocab_version: str | None = None) -> dict | None:
        """Retrieve cached content by key. Returns None if expired or vocabulary version mismatches."""
        pass

    @abc.abstractmethod
    def set(
        self,
        key: str,
        result: dict,
        ttl_seconds: float | None = TTL_FOREVER,
        vocab_version: str | None = None,
    ) -> None:
        """Write or update cache entry."""
        pass

    @abc.abstractmethod
    def iter_names(
        self,
        prefix: str = "pubchem:name:",
        vocab_version: str | None = None,
    ) -> list[tuple[str, dict]]:
        """List unexpired (key_text, result) entries under specified prefix to form fuzzy matching corpus."""
        pass

    @abc.abstractmethod
    def invalidate_version(self, vocab_version: str) -> int:
        """Purge cache entries that do not match the specified vocabulary version."""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """Return the total number of cached records."""
        pass

    @abc.abstractmethod
    def clear(self) -> int:
        """Clear all cache records."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close cache connection and release resources."""
        pass


class MemoryCache(BaseCache):
    """In-memory cache implementation, suitable for containers, test suites, or lightweight ephemeral workloads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}

    def get(self, key: str, vocab_version: str | None = None) -> dict | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            entry_ver = entry.get("vocab_version")
            if vocab_version is not None and entry_ver is not None and entry_ver != vocab_version:
                return None
            ttl = entry.get("ttl_seconds")
            if ttl is not None and time.time() - entry.get("created_at", 0) > ttl:
                return None
            return entry.get("result")

    def set(
        self,
        key: str,
        result: dict,
        ttl_seconds: float | None = TTL_FOREVER,
        vocab_version: str | None = None,
    ) -> None:
        with self._lock:
            self._data[key] = {
                "result": result,
                "created_at": time.time(),
                "ttl_seconds": ttl_seconds,
                "vocab_version": vocab_version,
            }

    def iter_names(
        self,
        prefix: str = "pubchem:name:",
        vocab_version: str | None = None,
    ) -> list[tuple[str, dict]]:
        with self._lock:
            out = []
            now = time.time()
            for k, entry in self._data.items():
                if not k.startswith(prefix):
                    continue
                ver = entry.get("vocab_version")
                if vocab_version is not None and ver != vocab_version:
                    continue
                ttl = entry.get("ttl_seconds")
                if ttl is not None and now - entry.get("created_at", 0) > ttl:
                    continue
                res = entry.get("result", {})
                if res.get("unreachable") or res.get("empty"):
                    continue
                out.append((k, res))
            return out

    def invalidate_version(self, vocab_version: str) -> int:
        with self._lock:
            to_del = [
                k for k, v in self._data.items()
                if v.get("vocab_version") is None or v.get("vocab_version") != vocab_version
            ]
            for k in to_del:
                del self._data[k]
            return len(to_del)

    def count(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def close(self) -> None:
        with self._lock:
            self._data.clear()


class NullCache(BaseCache):
    """No-op cache implementation where writes are discarded and reads always return None."""

    def get(self, key: str, vocab_version: str | None = None) -> dict | None:
        return None

    def set(
        self,
        key: str,
        result: dict,
        ttl_seconds: float | None = TTL_FOREVER,
        vocab_version: str | None = None,
    ) -> None:
        pass

    def iter_names(
        self,
        prefix: str = "pubchem:name:",
        vocab_version: str | None = None,
    ) -> list[tuple[str, dict]]:
        return []

    def invalidate_version(self, vocab_version: str) -> int:
        return 0

    def count(self) -> int:
        return 0

    def clear(self) -> int:
        return 0

    def close(self) -> None:
        pass


def get_default_cache(path: str | None = None) -> SQLiteCache:
    """Get the default SQLiteCache singleton (avoiding redundant connections and DDL overhead)."""
    global _DEFAULT_CACHE_INSTANCE
    if path is not None and path != _DEFAULT_CACHE:
        return SQLiteCache(path)
    if _DEFAULT_CACHE_INSTANCE is None:
        with _GLOBAL_CACHE_LOCK:
            if _DEFAULT_CACHE_INSTANCE is None:
                _DEFAULT_CACHE_INSTANCE = SQLiteCache(_DEFAULT_CACHE)
    return _DEFAULT_CACHE_INSTANCE


class SQLiteCache(BaseCache):
    def __init__(self, path: str | None = None):
        self.path = os.path.abspath(path or _DEFAULT_CACHE)
        self._lock = threading.Lock()
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Execute DDL only once per process path to eliminate redundant table creation and schema locks
        if self.path not in _INITIALIZED_PATHS:
            with _GLOBAL_CACHE_LOCK:
                if self.path not in _INITIALIZED_PATHS:
                    self._conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS resolve_cache (
                          key_hash TEXT PRIMARY KEY,
                          key_text TEXT,
                          result_json TEXT NOT NULL,
                          created_at REAL NOT NULL,
                          ttl_seconds REAL,
                          vocab_version TEXT
                        )
                        """
                    )
                    cols = {r[1] for r in self._conn.execute("PRAGMA table_info(resolve_cache)")}
                    if "key_text" not in cols:
                        self._conn.execute("ALTER TABLE resolve_cache ADD COLUMN key_text TEXT")
                    self._conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_resolve_cache_key_text ON resolve_cache(key_text)"
                    )
                    self._conn.commit()
                    _INITIALIZED_PATHS.add(self.path)

    def get(self, key: str, vocab_version: str | None = None) -> dict | None:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json, created_at, ttl_seconds, vocab_version "
                "FROM resolve_cache WHERE key_hash = ?",
                (h,),
            ).fetchone()
        if row is None:
            return None
        result_json, created_at, ttl, ver = row
        if vocab_version is not None and ver is not None and ver != vocab_version:
            return None  # Vocabulary updated, invalidate incremental entry
        if ttl is not None and time.time() - created_at > ttl:
            return None  # Expired
        try:
            return json.loads(result_json)
        except (TypeError, ValueError):
            return None

    def set(
        self,
        key: str,
        result: dict,
        ttl_seconds: float | None = TTL_FOREVER,
        vocab_version: str | None = None,
    ) -> None:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO resolve_cache "
                "(key_hash, key_text, result_json, created_at, ttl_seconds, vocab_version) "
                "VALUES (?,?,?,?,?,?)",
                (h, key, json.dumps(result, ensure_ascii=False), time.time(),
                 ttl_seconds, vocab_version),
            )
            self._conn.commit()

    def iter_names(self, prefix: str = "pubchem:name:", vocab_version: str | None = None,
                   ) -> list[tuple[str, dict]]:
        """List unexpired (key_text, result) entries under specified prefix to form fuzzy matching corpus.

        Negative cache entries (unreachable/empty) are excluded; results are deduplicated by key_text.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_text, result_json, created_at, ttl_seconds, vocab_version "
                "FROM resolve_cache WHERE key_text LIKE ?",
                (prefix + "%",),
            ).fetchall()
        out: list[tuple[str, dict]] = []
        seen: set[str] = set()
        for key_text, result_json, created_at, ttl, ver in rows:
            if key_text is None or key_text in seen:
                continue
            if vocab_version is not None and ver != vocab_version:
                continue
            if ttl is not None and time.time() - created_at > ttl:
                continue
            try:
                result = json.loads(result_json)
            except (TypeError, ValueError):
                continue
            if result.get("unreachable") or result.get("empty"):
                continue
            seen.add(key_text)
            out.append((key_text, result))
        return out

    def invalidate_version(self, vocab_version: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM resolve_cache WHERE vocab_version IS NULL OR vocab_version != ?",
                (vocab_version,),
            )
            self._conn.commit()
            return cur.rowcount

    def export_corpus(self, target_path: str) -> int:
        """Export all unexpired valid entity records from current cache to target SQLite file."""
        target_path = os.path.abspath(target_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_hash, key_text, result_json, created_at, ttl_seconds, vocab_version "
                "FROM resolve_cache WHERE result_json NOT LIKE '%\"unreachable\": true%' "
                "AND result_json NOT LIKE '%\"empty\": true%'"
            ).fetchall()
        dst = sqlite3.connect(target_path)
        dst.execute(
            """
            CREATE TABLE IF NOT EXISTS resolve_cache (
              key_hash TEXT PRIMARY KEY,
              key_text TEXT,
              result_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              ttl_seconds REAL,
              vocab_version TEXT
            )
            """
        )
        dst.execute("CREATE INDEX IF NOT EXISTS idx_resolve_cache_key_text ON resolve_cache(key_text)")
        dst.executemany(
            "INSERT OR REPLACE INTO resolve_cache "
            "(key_hash, key_text, result_json, created_at, ttl_seconds, vocab_version) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        dst.commit()
        dst.close()
        return len(rows)

    def import_corpus(self, source_path: str) -> int:
        """Import entity corpus cache from external SQLite file (for offline cold start)."""
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Corpus file not found: {source_path}")
        src = sqlite3.connect(source_path)
        rows = src.execute(
            "SELECT key_hash, key_text, result_json, created_at, ttl_seconds, vocab_version "
            "FROM resolve_cache"
        ).fetchall()
        src.close()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO resolve_cache "
                "(key_hash, key_text, result_json, created_at, ttl_seconds, vocab_version) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def count(self) -> int:
        """Return total number of records in cache."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM resolve_cache").fetchone()
            return row[0] if row else 0

    def clear(self) -> int:
        """Clear all records from cache table."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM resolve_cache")
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
