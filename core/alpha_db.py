"""
Alpha Database — SQLite-backed storage for all alpha backtesting results.
Replaces JSON-based alpha_store for better performance, concurrency, and querying.
"""

import sqlite3
import json
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from contextlib import contextmanager

from .alpha_lifecycle import AlphaState, validate_transition

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "alphas.db")


class AlphaDB:
    """Thread-safe SQLite storage for alpha results with full query capabilities."""

    _local = threading.local()

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=60.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alphas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expression TEXT NOT NULL,
                    alpha_id TEXT,
                    fitness REAL,
                    sharpe REAL,
                    turnover REAL,
                    returns REAL,
                    margin REAL,
                    long_count INTEGER,
                    short_count INTEGER,
                    grade TEXT,
                    source TEXT DEFAULT 'pipeline',
                    region TEXT DEFAULT 'USA',
                    universe TEXT DEFAULT 'TOP3000',
                    delay INTEGER DEFAULT 1,
                    decay INTEGER DEFAULT 0,
                    neutralization TEXT DEFAULT 'INDUSTRY',
                    truncation REAL DEFAULT 0.01,
                    status TEXT DEFAULT 'tested',
                    checks TEXT DEFAULT '[]',
                    raw_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Upgrade existing DB with raw_json if it doesn't exist
            try:
                cur.execute("ALTER TABLE alphas ADD COLUMN raw_json TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Upgrade: add lifecycle_state column
            try:
                cur.execute("ALTER TABLE alphas ADD COLUMN lifecycle_state TEXT DEFAULT 'simulated'")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
            cur.execute("""
                CREATE TABLE IF NOT EXISTS active_simulations (
                    id TEXT PRIMARY KEY,
                    process_id INTEGER,
                    start_time REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expression TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    fixed_expression TEXT,
                    fix_successful INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_alpha_fitness ON alphas(fitness)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_alpha_sharpe ON alphas(sharpe)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_alpha_source ON alphas(source)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_alpha_created ON alphas(created_at)"
            )
            # Drop old index if it exists
            cur.execute("DROP INDEX IF EXISTS idx_alpha_expr")
            # Create new composite index
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_alpha_expr_settings ON alphas(expression, region, universe, neutralization)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dpo_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    chosen TEXT,
                    rejected TEXT,
                    model_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_error_type ON errors(error_type)"
            )

    # ── Write operations ─────────────────────────────────────────────

    def save_alpha(
        self,
        expression: str,
        alpha_data: Dict,
        source: str = "pipeline",
        settings: Dict = None,
    ) -> int:
        """Save an alpha result. Returns the row ID. Upserts on duplicate expression+region."""
        is_data = alpha_data.get("is", {})
        api_settings = alpha_data.get("settings", {})
        if settings is None:
            settings = api_settings or {}

        region = settings.get("region", "USA")

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO alphas (
                    expression, alpha_id, fitness, sharpe, turnover,
                    returns, margin, long_count, short_count, grade,
                    source, region, universe, delay, decay,
                    neutralization, truncation, status, checks, raw_json,
                    lifecycle_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expression, region, universe, neutralization) DO UPDATE SET
                    fitness=excluded.fitness,
                    sharpe=excluded.sharpe,
                    turnover=excluded.turnover,
                    returns=excluded.returns,
                    alpha_id=excluded.alpha_id,
                    grade=excluded.grade,
                    status=excluded.status,
                    checks=excluded.checks,
                    raw_json=excluded.raw_json,
                    lifecycle_state=CASE
                        WHEN alphas.lifecycle_state IN ('submitted', 'checked')
                        THEN alphas.lifecycle_state
                        ELSE excluded.lifecycle_state
                    END,
                    created_at=CURRENT_TIMESTAMP
                """,
                (
                    expression,
                    alpha_data.get("id", ""),
                    is_data.get("fitness"),
                    is_data.get("sharpe"),
                    is_data.get("turnover"),
                    is_data.get("returns"),
                    is_data.get("margin"),
                    is_data.get("longCount"),
                    is_data.get("shortCount"),
                    alpha_data.get("grade", is_data.get("grade", "")),
                    source,
                    region,
                    settings.get("universe", "TOP3000"),
                    settings.get("delay", 1),
                    settings.get("decay", 0),
                    settings.get("neutralization", "INDUSTRY"),
                    settings.get("truncation", 0.01),
                    alpha_data.get("status", "tested"),
                    json.dumps(is_data.get("checks", [])),
                    json.dumps(alpha_data),
                    AlphaState.SIMULATED.value,  # first insert = just simulated
                ),
            )
            return int(cur.lastrowid or 0)

    def save_error(
        self,
        expression: str,
        error_type: str,
        error_message: str,
        fixed_expression: str = None,
        fix_successful: bool = False,
    ) -> int:
        """Save an error record for learning."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO errors (expression, error_type, error_message,
                    fixed_expression, fix_successful)
                VALUES (?, ?, ?, ?, ?)
                """,
                (expression, error_type, error_message, fixed_expression, int(fix_successful)),
            )
            return int(cur.lastrowid or 0)

    def save_dpo_pair(
        self,
        prompt: str,
        chosen: str = "",
        rejected: str = "",
        model_name: str = None
    ) -> int:
        """Save a preference pair for future LLM fine-tuning."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO dpo_pairs (prompt, chosen, rejected, model_name)
                VALUES (?, ?, ?, ?)
                """,
                (prompt, chosen, rejected, model_name),
            )
            return int(cur.lastrowid or 0)

    # ── Concurrency operations ───────────────────────────────────────

    def acquire_global_slot(self, sim_id: str, max_concurrent: int = 4, timeout: float = 300.0) -> bool:
        """Acquire a global concurrency slot using SQLite."""
        start_wait = time.time()
        pid = os.getpid()
        while time.time() - start_wait < timeout:
            try:
                with self._cursor() as cur:
                    # Clean up extremely old zombies (> 1 hour)
                    cur.execute("DELETE FROM active_simulations WHERE start_time < ?", (time.time() - 3600,))
                    
                    cur.execute("SELECT COUNT(*) FROM active_simulations")
                    count = cur.fetchone()[0]
                    if count < max_concurrent:
                        cur.execute(
                            "INSERT INTO active_simulations (id, process_id, start_time) VALUES (?, ?, ?)",
                            (sim_id, pid, time.time())
                        )
                        return True
            except sqlite3.IntegrityError:
                pass  # ID already exists
            except Exception as e:
                logger.error(f"Error acquiring slot: {e}")
            time.sleep(2)
        return False

    def release_global_slot(self, sim_id: str):
        """Release a global concurrency slot."""
        try:
            with self._cursor() as cur:
                cur.execute("DELETE FROM active_simulations WHERE id = ?", (sim_id,))
        except Exception as e:
            logger.error(f"Error releasing slot: {e}")

    # ── Read operations ──────────────────────────────────────────────

    def get_successful_alphas(
        self,
        min_fitness: float = 1.0,
        min_sharpe: float = 1.25,
        limit: int = 100,
    ) -> List[Dict]:
        """Get successful alphas sorted by fitness."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM alphas
                WHERE fitness >= ? AND sharpe >= ?
                ORDER BY fitness DESC
                LIMIT ?
                """,
                (min_fitness, min_sharpe, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_top_alphas(self, limit: int = 20, days: int = 7) -> List[Dict]:
        """Get top alphas from the last N days."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM alphas
                WHERE created_at >= ? AND fitness IS NOT NULL
                ORDER BY fitness DESC
                LIMIT ?
                """,
                (since, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_all_alphas(self, limit: int = 10000) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM alphas ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cur.fetchall()]

    def count_alphas(self, days: int = None) -> int:
        with self._cursor() as cur:
            if days:
                since = (datetime.now() - timedelta(days=days)).isoformat()
                cur.execute("SELECT COUNT(*) FROM alphas WHERE created_at >= ?", (since,))
            else:
                cur.execute("SELECT COUNT(*) FROM alphas")
            return cur.fetchone()[0]

    def expression_exists(self, expression: str, region: str = "USA", universe: str = None, neutralization: str = None) -> bool:
        """Check if an expression has already been tested with specific settings."""
        with self._cursor() as cur:
            query = "SELECT 1 FROM alphas WHERE expression = ? AND region = ?"
            params = [expression, region]
            if universe:
                query += " AND universe = ?"
                params.append(universe)
            if neutralization:
                query += " AND neutralization = ?"
                params.append(neutralization)
            query += " LIMIT 1"
            cur.execute(query, tuple(params))
            return cur.fetchone() is not None

    def delete_alpha_by_expression(
        self,
        expression: str,
        region: str = None,
        universe: str = None,
        neutralization: str = None,
    ) -> int:
        """Delete alpha records matching an expression and optional settings."""
        with self._cursor() as cur:
            query = "DELETE FROM alphas WHERE expression = ?"
            params = [expression]

            if region is not None:
                query += " AND region = ?"
                params.append(region)
            if universe is not None:
                query += " AND universe = ?"
                params.append(universe)
            if neutralization is not None:
                query += " AND neutralization = ?"
                params.append(neutralization)

            cur.execute(query, tuple(params))
            return cur.rowcount

    # ── Analytics / Retrospect ───────────────────────────────────────

    def get_operator_stats(self, days: int = 7) -> List[Dict]:
        """Get operator performance statistics."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        alphas = self.get_top_alphas(limit=500, days=days)

        import re
        op_stats: Dict[str, List[float]] = {}
        for alpha in alphas:
            expr = alpha.get("expression", "")
            fitness = alpha.get("fitness")
            if fitness is None:
                continue
            ops = re.findall(r"\b(ts_\w+|group_\w+|rank|zscore|log|sqrt|abs|sign|scale)\b", expr)
            for op in set(ops):
                if op not in op_stats:
                    op_stats[op] = []
                op_stats[op].append(fitness)

        result = []
        for op, fitnesses in op_stats.items():
            result.append({
                "operator": op,
                "count": len(fitnesses),
                "avg_fitness": round(sum(fitnesses) / len(fitnesses), 4),
                "max_fitness": round(max(fitnesses), 4),
                "success_count": sum(1 for f in fitnesses if f >= 1.0),
            })
        return sorted(result, key=lambda x: x["avg_fitness"], reverse=True)

    def get_field_stats(self, days: int = 7) -> List[Dict]:
        """Get field performance statistics."""
        alphas = self.get_top_alphas(limit=500, days=days)

        import re
        field_stats: Dict[str, List[float]] = {}
        for alpha in alphas:
            expr = alpha.get("expression", "")
            fitness = alpha.get("fitness")
            if fitness is None:
                continue
            fields = re.findall(r"[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+", expr, re.IGNORECASE)
            non_ops = [
                f for f in fields
                if not any(f.startswith(p) for p in ["ts_", "group_", "vec_"])
            ]
            for field in set(non_ops):
                if field not in field_stats:
                    field_stats[field] = []
                field_stats[field].append(fitness)

        result = []
        for field, fitnesses in field_stats.items():
            result.append({
                "field": field,
                "count": len(fitnesses),
                "avg_fitness": round(sum(fitnesses) / len(fitnesses), 4),
                "max_fitness": round(max(fitnesses), 4),
            })
        return sorted(result, key=lambda x: x["avg_fitness"], reverse=True)

    def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """Get daily mining summary for the last N days."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    DATE(created_at) as day,
                    COUNT(*) as total_tested,
                    SUM(CASE WHEN fitness >= 1.0 AND sharpe >= 1.25 THEN 1 ELSE 0 END) as successes,
                    ROUND(AVG(fitness), 4) as avg_fitness,
                    ROUND(MAX(fitness), 4) as max_fitness,
                    ROUND(AVG(sharpe), 4) as avg_sharpe,
                    ROUND(MAX(sharpe), 4) as max_sharpe
                FROM alphas
                WHERE created_at >= DATE('now', ?)
                GROUP BY DATE(created_at)
                ORDER BY day DESC
                """,
                (f"-{days} days",),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_error_stats(self, days: int = 7) -> List[Dict]:
        """Get error type statistics for self-correction learning."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    error_type,
                    COUNT(*) as count,
                    SUM(fix_successful) as fixed_count,
                    ROUND(CAST(SUM(fix_successful) AS FLOAT) / COUNT(*) * 100, 1) as fix_rate
                FROM errors
                WHERE created_at >= ?
                GROUP BY error_type
                ORDER BY count DESC
                """,
                (since,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_retrospect_report(self, days: int = 7) -> Dict:
        """Generate a comprehensive retrospect report."""
        return {
            "daily_summary": self.get_daily_summary(days),
            "top_operators": self.get_operator_stats(days)[:10],
            "top_fields": self.get_field_stats(days)[:10],
            "error_stats": self.get_error_stats(days),
            "total_alphas": self.count_alphas(),
            "recent_alphas": self.count_alphas(days=days),
            "successful_alphas": len(self.get_successful_alphas()),
        }

    # ── Lifecycle State Management ──────────────────────────────────

    def transition_state(
        self,
        expression: str,
        new_state: AlphaState,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        neutralization: str = "INDUSTRY",
    ) -> bool:
        """Transition an alpha to a new lifecycle state with validation."""
        current = self.get_lifecycle_state(
            expression, region=region, universe=universe,
            neutralization=neutralization,
        )
        validate_transition(current, new_state)

        with self._cursor() as cur:
            cur.execute(
                """UPDATE alphas SET lifecycle_state = ?
                   WHERE expression = ? AND region = ?
                     AND universe = ? AND neutralization = ?""",
                (new_state.value, expression, region, universe, neutralization),
            )
            return cur.rowcount > 0

    def get_lifecycle_state(
        self,
        expression: str,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        neutralization: str = "INDUSTRY",
    ) -> AlphaState:
        """Get the current lifecycle state of an alpha."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT lifecycle_state FROM alphas
                   WHERE expression = ? AND region = ?
                     AND universe = ? AND neutralization = ?""",
                (expression, region, universe, neutralization),
            )
            row = cur.fetchone()
            if row and row[0]:
                return AlphaState(row[0])
            return AlphaState.SIMULATED  # default for backwards compat

    def get_alphas_by_state(
        self, state: AlphaState, limit: int = 100
    ) -> List[Dict]:
        """Query alphas by lifecycle state, ordered by fitness DESC."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT * FROM alphas WHERE lifecycle_state = ?
                   ORDER BY fitness DESC LIMIT ?""",
                (state.value, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def count_by_state(self, state: Optional[AlphaState] = None) -> int:
        """Count alphas by lifecycle state, or all if state is None."""
        with self._cursor() as cur:
            if state:
                cur.execute(
                    "SELECT COUNT(*) FROM alphas WHERE lifecycle_state = ?",
                    (state.value,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM alphas")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def mark_all_submitted_in_batch(self, alpha_ids: List[str]) -> int:
        """Mark alphas as SUBMITTED by their WQ alpha_id."""
        count = 0
        with self._cursor() as cur:
            for aid in alpha_ids:
                cur.execute(
                    """UPDATE alphas SET lifecycle_state = ?
                       WHERE alpha_id = ?""",
                    (AlphaState.SUBMITTED.value, aid),
                )
                count += cur.rowcount
        return count

    # ── Migration ────────────────────────────────────────────────────

    def migrate_from_json(self, alpha_dir: str = "alpha") -> int:
        """Migrate existing JSON alpha files into SQLite."""
        if not os.path.exists(alpha_dir):
            return 0

        migrated = 0
        for filename in sorted(os.listdir(alpha_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(alpha_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for alpha in data.get("alphas", []):
                    basic = alpha.get("basic", {})
                    backtest = alpha.get("backtest", {})
                    settings = alpha.get("settings", {})
                    alpha_data = {
                        "id": basic.get("alpha_id", ""),
                        "grade": basic.get("grade", ""),
                        "is": backtest,
                        "settings": settings,
                    }
                    try:
                        self.save_alpha(
                            expression=basic.get("expression", ""),
                            alpha_data=alpha_data,
                            source=basic.get("source", "migration"),
                            settings=settings,
                        )
                        migrated += 1
                    except Exception as e:
                        logger.debug(f"Skip duplicate during migration: {e}")
            except Exception as e:
                logger.warning(f"Error migrating {filepath}: {e}")

        logger.info(f"Migrated {migrated} alphas from JSON to SQLite")
        return migrated


# ── Global singleton ─────────────────────────────────────────────────

_db_instance: Optional[AlphaDB] = None


def get_alpha_db(db_path: str = DB_PATH) -> AlphaDB:
    """Get the global AlphaDB singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AlphaDB(db_path)
    instance = _db_instance
    assert instance is not None
    return instance
