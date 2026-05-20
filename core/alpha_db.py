"""
Alpha Database — SQLite-backed storage for all alpha backtesting results.
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

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "alphas.db")


class AlphaDB:
    """Thread-safe SQLite storage for alpha results."""

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
                    alpha_id TEXT,
                    status TEXT DEFAULT 'tested',
                    grade TEXT,
                    expression TEXT NOT NULL,
                    fitness REAL,
                    sharpe REAL,
                    turnover REAL,
                    returns REAL,
                    margin REAL,
                    pnl REAL,
                    long_count INTEGER,
                    short_count INTEGER,
                    drawdown REAL,
                    source TEXT DEFAULT 'pipeline',
                    region TEXT DEFAULT 'USA',
                    universe TEXT DEFAULT 'TOP3000',
                    delay INTEGER DEFAULT 1,
                    decay INTEGER DEFAULT 0,
                    neutralization TEXT DEFAULT 'INDUSTRY',
                    truncation REAL DEFAULT 0.08,
                    checks TEXT DEFAULT '[]',
                    raw_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (expression, region, universe, neutralization)
                )
            """)

            # Upgrade existing DB with new columns if they don't exist
            new_columns = [
                ("pnl", "REAL"),
                ("drawdown", "REAL"),
                ("long_count", "INTEGER"),
                ("short_count", "INTEGER"),
                ("raw_json", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                try:
                    cur.execute(f"ALTER TABLE alphas ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

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

            # Rescue pool for borderline alphas
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rescue_pool (
                    alpha_id TEXT PRIMARY KEY,
                    expression TEXT NOT NULL,
                    sharpe REAL,
                    fitness REAL,
                    turnover REAL,
                    failed_checks TEXT DEFAULT '[]',
                    modules_used TEXT DEFAULT '[]',
                    attempt_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    # ── Write operations ─────────────────────────────────────────────

    def add_alpha(
        self,
        expression: str,
        sharpe: float = None,
        fitness: float = None,
        alpha_id: str = "",
        turnover: float = None,
        margin: float = None,
        returns: float = None,
        pnl: float = None,
        long_count: int = None,
        short_count: int = None,
        drawdown: float = None,
        grade: str = None,
        checks: list = None,
        source: str = "pipeline",
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 0,
        neutralization: str = "NONE",
        truncation: float = 0.08,
        raw_json: str = None,
        status: str = "tested",
    ) -> int:
        """Save an alpha result. Returns 1 if successful."""
        checks_json = json.dumps(checks) if checks else "[]"
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO alphas (
                    expression, alpha_id, fitness, sharpe, turnover,
                    margin, returns, pnl, long_count, short_count,
                    drawdown, grade, checks, source, region, universe,
                    delay, decay, neutralization, truncation, raw_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(expression, region, universe, neutralization) DO UPDATE SET
                    fitness=excluded.fitness,
                    sharpe=excluded.sharpe,
                    turnover=excluded.turnover,
                    margin=excluded.margin,
                    returns=excluded.returns,
                    pnl=excluded.pnl,
                    long_count=excluded.long_count,
                    short_count=excluded.short_count,
                    drawdown=excluded.drawdown,
                    grade=excluded.grade,
                    checks=excluded.checks,
                    alpha_id=excluded.alpha_id,
                    delay=excluded.delay,
                    decay=excluded.decay,
                    truncation=excluded.truncation,
                    raw_json=excluded.raw_json,
                    status=excluded.status,
                    created_at=CURRENT_TIMESTAMP
                """,
                (
                    expression, alpha_id, fitness, sharpe, turnover,
                    margin, returns, pnl, long_count, short_count,
                    drawdown, grade, checks_json, source, region, universe,
                    delay, decay, neutralization, truncation, raw_json, status
                ),
            )
            return 1

    def save_alpha(
        self,
        expression: str,
        alpha_data: Dict,
        source: str = "pipeline",
        settings: Dict = None,
    ) -> int:
        """Save an alpha result from API response. Returns the row ID."""
        is_data = alpha_data.get("is", {})
        api_settings = alpha_data.get("settings", {})
        if settings is None:
            settings = api_settings or {}

        region = settings.get("region", "USA")

        return self.add_alpha(
            expression=expression,
            sharpe=is_data.get("sharpe"),
            fitness=is_data.get("fitness"),
            alpha_id=alpha_data.get("id", ""),
            turnover=is_data.get("turnover"),
            margin=is_data.get("margin"),
            source=source,
            region=region,
            universe=settings.get("universe", "TOP3000"),
            neutralization=settings.get("neutralization", "INDUSTRY"),
        )

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

    def update_alpha_status(self, alpha_id: str, status: str) -> int:
        """Update alpha status by alpha_id. Returns number of rows updated."""
        with self._cursor() as cur:
            cur.execute("UPDATE alphas SET status = ? WHERE alpha_id = ?", (status, alpha_id))
            return cur.rowcount

    def delete_alpha_by_alpha_id(self, alpha_id: str) -> int:
        """Delete alpha by alpha_id. Returns number of rows deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM alphas WHERE alpha_id = ?", (alpha_id,))
            return cur.rowcount

    # ── Rescue Pool operations ───────────────────────────────────────

    def add_to_rescue_pool(
        self,
        alpha_id: str,
        expression: str,
        sharpe: float = 0,
        fitness: float = 0,
        turnover: float = 0,
        failed_checks: list = None,
        modules_used: list = None,
    ) -> int:
        """Add a borderline alpha to rescue pool. Returns 1 if successful."""
        checks_json = json.dumps(failed_checks) if failed_checks else "[]"
        modules_json = json.dumps(modules_used) if modules_used else "[]"
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO rescue_pool (alpha_id, expression, sharpe, fitness, turnover, failed_checks, modules_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alpha_id) DO UPDATE SET
                    expression=excluded.expression,
                    sharpe=excluded.sharpe,
                    fitness=excluded.fitness,
                    turnover=excluded.turnover,
                    failed_checks=excluded.failed_checks,
                    modules_used=excluded.modules_used,
                    attempt_count=0,
                    created_at=CURRENT_TIMESTAMP
                """,
                (alpha_id, expression, sharpe, fitness, turnover, checks_json, modules_json),
            )
            return 1

    def get_rescue_candidate(self) -> Optional[Dict]:
        """Get a rescue candidate with attempt_count < 3. Returns None if pool is empty."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM rescue_pool
                WHERE attempt_count < 3
                ORDER BY sharpe DESC, fitness DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                result["failed_checks"] = json.loads(result.get("failed_checks", "[]"))
                result["modules_used"] = json.loads(result.get("modules_used", "[]"))
                return result
            return None

    def increment_rescue_attempt(self, alpha_id: str) -> int:
        """Increment attempt_count for a rescue candidate. Returns number of rows updated."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE rescue_pool SET attempt_count = attempt_count + 1 WHERE alpha_id = ?",
                (alpha_id,),
            )
            return cur.rowcount

    def delete_from_rescue_pool(self, alpha_id: str) -> int:
        """Delete alpha from rescue pool. Returns number of rows deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM rescue_pool WHERE alpha_id = ?", (alpha_id,))
            return cur.rowcount

    def cleanup_rescue_pool(self) -> int:
        """Delete all rescue candidates with attempt_count >= 3. Returns number of rows deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM rescue_pool WHERE attempt_count >= 3")
            return cur.rowcount

    def count_rescue_pool(self) -> int:
        """Count rescue candidates with attempt_count < 3."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rescue_pool WHERE attempt_count < 3")
            return cur.fetchone()[0]

    # ── Analytics ────────────────────────────────────────────────────

    def get_operator_stats(self, days: int = 7) -> List[Dict]:
        """Get operator performance statistics."""
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

    def get_retrospect_report(self, days: int = 7) -> Dict:
        """Generate a comprehensive retrospect report."""
        return {
            "daily_summary": self.get_daily_summary(days),
            "top_operators": self.get_operator_stats(days)[:10],
            "top_fields": self.get_field_stats(days)[:10],
            "total_alphas": self.count_alphas(),
            "recent_alphas": self.count_alphas(days=days),
            "successful_alphas": len(self.get_successful_alphas()),
        }


# ── Global singleton ─────────────────────────────────────────────────

_db_instance: Optional[AlphaDB] = None


def get_alpha_db(db_path: str = DB_PATH) -> AlphaDB:
    """Get the global AlphaDB singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AlphaDB(db_path)
    return _db_instance
