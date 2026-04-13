"""
DuckDB Repository
=================
All database read/write operations go through this class.
DuckDB gives us full OLAP analytical query power with zero infrastructure —
it runs in-process, writes to a single file, and handles columnar analytics
orders of magnitude faster than SQLite for our query patterns.

Design: thin repository layer — no business logic, pure persistence.
Business logic lives in QueryEngine (query_engine.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from rich.console import Console

from src.core.models import BenchmarkQuery, ProcessedBenchmarkRecord

console = Console()

DB_PATH = "data/benchmarks.duckdb"

# Schema version — bump when schema changes to trigger migration
SCHEMA_VERSION = 1


class BenchmarkRepository:
    """
    Handles all DuckDB interactions.
    One instance per process — DuckDB supports multiple readers
    but only one writer at a time.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._init_schema()

    # — Connection management —

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # — Schema initialization —

    def _init_schema(self):
        conn = self._connect()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                -- Identity
                model_id             TEXT    NOT NULL,
                model_file           TEXT    NOT NULL,
                quantization         TEXT    NOT NULL,
                hardware_id          TEXT    NOT NULL,
                hardware_profile     TEXT    NOT NULL,

                -- Primary metrics
                tokens_per_sec       DOUBLE  NOT NULL,
                latency_first_ms     DOUBLE  NOT NULL,
                latency_avg_ms       DOUBLE  NOT NULL,
                memory_used_mb       DOUBLE  NOT NULL,
                memory_peak_mb       DOUBLE  NOT NULL,

                -- Derived metrics
                tokens_per_sec_per_gb  DOUBLE NOT NULL,
                memory_efficiency      DOUBLE NOT NULL,
                latency_per_token_ms   DOUBLE NOT NULL,

                -- Run metadata
                prompt_tokens        INTEGER NOT NULL,
                completion_tokens    INTEGER NOT NULL,
                n_runs               INTEGER NOT NULL,
                status               TEXT    NOT NULL,

                -- Reproducibility
                timestamp            TIMESTAMP NOT NULL,
                prompt_hash          TEXT NOT NULL,
                seed                 INTEGER NOT NULL,

                -- Software stack
                python_version       TEXT,
                framework            TEXT,
                cuda_version         TEXT,
                driver_version       TEXT,

                -- Pipeline metadata
                is_outlier           BOOLEAN DEFAULT FALSE,
                pipeline_version     TEXT    DEFAULT '1.0.0',
                data_source          TEXT    DEFAULT 'real'
            )
        """)

        # Schema version tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)
        """, [str(SCHEMA_VERSION)])

        # Indexes for common query patterns
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hw_profile
            ON benchmarks (hardware_profile)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_quant
            ON benchmarks (model_id, quantization)
        """)

        console.print(f"[dim]Database initialized: {self.db_path}[/dim]")

    # — Write operations —

    def insert(self, record: ProcessedBenchmarkRecord) -> bool:
        """Insert a single processed record. Returns True on success."""
        try:
            conn = self._connect()
            d    = record.model_dump()
            conn.execute("""
                INSERT INTO benchmarks VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?
                )
            """, [
                d["model_id"], d["model_file"], d["quantization"],
                d["hardware_id"], d["hardware_profile"],

                d["tokens_per_sec"], d["latency_first_ms"], d["latency_avg_ms"],
                d["memory_used_mb"], d["memory_peak_mb"],

                d["tokens_per_sec_per_gb"], d["memory_efficiency"],
                d["latency_per_token_ms"],

                d["prompt_tokens"], d["completion_tokens"],
                d["n_runs"], d["status"],

                d["timestamp"], d["prompt_hash"], d["seed"],

                d["python_version"], d["framework"],
                d["cuda_version"], d["driver_version"],

                d["is_outlier"], d["pipeline_version"],
            ])
            return True
        except Exception as e:
            console.print(f"[red]Insert failed: {e}[/red]")
            return False

    def insert_batch(self, records: list[ProcessedBenchmarkRecord]) -> int:
        """Insert a list of records. Returns count of successful inserts."""
        success = sum(1 for r in records if self.insert(r))
        console.print(f"[green]Inserted {success}/{len(records)} records[/green]")
        return success

    def insert_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Bulk insert from a Pandas DataFrame.
        Used to load the seed CSV dataset.
        Explicitly casts columns to avoid type mismatch on booleans.
        """
        conn = self._connect()
        try:
            # Normalise boolean columns — CSV stores them as strings
            if "is_outlier" in df.columns:
                df = df.copy()
                df["is_outlier"] = df["is_outlier"].map(
                    lambda v: str(v).strip().lower() in ("true", "1", "yes")
                )
            # Add data_source if missing
            if "data_source" not in df.columns:
                df = df.copy()
                df["data_source"] = "seed"
            conn.execute("""
                INSERT INTO benchmarks
                SELECT
                    model_id, model_file, quantization, hardware_id, hardware_profile,
                    tokens_per_sec::DOUBLE, latency_first_ms::DOUBLE,
                    latency_avg_ms::DOUBLE, memory_used_mb::DOUBLE, memory_peak_mb::DOUBLE,
                    tokens_per_sec_per_gb::DOUBLE, memory_efficiency::DOUBLE,
                    latency_per_token_ms::DOUBLE,
                    prompt_tokens::INTEGER, completion_tokens::INTEGER,
                    n_runs::INTEGER, status,
                    timestamp::TIMESTAMP, prompt_hash, seed::INTEGER,
                    python_version, framework, cuda_version, driver_version,
                    is_outlier::BOOLEAN, pipeline_version, data_source
                FROM df
            """)
            return len(df)
        except Exception as e:
            console.print(f"[red]Bulk insert failed: {e}[/red]")
            return 0

    # — Read operations —

    def query(self, filters: BenchmarkQuery) -> pd.DataFrame:
        """
        Execute a filtered query and return results as a DataFrame.
        This is the hot path — called by the query engine on every UI interaction.
        """
        conn  = self._connect()
        where = ["status = 'success'", "is_outlier = FALSE"]
        params: list = []

        if filters.hardware_profile:
            where.append("hardware_profile = ?")
            params.append(filters.hardware_profile.value)

        if filters.quantization_types:
            placeholders = ", ".join("?" * len(filters.quantization_types))
            where.append(f"quantization IN ({placeholders})")
            params.extend(q.value for q in filters.quantization_types)

        if filters.model_ids:
            placeholders = ", ".join("?" * len(filters.model_ids))
            where.append(f"model_id IN ({placeholders})")
            params.extend(filters.model_ids)

        if filters.min_tokens_per_sec:
            where.append("tokens_per_sec >= ?")
            params.append(filters.min_tokens_per_sec)

        if filters.max_memory_mb:
            where.append("memory_used_mb <= ?")
            params.append(filters.max_memory_mb)

        where_clause = " AND ".join(where)
        sql = f"""
            SELECT
                model_id,
                quantization,
                hardware_profile,
                ROUND(AVG(tokens_per_sec), 2)          AS tokens_per_sec,
                ROUND(AVG(latency_first_ms), 1)        AS latency_first_ms,
                ROUND(AVG(latency_avg_ms), 1)          AS latency_avg_ms,
                ROUND(AVG(memory_used_mb), 0)          AS memory_used_mb,
                ROUND(AVG(memory_peak_mb), 0)          AS memory_peak_mb,
                ROUND(AVG(tokens_per_sec_per_gb), 2)   AS tokens_per_sec_per_gb,
                ROUND(AVG(memory_efficiency), 4)       AS memory_efficiency,
                ROUND(AVG(latency_per_token_ms), 2)    AS latency_per_token_ms,
                COUNT(*)                               AS n_records,
                MAX(timestamp)                         AS last_updated,
                STRING_AGG(DISTINCT framework, ', ')   AS frameworks
            FROM benchmarks
            WHERE {where_clause}
            GROUP BY model_id, quantization, hardware_profile
            ORDER BY tokens_per_sec DESC
            LIMIT ?
        """
        params.append(filters.limit)

        return conn.execute(sql, params).df()

    def get_all(self, include_outliers: bool = False) -> pd.DataFrame:
        """Return all records. Used for admin/debug views."""
        conn  = self._connect()
        where = "" if include_outliers else "WHERE is_outlier = FALSE"
        return conn.execute(f"SELECT * FROM benchmarks {where}").df()

    def get_model_list(self) -> list[str]:
        conn = self._connect()
        return conn.execute(
            "SELECT DISTINCT model_id FROM benchmarks ORDER BY model_id"
        ).df()["model_id"].tolist()

    def get_hardware_profiles(self) -> list[str]:
        conn = self._connect()
        return conn.execute(
            "SELECT DISTINCT hardware_profile FROM benchmarks ORDER BY hardware_profile"
        ).df()["hardware_profile"].tolist()

    def get_quantization_types(self) -> list[str]:
        conn = self._connect()
        return conn.execute(
            "SELECT DISTINCT quantization FROM benchmarks ORDER BY quantization"
        ).df()["quantization"].tolist()

    def count(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0]

    def clear(self):
        """Wipe all records. Use with care."""
        conn = self._connect()
        conn.execute("DELETE FROM benchmarks")
        console.print("[yellow]All benchmark records cleared.[/yellow]")
