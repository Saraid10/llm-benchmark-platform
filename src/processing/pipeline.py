"""
Processing Pipeline
===================
Transforms raw benchmark JSON files into clean, enriched
ProcessedBenchmarkRecords ready for database insertion.

Pipeline stages:
    1. Load      — read raw JSON, validate with Pydantic
    2. Normalize  — unit standardization, sanity bounds
    3. Outlier   — IQR-based per-group outlier flagging
    4. Enrich    — compute derived metrics
    5. Output    — ProcessedBenchmarkRecord

This module is intentionally stateless — same input always
produces same output. No side effects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional
import statistics

from rich.console import Console
from rich.table import Table

from src.core.models import (
    ProcessedBenchmarkRecord,
    RawBenchmarkResult,
    RunStatus,
)

console = Console()


# ---------------------------------------------------------------------------
# Sanity bounds — reject physically impossible values
# ---------------------------------------------------------------------------

BOUNDS = {
    "tokens_per_sec":   (0.1,  5000.0),   # 0.1 = almost frozen, 5000 = unrealistic
    "latency_first_ms": (1.0,  60_000.0), # 1ms = impossible fast, 60s = timeout
    "latency_avg_ms":   (0.1,  10_000.0),
    "memory_used_mb":   (1.0,  500_000.0),
}


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Validates and normalizes a single RawBenchmarkResult.
    Returns None if the record is fundamentally broken.
    """

    def normalize(self, raw: RawBenchmarkResult) -> Optional[RawBenchmarkResult]:
        if raw.status == RunStatus.FAILED:
            return None   # Never process failed runs

        # Check sanity bounds
        for field, (lo, hi) in BOUNDS.items():
            val = getattr(raw, field)
            if not math.isfinite(val) or not (lo <= val <= hi):
                console.print(
                    f"[yellow]Sanity check failed: {field}={val} "
                    f"(expected [{lo}, {hi}]) — {raw.model_id}[/yellow]"
                )
                return None

        # memory_peak must be >= memory_used
        if raw.memory_peak_mb < raw.memory_used_mb:
            # Swap them — this can happen on some platforms
            object.__setattr__(raw, "memory_peak_mb", raw.memory_used_mb)

        return raw


# ---------------------------------------------------------------------------
# Outlier detector (IQR method)
# ---------------------------------------------------------------------------

class OutlierDetector:
    """
    Flags statistical outliers within a group of records
    sharing the same (model_id, quantization, hardware_profile).

    Uses the 1.5×IQR rule on tokens_per_sec.
    Outliers are flagged, not deleted — the UI can choose to hide them.
    """

    def flag_outliers(
        self, records: list[ProcessedBenchmarkRecord]
    ) -> list[ProcessedBenchmarkRecord]:
        if len(records) < 4:
            return records   # Not enough data for IQR

        # Group by (model, quant, hw)
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(records):
            key = (r.model_id, r.quantization, r.hardware_profile)
            groups.setdefault(key, []).append(i)

        for key, indices in groups.items():
            if len(indices) < 4:
                continue
            vals = [records[i].tokens_per_sec for i in indices]
            q1   = statistics.quantiles(vals, n=4)[0]
            q3   = statistics.quantiles(vals, n=4)[2]
            iqr  = q3 - q1
            lo   = q1 - 1.5 * iqr
            hi   = q3 + 1.5 * iqr

            for i in indices:
                if not (lo <= records[i].tokens_per_sec <= hi):
                    # Replace with a flagged copy
                    d = records[i].model_dump()
                    d["is_outlier"] = True
                    records[i] = ProcessedBenchmarkRecord(**d)

        return records


# ---------------------------------------------------------------------------
# Enricher — derived metrics
# ---------------------------------------------------------------------------

class Enricher:
    """
    Computes derived metrics that make the benchmarks more comparable.
    These are the columns that show up in the UI.
    """

    def enrich(self, raw: RawBenchmarkResult) -> ProcessedBenchmarkRecord:
        # tokens/sec per GB of memory used
        memory_gb = raw.memory_used_mb / 1024
        tps_per_gb = (
            raw.tokens_per_sec / memory_gb
            if memory_gb > 0 else 0.0
        )

        # tokens/sec ÷ memory_used_mb  (raw efficiency)
        mem_efficiency = (
            raw.tokens_per_sec / raw.memory_used_mb
            if raw.memory_used_mb > 0 else 0.0
        )

        # latency per token
        lat_per_tok = (
            raw.latency_avg_ms / raw.completion_tokens
            if raw.completion_tokens > 0 else raw.latency_avg_ms
        )

        return ProcessedBenchmarkRecord(
            model_id        = raw.model_id,
            model_file      = raw.model_file,
            quantization    = raw.quantization.value,
            hardware_id     = raw.hardware_id,
            hardware_profile= raw.hardware_profile.value,

            tokens_per_sec  = round(raw.tokens_per_sec,   4),
            latency_first_ms= round(raw.latency_first_ms, 4),
            latency_avg_ms  = round(raw.latency_avg_ms,   4),
            memory_used_mb  = round(raw.memory_used_mb,   2),
            memory_peak_mb  = round(raw.memory_peak_mb,   2),

            tokens_per_sec_per_gb = round(tps_per_gb,    4),
            memory_efficiency     = round(mem_efficiency, 6),
            latency_per_token_ms  = round(lat_per_tok,   4),

            prompt_tokens    = raw.prompt_tokens,
            completion_tokens= raw.completion_tokens,
            n_runs           = raw.n_runs,
            status           = raw.status.value,

            timestamp        = raw.timestamp,
            prompt_hash      = raw.prompt_hash,
            seed             = raw.seed,

            python_version   = raw.python_version,
            framework        = raw.framework,
            cuda_version     = raw.cuda_version,
            driver_version   = raw.driver_version,

            is_outlier       = False,
            pipeline_version = "1.0.0",
            data_source      = getattr(raw, "data_source", "real"),
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Orchestrates: Load → Normalize → Enrich → (Outlier flag on batch).
    """

    def __init__(self):
        self.normalizer = Normalizer()
        self.enricher   = Enricher()
        self.detector   = OutlierDetector()

    def process_file(
        self, filepath: str | Path
    ) -> Optional[ProcessedBenchmarkRecord]:
        """Process a single raw JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            raw = RawBenchmarkResult(**data)
        except Exception as e:
            console.print(f"[red]Failed to load {filepath}: {e}[/red]")
            return None

        normalized = self.normalizer.normalize(raw)
        if normalized is None:
            return None

        return self.enricher.enrich(normalized)

    def process_directory(
        self, raw_dir: str | Path
    ) -> list[ProcessedBenchmarkRecord]:
        """
        Process all JSON files in a directory.
        Applies outlier detection across the full batch.
        """
        raw_path = Path(raw_dir)
        files    = sorted(raw_path.glob("*.json"))

        console.print(f"[cyan]Processing {len(files)} raw files...[/cyan]")

        records: list[ProcessedBenchmarkRecord] = []
        failed  = 0

        for f in files:
            if f.name.startswith("FAILED_"):
                continue
            record = self.process_file(f)
            if record:
                records.append(record)
            else:
                failed += 1

        # Batch outlier detection
        records = self.detector.flag_outliers(records)

        outliers = sum(1 for r in records if r.is_outlier)
        console.print(
            f"[green]✓ Processed:[/green] {len(records)} records "
            f"({failed} failed, {outliers} outliers flagged)"
        )

        self._print_summary(records)
        return records

    def _print_summary(self, records: list[ProcessedBenchmarkRecord]) -> None:
        if not records:
            return

        table = Table(title="Processing Summary", show_header=True)
        table.add_column("Model",        style="cyan")
        table.add_column("Quant",        style="yellow")
        table.add_column("HW Profile",   style="magenta")
        table.add_column("tok/s (avg)",  justify="right")
        table.add_column("Mem MB (avg)", justify="right")
        table.add_column("Outlier",      justify="center")

        for r in records:
            table.add_row(
                r.model_id,
                r.quantization,
                r.hardware_profile,
                f"{r.tokens_per_sec:.1f}",
                f"{r.memory_used_mb:.0f}",
                "⚠" if r.is_outlier else "✓",
            )

        console.print(table)
