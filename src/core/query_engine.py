"""
Query Engine
============
The business logic layer between the database and the UI.
Translates user intent (hardware profile, model preferences)
into structured DuckDB queries and enriched comparison results.

The UI never touches the repository directly — everything goes
through here. This keeps the UI layer pure presentation.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from src.core.hardware_mapper import HardwareMapper
from src.core.models import BenchmarkQuery, HardwareProfile, HardwareSpec, QuantizationType
from src.db.repository import BenchmarkRepository


class QueryEngine:

    def __init__(self, repo: BenchmarkRepository):
        self.repo   = repo
        self.mapper = HardwareMapper()

    # — Primary query: what the UI calls —

    def compare(
        self,
        hardware_profile:   Optional[str]       = None,
        quantization_types: Optional[list[str]] = None,
        model_ids:          Optional[list[str]] = None,
        min_tokens_per_sec: Optional[float]     = None,
        max_memory_mb:      Optional[float]     = None,
    ) -> pd.DataFrame:
        """
        Main comparison query.
        Returns a DataFrame ready for Gradio charts and tables.
        """
        filters = BenchmarkQuery(
            hardware_profile   = HardwareProfile(hardware_profile) if hardware_profile else None,
            quantization_types = [QuantizationType(q) for q in quantization_types] if quantization_types else None,
            model_ids          = model_ids or None,
            min_tokens_per_sec = min_tokens_per_sec,
            max_memory_mb      = max_memory_mb,
        )
        return self.repo.query(filters)

    def compare_from_spec(
        self,
        spec: HardwareSpec,
        quantization_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Auto-detect hardware tier from spec and run comparison.
        Used when hardware profile is computed from user inputs.
        """
        profile = self.mapper.map(spec)
        return self.compare(
            hardware_profile   = profile.value,
            quantization_types = quantization_types,
        )

    # — Recommendation logic —

    def recommend(
        self,
        hardware_profile: str,
        max_memory_mb:    Optional[float] = None,
    ) -> dict:
        """
        Returns top-3 recommendations for a given hardware profile.
        Considers: throughput, memory efficiency, latency.

        This is a deliberate simplification — we surface the tradeoffs,
        not a single "winner", because the right model depends on workload.
        """
        df = self.compare(
            hardware_profile = hardware_profile,
            max_memory_mb    = max_memory_mb,
        )

        if df.empty:
            return {"error": "No data for this hardware profile yet."}

        # Best throughput
        best_tps = df.loc[df["tokens_per_sec"].idxmax()]

        # Best memory efficiency (tokens/sec per GB)
        best_mem = df.loc[df["tokens_per_sec_per_gb"].idxmax()]

        # Best latency (lowest latency_avg_ms)
        best_lat = df.loc[df["latency_avg_ms"].idxmin()]

        return {
            "best_throughput": {
                "model":    best_tps["model_id"],
                "quant":    best_tps["quantization"],
                "tokens_per_sec": best_tps["tokens_per_sec"],
                "reason":   "Highest raw tokens/sec — best for batch or streaming use cases.",
            },
            "best_memory_efficiency": {
                "model":    best_mem["model_id"],
                "quant":    best_mem["quantization"],
                "tps_per_gb": best_mem["tokens_per_sec_per_gb"],
                "reason":   "Most throughput per GB used — best for memory-constrained deployments.",
            },
            "best_latency": {
                "model":    best_lat["model_id"],
                "quant":    best_lat["quantization"],
                "latency_ms": best_lat["latency_avg_ms"],
                "reason":   "Lowest average latency — best for interactive / real-time apps.",
            },
        }

    # — Metadata helpers for UI dropdowns —

    def get_available_models(self) -> list[str]:
        return self.repo.get_model_list()

    def get_available_hardware_profiles(self) -> list[str]:
        return self.repo.get_hardware_profiles()

    def get_available_quantization_types(self) -> list[str]:
        return self.repo.get_quantization_types()

    def get_db_stats(self) -> dict:
        return {
            "total_records":    self.repo.count(),
            "models":           len(self.get_available_models()),
            "hardware_profiles": len(self.get_available_hardware_profiles()),
            "quantizations":    len(self.get_available_quantization_types()),
        }
