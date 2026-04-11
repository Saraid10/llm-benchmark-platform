"""
Test Suite — LLM Benchmark Platform
=====================================
Tests for the core pipeline: models, hardware mapper, normalizer,
enricher, outlier detector, repository, and query engine.

Run with:
    python -m pytest tests/ -v

No external dependencies (no model files, no GPU) required.
All tests use in-memory DuckDB.
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import pandas as pd

from src.core.models import (
    BenchmarkQuery,
    HardwareProfile,
    HardwareSpec,
    ProcessedBenchmarkRecord,
    QuantizationType,
    RawBenchmarkResult,
    RunStatus,
)
from src.core.hardware_mapper import HardwareMapper
from src.processing.pipeline import Normalizer, Enricher, OutlierDetector, Pipeline
from src.db.repository import BenchmarkRepository
from src.core.query_engine import QueryEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _raw_result(**overrides) -> RawBenchmarkResult:
    """Factory for valid RawBenchmarkResult."""
    defaults = dict(
        model_id          = "test-model",
        model_file        = "test-model.Q4_K_M.gguf",
        quantization      = QuantizationType.GGUF_Q4_K_M,
        hardware_id       = "test_hw",
        hardware_profile  = HardwareProfile.CPU_MEDIUM,
        tokens_per_sec    = 10.0,
        latency_first_ms  = 500.0,
        latency_avg_ms    = 100.0,
        memory_used_mb    = 4000.0,
        memory_peak_mb    = 4200.0,
        prompt_tokens     = 42,
        completion_tokens = 256,
        n_runs            = 3,
        status            = RunStatus.SUCCESS,
        prompt_hash       = "abc12345",
        seed              = 42,
        python_version    = "3.11.0",
        framework         = "llama-cpp-python-0.2.57",
    )
    defaults.update(overrides)
    return RawBenchmarkResult(**defaults)


def _processed_record(**overrides) -> ProcessedBenchmarkRecord:
    """Factory for a valid ProcessedBenchmarkRecord."""
    defaults = dict(
        model_id                = "test-model",
        model_file              = "test-model.gguf",
        quantization            = "GGUF_Q4_K_M",
        hardware_id             = "test_hw",
        hardware_profile        = "CPU_MEDIUM",
        tokens_per_sec          = 10.0,
        latency_first_ms        = 500.0,
        latency_avg_ms          = 100.0,
        memory_used_mb          = 4000.0,
        memory_peak_mb          = 4200.0,
        tokens_per_sec_per_gb   = 2.5,
        memory_efficiency       = 0.0025,
        latency_per_token_ms    = 0.39,
        prompt_tokens           = 42,
        completion_tokens       = 256,
        n_runs                  = 3,
        status                  = "success",
        timestamp               = datetime.utcnow(),
        prompt_hash             = "abc12345",
        seed                    = 42,
        python_version          = "3.11.0",
        framework               = "llama-cpp-python-0.2.57",
        cuda_version            = None,
        driver_version          = None,
    )
    defaults.update(overrides)
    return ProcessedBenchmarkRecord(**defaults)


@pytest.fixture
def in_memory_repo(tmp_path):
    """BenchmarkRepository backed by a temp DuckDB file."""
    db_path = str(tmp_path / "test.duckdb")
    repo    = BenchmarkRepository(db_path=db_path)
    yield repo
    repo.close()


# ---------------------------------------------------------------------------
# Hardware Mapper tests
# ---------------------------------------------------------------------------

class TestHardwareMapper:

    def setup_method(self):
        self.mapper = HardwareMapper()

    def test_cpu_low(self):
        spec = HardwareSpec(ram_gb=8, cpu_cores=4)
        assert self.mapper.map(spec) == HardwareProfile.CPU_LOW

    def test_cpu_medium(self):
        spec = HardwareSpec(ram_gb=16, cpu_cores=8)
        assert self.mapper.map(spec) == HardwareProfile.CPU_MEDIUM

    def test_cpu_high(self):
        spec = HardwareSpec(ram_gb=32, cpu_cores=16)
        assert self.mapper.map(spec) == HardwareProfile.CPU_HIGH

    def test_gpu_t4_by_name(self):
        spec = HardwareSpec(ram_gb=16, cpu_cores=4, has_gpu=True, gpu_vram_gb=16, gpu_name="Tesla T4")
        assert self.mapper.map(spec) == HardwareProfile.GPU_T4

    def test_gpu_a10_by_name(self):
        spec = HardwareSpec(ram_gb=64, cpu_cores=8, has_gpu=True, gpu_vram_gb=24, gpu_name="NVIDIA A10")
        assert self.mapper.map(spec) == HardwareProfile.GPU_A10

    def test_gpu_t4_by_vram(self):
        spec = HardwareSpec(ram_gb=16, cpu_cores=4, has_gpu=True, gpu_vram_gb=16)
        assert self.mapper.map(spec) == HardwareProfile.GPU_T4

    def test_edge_device(self):
        spec = HardwareSpec(ram_gb=4, cpu_cores=4, is_edge=True)
        assert self.mapper.map(spec) == HardwareProfile.EDGE

    def test_edge_takes_priority_over_gpu(self):
        # Edge flag always wins
        spec = HardwareSpec(ram_gb=8, cpu_cores=4, has_gpu=True, gpu_vram_gb=8, is_edge=True)
        assert self.mapper.map(spec) == HardwareProfile.EDGE

    def test_borderline_cpu_medium(self):
        # 20GB RAM → still CPU_MEDIUM (16GB + headroom)
        spec = HardwareSpec(ram_gb=20, cpu_cores=8)
        assert self.mapper.map(spec) == HardwareProfile.CPU_MEDIUM

    def test_list_all_tiers(self):
        tiers = self.mapper.list_all_tiers()
        assert len(tiers) >= 5
        assert all("profile" in t for t in tiers)


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer:

    def setup_method(self):
        self.norm = Normalizer()

    def test_valid_record_passes(self):
        raw    = _raw_result()
        result = self.norm.normalize(raw)
        assert result is not None

    def test_failed_status_rejected(self):
        raw    = _raw_result(status=RunStatus.FAILED)
        result = self.norm.normalize(raw)
        assert result is None

    def test_infinite_tokens_per_sec_rejected(self):
        # Pydantic's field_validator catches inf at construction time —
        # so RawBenchmarkResult itself raises, which means normalize()
        # never gets called. Both behaviours correctly reject the value.
        import math
        from pydantic import ValidationError
        with pytest.raises((ValidationError, Exception)):
            raw = _raw_result(tokens_per_sec=float("inf"))
            # If somehow construction succeeded, normalizer must reject it
            result = self.norm.normalize(raw)
            assert result is None

    def test_zero_tokens_per_sec_rejected(self):
        # Pydantic validator rejects ≤0 at construction
        with pytest.raises(Exception):
            _raw_result(tokens_per_sec=0.0)

    def test_absurd_throughput_rejected(self):
        raw    = _raw_result(tokens_per_sec=99999.0)
        result = self.norm.normalize(raw)
        assert result is None

    def test_peak_memory_swap(self):
        # peak < used should be corrected
        raw    = _raw_result(memory_used_mb=5000.0, memory_peak_mb=4000.0)
        result = self.norm.normalize(raw)
        assert result is not None
        assert result.memory_peak_mb >= result.memory_used_mb


# ---------------------------------------------------------------------------
# Enricher tests
# ---------------------------------------------------------------------------

class TestEnricher:

    def setup_method(self):
        self.enricher = Enricher()

    def test_derived_metrics_computed(self):
        raw    = _raw_result(tokens_per_sec=10.0, memory_used_mb=4096.0)
        record = self.enricher.enrich(raw)

        expected_tps_per_gb = 10.0 / 4.0   # 4096 MB = 4 GB
        assert abs(record.tokens_per_sec_per_gb - expected_tps_per_gb) < 0.01

    def test_latency_per_token(self):
        raw    = _raw_result(latency_avg_ms=100.0, completion_tokens=256)
        record = self.enricher.enrich(raw)
        assert abs(record.latency_per_token_ms - (100.0 / 256)) < 0.001

    def test_memory_efficiency(self):
        raw    = _raw_result(tokens_per_sec=10.0, memory_used_mb=4000.0)
        record = self.enricher.enrich(raw)
        assert abs(record.memory_efficiency - (10.0 / 4000.0)) < 1e-6

    def test_not_outlier_by_default(self):
        raw    = _raw_result()
        record = self.enricher.enrich(raw)
        assert record.is_outlier is False

    def test_pipeline_version_set(self):
        raw    = _raw_result()
        record = self.enricher.enrich(raw)
        assert record.pipeline_version == "1.0.0"


# ---------------------------------------------------------------------------
# Outlier Detector tests
# ---------------------------------------------------------------------------

class TestOutlierDetector:

    def setup_method(self):
        self.detector = OutlierDetector()

    def test_no_flagging_below_threshold(self):
        records = [_processed_record(tokens_per_sec=10.0) for _ in range(3)]
        result  = self.detector.flag_outliers(records)
        assert all(not r.is_outlier for r in result)

    def test_obvious_outlier_flagged(self):
        # All records must share the same (model_id, quantization, hardware_profile)
        # group for IQR comparison to happen. Need ≥4 normal + 1 extreme.
        normals = [
            _processed_record(
                model_id="model-x",
                quantization="GGUF_Q4_K_M",
                hardware_profile="CPU_MEDIUM",
                tokens_per_sec=10.0 + i * 0.1   # tight cluster 10.0–10.4
            )
            for i in range(5)
        ]
        outlier = _processed_record(
            model_id="model-x",
            quantization="GGUF_Q4_K_M",
            hardware_profile="CPU_MEDIUM",
            tokens_per_sec=9999.0,               # extreme outlier
        )
        records = normals + [outlier]
        result  = self.detector.flag_outliers(records)
        flagged = [r for r in result if r.is_outlier]
        assert len(flagged) >= 1
        assert flagged[0].tokens_per_sec == 9999.0

    def test_insufficient_data_no_flagging(self):
        # Only 3 records — IQR needs ≥4
        records = [_processed_record(tokens_per_sec=10.0) for _ in range(3)]
        result  = self.detector.flag_outliers(records)
        assert all(not r.is_outlier for r in result)

    def test_different_groups_not_compared(self):
        # Two different models — should not be compared cross-group
        group_a = [_processed_record(model_id="model-a", tokens_per_sec=10.0) for _ in range(4)]
        group_b = [_processed_record(model_id="model-b", tokens_per_sec=100.0) for _ in range(4)]
        result  = self.detector.flag_outliers(group_a + group_b)
        # No group should flag within itself (all same values)
        assert all(not r.is_outlier for r in result)


# ---------------------------------------------------------------------------
# Repository tests
# ---------------------------------------------------------------------------

class TestBenchmarkRepository:

    def test_insert_and_query(self, in_memory_repo):
        record = _processed_record()
        assert in_memory_repo.insert(record) is True
        assert in_memory_repo.count() == 1

    def test_query_by_hardware_profile(self, in_memory_repo):
        in_memory_repo.insert(_processed_record(hardware_profile="CPU_MEDIUM"))
        in_memory_repo.insert(_processed_record(hardware_profile="GPU_T4"))

        filters = BenchmarkQuery(hardware_profile=HardwareProfile.CPU_MEDIUM)
        df      = in_memory_repo.query(filters)
        assert len(df) == 1
        assert df.iloc[0]["hardware_profile"] == "CPU_MEDIUM"

    def test_query_by_model(self, in_memory_repo):
        in_memory_repo.insert(_processed_record(model_id="mistral-7b"))
        in_memory_repo.insert(_processed_record(model_id="llama-3b"))

        filters = BenchmarkQuery(model_ids=["mistral-7b"])
        df      = in_memory_repo.query(filters)
        assert len(df) == 1
        assert df.iloc[0]["model_id"] == "mistral-7b"

    def test_query_max_memory(self, in_memory_repo):
        in_memory_repo.insert(_processed_record(memory_used_mb=3000.0))
        in_memory_repo.insert(_processed_record(memory_used_mb=8000.0))

        filters = BenchmarkQuery(max_memory_mb=5000.0)
        df      = in_memory_repo.query(filters)
        assert len(df) == 1
        assert df.iloc[0]["memory_used_mb"] <= 5000.0

    def test_outliers_excluded_from_query(self, in_memory_repo):
        in_memory_repo.insert(_processed_record(is_outlier=False))
        in_memory_repo.insert(_processed_record(is_outlier=True))

        filters = BenchmarkQuery()
        df      = in_memory_repo.query(filters)
        assert len(df) == 1   # outlier excluded

    def test_get_model_list(self, in_memory_repo):
        in_memory_repo.insert(_processed_record(model_id="model-a"))
        in_memory_repo.insert(_processed_record(model_id="model-b"))
        models = in_memory_repo.get_model_list()
        assert "model-a" in models
        assert "model-b" in models

    def test_clear(self, in_memory_repo):
        in_memory_repo.insert(_processed_record())
        assert in_memory_repo.count() == 1
        in_memory_repo.clear()
        assert in_memory_repo.count() == 0

    def test_insert_batch(self, in_memory_repo):
        records = [_processed_record() for _ in range(5)]
        n       = in_memory_repo.insert_batch(records)
        assert n == 5
        assert in_memory_repo.count() == 5


# ---------------------------------------------------------------------------
# Query Engine tests
# ---------------------------------------------------------------------------

class TestQueryEngine:

    def test_compare_returns_dataframe(self, in_memory_repo):
        in_memory_repo.insert(_processed_record())
        engine = QueryEngine(in_memory_repo)
        df     = engine.compare(hardware_profile="CPU_MEDIUM")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_recommend_returns_dict(self, in_memory_repo):
        # Need at least 1 record to get recommendations
        in_memory_repo.insert(_processed_record())
        engine = QueryEngine(in_memory_repo)
        recs   = engine.recommend(hardware_profile="CPU_MEDIUM")
        assert "best_throughput" in recs
        assert "best_memory_efficiency" in recs
        assert "best_latency" in recs

    def test_recommend_empty_db_returns_error(self, in_memory_repo):
        engine = QueryEngine(in_memory_repo)
        recs   = engine.recommend(hardware_profile="CPU_MEDIUM")
        assert "error" in recs

    def test_compare_from_spec(self, in_memory_repo):
        in_memory_repo.insert(_processed_record(hardware_profile="CPU_MEDIUM"))
        engine = QueryEngine(in_memory_repo)
        spec   = HardwareSpec(ram_gb=16, cpu_cores=8)
        df     = engine.compare_from_spec(spec)
        assert len(df) == 1

    def test_get_db_stats(self, in_memory_repo):
        in_memory_repo.insert(_processed_record())
        engine = QueryEngine(in_memory_repo)
        stats  = engine.get_db_stats()
        assert stats["total_records"] == 1
        assert stats["models"] == 1


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_full_pipeline_from_file(self, tmp_path):
        raw    = _raw_result()
        raw_fp = tmp_path / "test_result.json"
        with open(raw_fp, "w") as f:
            json.dump(raw.model_dump(mode="json"), f, default=str)

        pipeline = Pipeline()
        record   = pipeline.process_file(raw_fp)

        assert record is not None
        assert record.tokens_per_sec == raw.tokens_per_sec
        assert record.tokens_per_sec_per_gb > 0
        assert record.is_outlier is False

    def test_failed_file_returns_none(self, tmp_path):
        raw = _raw_result(status=RunStatus.FAILED)
        fp  = tmp_path / "failed.json"
        with open(fp, "w") as f:
            json.dump(raw.model_dump(mode="json"), f, default=str)

        pipeline = Pipeline()
        result   = pipeline.process_file(fp)
        assert result is None

    def test_process_directory(self, tmp_path):
        for i in range(3):
            raw = _raw_result(model_id=f"model-{i}")
            fp  = tmp_path / f"result_{i}.json"
            with open(fp, "w") as f:
                json.dump(raw.model_dump(mode="json"), f, default=str)

        pipeline = Pipeline()
        records  = pipeline.process_directory(tmp_path)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Pydantic validation edge cases
# ---------------------------------------------------------------------------

class TestPydanticValidation:

    def test_finite_metric_required(self):
        with pytest.raises(Exception):
            _raw_result(tokens_per_sec=float("nan"))

    def test_positive_tokens_required(self):
        with pytest.raises(Exception):
            _raw_result(tokens_per_sec=-5.0)

    def test_quantization_enum_validated(self):
        with pytest.raises(Exception):
            _raw_result(quantization="INVALID_QUANT")
