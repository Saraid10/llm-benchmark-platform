"""
Core data models for the LLM Benchmark Platform.
Every benchmark result, hardware profile, and processed metric
flows through these schemas. Pydantic ensures strict validation
at ingestion time — garbage in, garbage out is not acceptable.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations — strict vocabulary for the entire system
# ---------------------------------------------------------------------------

class QuantizationType(str, Enum):
    GGUF_Q4_K_M  = "GGUF_Q4_K_M"
    GGUF_Q4_0    = "GGUF_Q4_0"
    GGUF_Q5_K_M  = "GGUF_Q5_K_M"
    GGUF_Q8_0    = "GGUF_Q8_0"
    GPTQ_4BIT    = "GPTQ_4BIT"
    GPTQ_8BIT    = "GPTQ_8BIT"
    AWQ_4BIT     = "AWQ_4BIT"
    FP16         = "FP16"


class HardwareProfile(str, Enum):
    CPU_LOW     = "CPU_LOW"      # ≤8 GB RAM
    CPU_MEDIUM  = "CPU_MEDIUM"   # 16 GB RAM
    CPU_HIGH    = "CPU_HIGH"     # 32+ GB RAM
    GPU_T4      = "GPU_T4"       # 16 GB VRAM (Colab)
    GPU_A10     = "GPU_A10"      # 24 GB VRAM
    EDGE        = "EDGE"         # Jetson class


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Raw benchmark result — output of a worker run
# ---------------------------------------------------------------------------

class RawBenchmarkResult(BaseModel):
    """
    Produced by a worker (cpu_worker.py / gpu_worker.py).
    Contains everything needed to reproduce and validate the run.
    """
    model_id:           str               = Field(..., description="e.g. mistral-7b")
    model_file:         str               = Field(..., description="exact filename / HF repo")
    quantization:       QuantizationType
    hardware_id:        str               = Field(..., description="e.g. laptop_i7_16gb")
    hardware_profile:   HardwareProfile

    # — Primary metrics —
    tokens_per_sec:     float             = Field(..., gt=0)
    latency_first_ms:   float             = Field(..., gt=0,  description="Time-to-first-token (ms)")
    latency_avg_ms:     float             = Field(..., gt=0,  description="Avg per-token latency (ms)")
    memory_used_mb:     float             = Field(..., gt=0)
    memory_peak_mb:     float             = Field(..., gt=0)

    # — Run metadata —
    prompt_tokens:      int               = Field(..., gt=0)
    completion_tokens:  int               = Field(..., gt=0)
    n_runs:             int               = Field(default=3, description="Number of warm runs averaged")
    status:             RunStatus         = RunStatus.SUCCESS
    error_message:      Optional[str]     = None

    # — Reproducibility —
    timestamp:          datetime          = Field(default_factory=datetime.utcnow)
    prompt_hash:        str               = Field(..., description="SHA256 of prompt set used")
    seed:               int               = Field(default=42)

    # — Software stack —
    python_version:     str               = ""
    framework:          str               = ""          # e.g. "llama-cpp-python-0.2.57"
    cuda_version:       Optional[str]     = None
    driver_version:     Optional[str]     = None

    # — Pipeline metadata —
    is_outlier:         bool  = False
    pipeline_version:   str   = "1.0.0"
    data_source:        str   = "real"  # seed | real_cpu | real_colab_t4

    @field_validator("tokens_per_sec", "latency_first_ms", "latency_avg_ms")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Metric must be a finite number")
        return round(v, 4)


# ---------------------------------------------------------------------------
# Processed / enriched record — what goes into DuckDB
# ---------------------------------------------------------------------------

class ProcessedBenchmarkRecord(BaseModel):
    """
    After normalization and derived-metric computation.
    This is the schema stored in the database.
    """
    # — Identity (same as raw) —
    model_id:           str
    model_file:         str
    quantization:       str
    hardware_id:        str
    hardware_profile:   str

    # — Normalized primary metrics —
    tokens_per_sec:     float
    latency_first_ms:   float
    latency_avg_ms:     float
    memory_used_mb:     float
    memory_peak_mb:     float

    # — Derived metrics (computed by pipeline) —
    tokens_per_sec_per_gb:  float    # throughput efficiency
    memory_efficiency:      float    # tokens/sec ÷ memory_used_mb
    latency_per_token_ms:   float    # latency_avg_ms ÷ completion_tokens

    # — Run metadata —
    prompt_tokens:      int
    completion_tokens:  int
    n_runs:             int
    status:             str

    # — Reproducibility —
    timestamp:          datetime
    prompt_hash:        str
    seed:               int

    # — Software stack —
    python_version:     str
    framework:          str
    cuda_version:       Optional[str]
    driver_version:     Optional[str]

    # — Pipeline metadata —
    is_outlier:         bool  = False
    pipeline_version:   str   = "1.0.0"
    data_source:        str   = "real"  # seed | real_cpu | real_colab_t4


# ---------------------------------------------------------------------------
# Hardware spec — input to HardwareMapper
# ---------------------------------------------------------------------------

class HardwareSpec(BaseModel):
    """
    Raw hardware description from the user or auto-detected.
    """
    ram_gb:         float
    cpu_cores:      int
    has_gpu:        bool            = False
    gpu_vram_gb:    Optional[float] = None
    gpu_name:       Optional[str]   = None
    is_edge:        bool            = False


# ---------------------------------------------------------------------------
# Query filters — what the UI sends to the query engine
# ---------------------------------------------------------------------------

class BenchmarkQuery(BaseModel):
    hardware_profile:   Optional[HardwareProfile]     = None
    quantization_types: Optional[list[QuantizationType]] = None
    model_ids:          Optional[list[str]]           = None
    min_tokens_per_sec: Optional[float]               = None
    max_memory_mb:      Optional[float]               = None
    limit:              int                           = 100
