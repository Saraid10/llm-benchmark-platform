"""
Ingestion API
=============
REST API that accepts benchmark results from workers.
Built with FastAPI — fast, typed, auto-documented at /docs.

Endpoints:
    POST /benchmark          — submit a single result
    POST /benchmark/batch    — submit multiple results
    GET  /status             — health check + DB stats
    GET  /versions           — version history
    GET  /models             — available models
    GET  /hardware-profiles  — available hardware tiers

Authentication:
    API key via X-API-Key header.
    Set API_KEY env variable. Default "dev" for local use.

Usage:
    uvicorn src.api.server:app --reload --port 8000

    # Submit a result
    curl -X POST http://localhost:8000/benchmark \\
         -H "X-API-Key: dev" \\
         -H "Content-Type: application/json" \\
         -d @data/raw/my_result.json
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.models import RawBenchmarkResult, RunStatus
from src.db.repository import BenchmarkRepository
from src.processing.pipeline import Pipeline
from src.versioning.versioning import VersionRegistry

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "LLM Benchmark Platform API",
    description = "Ingestion and query API for quantized LLM benchmark data.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Shared instances (initialized on startup)
repo     : Optional[BenchmarkRepository] = None
pipeline : Optional[Pipeline]            = None
registry : Optional[VersionRegistry]     = None

API_KEY = os.getenv("BENCHMARK_API_KEY", "dev")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global repo, pipeline, registry
    repo     = BenchmarkRepository()
    pipeline = Pipeline()
    registry = VersionRegistry()
    print(f"[API] Started. DB has {repo.count()} records.")


@app.on_event("shutdown")
async def shutdown():
    if repo:
        repo.close()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid API key.",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    success:      bool
    message:      str
    record_id:    Optional[str] = None
    version_id:   Optional[str] = None


class BatchIngestResponse(BaseModel):
    total:    int
    accepted: int
    rejected: int
    errors:   list[str]
    version_id: Optional[str] = None


class StatusResponse(BaseModel):
    status:          str
    total_records:   int
    models:          int
    hardware_profiles: int
    quantizations:   int
    current_version: Optional[str]
    uptime_since:    str


class SubmitBenchmarkPayload(BaseModel):
    """
    What a worker POSTs to /benchmark.
    Identical to RawBenchmarkResult — re-validated here.
    """
    model_id:          str
    model_file:        str
    quantization:      str
    hardware_id:       str
    hardware_profile:  str
    tokens_per_sec:    float
    latency_first_ms:  float
    latency_avg_ms:    float
    memory_used_mb:    float
    memory_peak_mb:    float
    prompt_tokens:     int
    completion_tokens: int
    n_runs:            int
    status:            str
    prompt_hash:       str
    seed:              int
    python_version:    str
    framework:         str
    cuda_version:      Optional[str] = None
    driver_version:    Optional[str] = None
    error_message:     Optional[str] = None
    changelog:         str = "Automated worker submission"


_startup_time = datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def get_status():
    """Health check and database statistics."""
    current = registry.get_current()
    return StatusResponse(
        status            = "ok",
        total_records     = repo.count(),
        models            = len(repo.get_model_list()),
        hardware_profiles = len(repo.get_hardware_profiles()),
        quantizations     = len(repo.get_quantization_types()),
        current_version   = current.version_id if current else None,
        uptime_since      = _startup_time,
    )


@app.post("/benchmark", response_model=IngestResponse, tags=["Ingest"])
async def submit_benchmark(
    payload: SubmitBenchmarkPayload,
    _key:    str = Depends(verify_api_key),
):
    """
    Submit a single benchmark result.
    The result is validated, processed, inserted, and a new version is created.
    """
    # Reconstruct as RawBenchmarkResult for pipeline processing
    try:
        from src.core.models import QuantizationType, HardwareProfile
        raw = RawBenchmarkResult(
            model_id          = payload.model_id,
            model_file        = payload.model_file,
            quantization      = QuantizationType(payload.quantization),
            hardware_id       = payload.hardware_id,
            hardware_profile  = HardwareProfile(payload.hardware_profile),
            tokens_per_sec    = payload.tokens_per_sec,
            latency_first_ms  = payload.latency_first_ms,
            latency_avg_ms    = payload.latency_avg_ms,
            memory_used_mb    = payload.memory_used_mb,
            memory_peak_mb    = payload.memory_peak_mb,
            prompt_tokens     = payload.prompt_tokens,
            completion_tokens = payload.completion_tokens,
            n_runs            = payload.n_runs,
            status            = RunStatus(payload.status),
            prompt_hash       = payload.prompt_hash,
            seed              = payload.seed,
            python_version    = payload.python_version,
            framework         = payload.framework,
            cuda_version      = payload.cuda_version,
            driver_version    = payload.driver_version,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    # Process through pipeline
    normalized = pipeline.normalizer.normalize(raw)
    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail="Record failed normalization (sanity bounds check). "
                   "Check metric values are physically plausible."
        )

    processed = pipeline.enricher.enrich(normalized)
    success   = repo.insert(processed)

    if not success:
        raise HTTPException(status_code=500, detail="Database insert failed.")

    # Create new version
    all_models = repo.get_model_list()
    all_hw     = repo.get_hardware_profiles()
    all_quants = repo.get_quantization_types()

    record_id  = f"{processed.model_id}_{processed.quantization}_{processed.hardware_profile}_{processed.timestamp}"
    version    = registry.create_version(
        record_count      = repo.count(),
        model_ids         = all_models,
        hardware_profiles = all_hw,
        quantizations     = all_quants,
        changelog         = payload.changelog,
        prompt_hash       = processed.prompt_hash,
        record_ids        = [record_id],
    )

    return IngestResponse(
        success    = True,
        message    = f"Record accepted and processed.",
        record_id  = record_id,
        version_id = version.version_id,
    )


@app.post("/benchmark/batch", response_model=BatchIngestResponse, tags=["Ingest"])
async def submit_benchmark_batch(
    payloads: list[SubmitBenchmarkPayload],
    _key:     str = Depends(verify_api_key),
):
    """
    Submit multiple benchmark results in one request.
    Partial success is supported — valid records are inserted even if some fail.
    """
    accepted = 0
    rejected = 0
    errors   = []

    for i, payload in enumerate(payloads):
        try:
            # Reuse single-submit logic inline
            from src.core.models import QuantizationType, HardwareProfile
            raw = RawBenchmarkResult(
                model_id          = payload.model_id,
                model_file        = payload.model_file,
                quantization      = QuantizationType(payload.quantization),
                hardware_id       = payload.hardware_id,
                hardware_profile  = HardwareProfile(payload.hardware_profile),
                tokens_per_sec    = payload.tokens_per_sec,
                latency_first_ms  = payload.latency_first_ms,
                latency_avg_ms    = payload.latency_avg_ms,
                memory_used_mb    = payload.memory_used_mb,
                memory_peak_mb    = payload.memory_peak_mb,
                prompt_tokens     = payload.prompt_tokens,
                completion_tokens = payload.completion_tokens,
                n_runs            = payload.n_runs,
                status            = RunStatus(payload.status),
                prompt_hash       = payload.prompt_hash,
                seed              = payload.seed,
                python_version    = payload.python_version,
                framework         = payload.framework,
                cuda_version      = payload.cuda_version,
                driver_version    = payload.driver_version,
            )
            normalized = pipeline.normalizer.normalize(raw)
            if normalized is None:
                raise ValueError("Failed normalization sanity check")

            processed = pipeline.enricher.enrich(normalized)
            if repo.insert(processed):
                accepted += 1
            else:
                rejected += 1
                errors.append(f"Record {i}: DB insert failed")

        except Exception as e:
            rejected += 1
            errors.append(f"Record {i} ({payload.model_id}): {str(e)[:100]}")

    # Create one version for the whole batch
    version_id = None
    if accepted > 0:
        version = registry.create_version(
            record_count      = repo.count(),
            model_ids         = repo.get_model_list(),
            hardware_profiles = repo.get_hardware_profiles(),
            quantizations     = repo.get_quantization_types(),
            changelog         = f"Batch ingest: {accepted} records added",
            prompt_hash       = payloads[0].prompt_hash if payloads else "",
            record_ids        = [str(i) for i in range(accepted)],
        )
        version_id = version.version_id

    return BatchIngestResponse(
        total      = len(payloads),
        accepted   = accepted,
        rejected   = rejected,
        errors     = errors,
        version_id = version_id,
    )


@app.get("/versions", tags=["Metadata"])
async def list_versions():
    """List all dataset versions, newest first."""
    versions = registry.list_versions()
    return {
        "versions": [v.model_dump(mode="json") for v in versions],
        "total":    len(versions),
    }


@app.get("/versions/current", tags=["Metadata"])
async def get_current_version():
    """Get the current (latest) dataset version."""
    v = registry.get_current()
    if not v:
        return {"version": None, "message": "No versions yet."}
    return v.model_dump(mode="json")


@app.get("/models", tags=["Metadata"])
async def list_models():
    return {"models": repo.get_model_list()}


@app.get("/hardware-profiles", tags=["Metadata"])
async def list_hardware_profiles():
    return {"hardware_profiles": repo.get_hardware_profiles()}


@app.get("/quantizations", tags=["Metadata"])
async def list_quantizations():
    return {"quantizations": repo.get_quantization_types()}
