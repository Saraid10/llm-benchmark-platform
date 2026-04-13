---
title: LLM Benchmark Platform
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/Saraid10/llm-benchmark-platform)
# 🔬 LLM Benchmark Platform

> A data platform for comparing quantized LLM inference performance across real hardware.
> GGUF · GPTQ · AWQ · CPU · GPU · Edge

[![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/llm-benchmark-platform)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![DuckDB](https://img.shields.io/badge/database-DuckDB-yellow)](https://duckdb.org)

---

## What This Is

Most LLM benchmarks are run on A100s or H100s. This platform focuses on the hardware most developers actually use: laptops, Colab T4s, and edge devices. All benchmark data is collected from controlled runs using a standardized methodology.

**Key design decisions:**
- Precomputed data (reproducible, versionable) over live benchmarking
- OLAP-optimized DuckDB for fast analytical queries
- Hardware abstraction layer maps raw specs → capability tiers
- Processing pipeline with IQR-based outlier detection
- Clean separation: workers → pipeline → DB → query engine → UI

## Architecture

```
Benchmark Workers (CPU / T4 / Edge)
         │
         ▼
Raw JSON files (data/raw/)
         │
         ▼
Processing Pipeline (normalize → outlier detect → enrich)
         │
         ▼
DuckDB (columnar OLAP, single file, zero infra)
         │
         ▼
Query Engine (business logic, recommendations)
         │
         ▼
Gradio UI (pure presentation, HF Spaces)
```

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-benchmark-platform
cd llm-benchmark-platform
pip install -r requirements.txt

# Load seed data
python scripts/load_seed_data.py

# Launch UI
python src/ui/app.py
```

## Running Your Own Benchmarks

### CPU (your laptop)
```bash
# Download a GGUF model first, e.g. from HuggingFace
python -m src.workers.cpu_worker \
    --model models/mistral-7b-v0.1.Q4_K_M.gguf \
    --quantization GGUF_Q4_K_M \
    --n-runs 3 \
    --output data/raw/

# Load results into DB
python scripts/load_seed_data.py --raw-dir data/raw/
```

### GPU (Google Colab T4)
Open `notebooks/GPU_Benchmark_Colab.ipynb` in Colab with a T4 runtime.
Results are downloaded as JSON and loaded with the same script above.

## Benchmark Methodology

| Parameter | Value |
|---|---|
| Prompt set | 5 standardized prompts (SHA256: `a1b2c3d4...`) |
| Warm-up runs | 1 (discarded) |
| Measured runs | 3 (averaged) |
| Max tokens | 256 completion tokens |
| Temperature | 0.0 (deterministic) |
| Seed | 42 |

**Metrics collected:**
- `tokens_per_sec` — completion tokens ÷ total generation time
- `latency_first_ms` — time to first token (TTFT)
- `latency_avg_ms` — average per-token latency
- `memory_used_mb` — RSS delta during generation
- `memory_peak_mb` — peak RSS during generation

**Derived metrics:**
- `tokens_per_sec_per_gb` — throughput efficiency (higher = better)
- `memory_efficiency` — tokens/sec ÷ memory_used_mb
- `latency_per_token_ms` — normalized latency

**Outlier detection:** IQR method (1.5×IQR) per (model, quantization, hardware) group.

## Hardware Tiers

| Tier | Definition | Example |
|---|---|---|
| `CPU_LOW` | ≤8 GB RAM | Budget laptop |
| `CPU_MEDIUM` | 16 GB RAM | Standard dev laptop |
| `CPU_HIGH` | 32+ GB RAM | Workstation |
| `GPU_T4` | 16 GB VRAM | Google Colab free |
| `GPU_A10` | 24 GB VRAM | Colab Pro+ |
| `EDGE` | Jetson / ARM | Jetson Nano |

## Project Structure

```
llm-benchmark-platform/
├── app.py                          # HF Spaces entrypoint
├── data/
│   ├── seed_benchmarks.csv         # Curated seed dataset
│   └── benchmarks.duckdb           # Generated on first run
├── src/
│   ├── core/
│   │   ├── models.py               # Pydantic schemas
│   │   ├── hardware_mapper.py      # Hardware abstraction layer
│   │   └── query_engine.py         # Business logic
│   ├── processing/
│   │   └── pipeline.py             # Normalize → outlier detect → enrich
│   ├── workers/
│   │   ├── cpu_worker.py           # GGUF benchmarks (laptop)
│   │   └── gpu_worker.py           # GPTQ/AWQ benchmarks (Colab)
│   ├── db/
│   │   └── repository.py           # DuckDB operations
│   └── ui/
│       └── app.py                  # Gradio frontend
├── scripts/
│   └── load_seed_data.py           # Data ingestion script
└── notebooks/
    └── GPU_Benchmark_Colab.ipynb   # Colab GPU worker
```

## Dataset

The seed dataset covers:
- **Models:** Mistral-7B, Llama-3.2-3B, Phi-3-mini, Gemma-2B
- **Quantizations:** GGUF Q4_K_M, Q4_0, Q5_K_M, Q8_0 · GPTQ 4bit/8bit · AWQ 4bit
- **Hardware:** CPU_LOW, CPU_MEDIUM, CPU_HIGH, GPU_T4

All data points include full software stack metadata (framework version, CUDA version) for reproducibility.

---

*Built to help developers make informed LLM deployment decisions on real hardware.*
