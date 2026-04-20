---
title: LLM Benchmark Platform
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🔬 LLM Benchmark Platform

> **A data platform for making informed LLM deployment decisions on budget hardware.**
> Compare quantized model performance across real CPU and GPU environments — before you spend hours downloading the wrong model.

[![CI](https://github.com/Saraid10/llm-benchmark-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Saraid10/llm-benchmark-platform/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/Saraid10/llm-benchmark-platform)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![DuckDB](https://img.shields.io/badge/database-DuckDB-yellow)](https://duckdb.org)

---

## The Problem

Most LLM benchmarks are run on A100s or H100s. Developers deploying on laptops, free Colab T4s, or edge devices get almost no useful guidance. The result: hours wasted downloading models that won't fit in RAM, or run too slowly to be usable.

This platform answers: **"Which quantized model should I actually run on my hardware?"**

---

## Live Demo

👉 **[huggingface.co/spaces/Saraid10/llm-benchmark-platform](https://huggingface.co/spaces/Saraid10/llm-benchmark-platform)**

Select your hardware → click Compare → get data-driven recommendations instantly.

---

## Architecture

```
CPU Worker (llama-cpp-python)    GPU Worker (autoawq, Colab T4)
         │  Raw JSON                      │  Raw JSON
         └──────────────┬─────────────────┘
                        ▼
         ┌──────────────────────────┐
         │  Processing Pipeline     │
         │  Normalize · IQR Outlier │
         │  Detection · Enrichment  │
         └──────────────┬───────────┘
                        ▼
         ┌──────────────────────────┐
         │  DuckDB (OLAP, in-proc) │
         └──────────────┬───────────┘
                        ▼
         ┌──────────────────────────┐
         │  Query Engine +          │
         │  Hardware Abstraction    │
         └──────────────┬───────────┘
                        ▼
         ┌──────────────────────────┐
         │  Gradio Dashboard        │
         │  HuggingFace Spaces      │
         └──────────────────────────┘
```

**Key design decisions:**
- **Precomputed benchmarks** — reproducible, versionable, instantly queryable
- **DuckDB** — columnar OLAP, zero infrastructure, 10-100× faster than SQLite for analytical queries
- **Hardware abstraction tiers** — maps raw specs to comparable tiers (CPU_LOW / CPU_MEDIUM / GPU_T4 etc.)
- **IQR outlier detection** — noisy runs don't pollute averages

---

## Benchmark Methodology

| Parameter | Value |
|---|---|
| Prompt set | 5 standardized prompts · SHA256: `e3efec0e5fcd0b22` |
| Warmup runs | 1 discarded |
| Measured runs | 3 averaged |
| Max tokens | 256 completion tokens |
| Temperature | 0.0 · Seed: 42 |
| Outlier detection | IQR method (1.5×) per (model, quant, hw) group |

**Hardware tiers:**

| Tier | RAM/VRAM | Example |
|---|---|---|
| `CPU_LOW` | ≤ 8 GB | Budget laptop |
| `CPU_MEDIUM` | 16 GB | Dev machine |
| `CPU_HIGH` | 32+ GB | Workstation |
| `GPU_T4` | 16 GB VRAM | Google Colab free |
| `GPU_A10` | 24 GB VRAM | Colab Pro+ |

---

## Data Provenance

| Field | Values |
|---|---|
| `data_source` | `seed` · `real_cpu` · `real_colab_t4` |
| Dataset version | `2026.04.20.2` (see `data/versions/`) |
| Prompt hash | `e3efec0e5fcd0b22` (consistent across all workers, app, and DB) |

Sample raw benchmark JSONs: [`data/examples/`](data/examples/)

---

## Limitations

- **Seed data is synthetic baseline** — 25 seed rows are controlled estimates. Real runs (`data_source = real_*`) are ground truth.
- **Hardware variance** — CPU performance varies by core count, memory speed, and thermal state.
- **Benchmark scope** — prompts are general-purpose; code or long-context tasks may rank models differently.
- **AWQ speed** — current results use `fuse_layers=False` (autoawq ≥ 0.2.7 compatibility). Real fused AWQ is 3–5× faster.
- **Validate on your workload** — use this to narrow candidates, then test your specific use case.

---

## Quickstart

```bash
git clone https://github.com/Saraid10/llm-benchmark-platform.git
cd llm-benchmark-platform
pip install -r requirements-dev.txt

python scripts/load_seed_data.py    # load seed data
python -m pytest tests/ -v          # 44 tests
python app.py                       # UI at localhost:7860
```

## Run Your Own Benchmarks

**CPU:**
```bash
pip install -r requirements-worker-cpu.txt
python -m src.workers.cpu_worker \
    --model models/phi-3-mini.Q4_K_M.gguf \
    --quantization GGUF_Q4_K_M --n-runs 3
python scripts/load_seed_data.py --raw-dir data/raw/
```

**GPU (Colab T4):** Open [`notebooks/GPU_Benchmark_Colab.ipynb`](notebooks/GPU_Benchmark_Colab.ipynb) with a T4 runtime.

---

## Project Structure

```
├── app.py                          # HF Spaces entrypoint
├── Dockerfile
├── data/
│   ├── seed_benchmarks.csv
│   ├── examples/                   # Sample raw benchmark JSONs
│   └── versions/                   # Dataset version manifests
├── src/
│   ├── core/                       # models.py · hardware_mapper.py · query_engine.py
│   ├── processing/                 # pipeline.py (normalize → outlier → enrich)
│   ├── workers/                    # cpu_worker.py · gpu_worker.py
│   ├── db/                         # repository.py (DuckDB)
│   ├── api/                        # server.py (FastAPI ingestion)
│   ├── versioning/                 # versioning.py
│   └── monitoring/                 # monitor.py (data quality)
├── scripts/
├── notebooks/
└── tests/                          # 44 tests across all layers
```

*Built to help developers make informed LLM deployment decisions on real, budget hardware.*
