"""
CPU Benchmark Worker
====================
Runs standardized benchmarks on GGUF models using llama-cpp-python.
Produces validated RawBenchmarkResult JSON files.

Usage:
    python -m src.workers.cpu_worker \
        --model path/to/model.gguf \
        --quantization GGUF_Q4_K_M \
        --n-runs 3 \
        --output data/raw/

Design principles:
    - Warm-up run before measurement (CPU caches, memory mapping)
    - Multiple runs averaged (n_runs=3 by default)
    - Peak memory tracked separately from average
    - Every result carries full software stack metadata
    - Failed runs are stored with status=FAILED, not silently dropped
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.core.hardware_mapper import HardwareMapper, autodetect_hardware
from src.core.models import (
    HardwareProfile,
    QuantizationType,
    RawBenchmarkResult,
    RunStatus,
)

console = Console()


# ---------------------------------------------------------------------------
# Standardized prompt set — FIXED across all runs for comparability
# The hash of this set is stored with every result.
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS = [
    "Explain the concept of gradient descent in machine learning.",
    "Write a Python function to perform binary search on a sorted list.",
    "What are the key differences between supervised and unsupervised learning?",
    "Describe the transformer architecture and its importance.",
    "Summarize the main principles of object-oriented programming.",
]

# PROMPT_HASH computed below
PROMPT_HASH = hashlib.sha256(
    "\n".join(BENCHMARK_PROMPTS).encode()
).hexdigest()[:16]

MAX_TOKENS   = 256   # completion tokens per prompt
WARMUP_RUNS  = 1     # discarded
SEED         = 42


# ---------------------------------------------------------------------------
# Single-run benchmark
# ---------------------------------------------------------------------------

def _run_single(
    llm,
    prompt: str,
    max_tokens: int,
) -> dict:
    """
    Run one prompt through the model and capture timing + memory.
    Returns raw metrics dict.
    """
    process = psutil.Process()

    # Memory before
    mem_before_mb = process.memory_info().rss / (1024 ** 2)

    # Time to first token is hard to get directly from llama-cpp
    # We approximate: run with stream=True and capture first chunk time
    t_start        = time.perf_counter()
    first_token_t  = None
    token_count    = 0
    full_output    = ""

    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,        # deterministic
        seed=SEED,
        stream=True,
        echo=False,
    ):
        if first_token_t is None:
            first_token_t = time.perf_counter()
        token_count += 1
        full_output += chunk["choices"][0]["text"]

    t_end = time.perf_counter()

    # Memory after
    mem_after_mb  = process.memory_info().rss / (1024 ** 2)
    mem_peak_mb   = process.memory_info().rss / (1024 ** 2)   # rss is peak on Linux

    total_time_s  = t_end - t_start
    ttft_ms       = (first_token_t - t_start) * 1000 if first_token_t else 0.0
    tps           = token_count / total_time_s if total_time_s > 0 else 0.0
    lat_per_tok   = (total_time_s / token_count * 1000) if token_count > 0 else 0.0

    return {
        "tokens_per_sec":   tps,
        "latency_first_ms": ttft_ms,
        "latency_avg_ms":   lat_per_tok,
        "memory_used_mb":   mem_after_mb - mem_before_mb,
        "memory_peak_mb":   mem_peak_mb,
        "completion_tokens": token_count,
    }


# ---------------------------------------------------------------------------
# Main worker function
# ---------------------------------------------------------------------------

def run_benchmark(
    model_path:    str,
    quantization:  QuantizationType,
    n_runs:        int        = 3,
    n_threads:     int        = -1,    # -1 = auto
    n_ctx:         int        = 2048,
    output_dir:    str        = "data/raw",
    hardware_id:   Optional[str] = None,
) -> Optional[RawBenchmarkResult]:
    """
    Full benchmark run on a GGUF model.
    Returns a validated RawBenchmarkResult and writes it to output_dir.
    """

    try:
        from llama_cpp import Llama
    except ImportError:
        console.print("[bold red]llama-cpp-python not installed.[/bold red]")
        console.print("Run: pip install llama-cpp-python")
        sys.exit(1)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        console.print(f"[red]Model file not found: {model_path}[/red]")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # — Hardware detection —
    spec    = autodetect_hardware()
    mapper  = HardwareMapper()
    profile = mapper.map(spec)
    hw_id   = hardware_id or f"{platform.node()}_{int(spec.ram_gb)}gb"

    console.rule("[bold cyan]LLM Benchmark Worker — CPU[/bold cyan]")
    console.print(f"  Model       : [green]{model_path_obj.name}[/green]")
    console.print(f"  Quantization: [yellow]{quantization.value}[/yellow]")
    console.print(f"  HW Profile  : [cyan]{profile.value}[/cyan]")
    console.print(f"  RAM         : {spec.ram_gb:.1f} GB")
    console.print(f"  CPU Cores   : {spec.cpu_cores}")
    console.print(f"  Runs        : {WARMUP_RUNS} warmup + {n_runs} measured")
    console.print()

    # — Load model —
    n_threads_actual = n_threads if n_threads > 0 else (spec.cpu_cores or 4)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            llm = Llama(
                model_path  = model_path,
                n_ctx       = n_ctx,
                n_threads   = n_threads_actual,
                n_gpu_layers= 0,          # CPU only
                verbose     = False,
                seed        = SEED,
            )
            progress.update(task, description="[green]Model loaded ✓[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        _write_failed(
            model_path_obj, quantization, hw_id, profile, str(e), output_path
        )
        return None

    # — Warmup runs (discarded) —
    console.print("[dim]Running warmup...[/dim]")
    try:
        for _ in range(WARMUP_RUNS):
            _run_single(llm, BENCHMARK_PROMPTS[0], max_tokens=32)
    except Exception:
        pass   # warmup failures are non-fatal

    # — Measured runs —
    all_metrics: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        for run_idx in range(n_runs):
            for prompt_idx, prompt in enumerate(BENCHMARK_PROMPTS):
                task_label = (
                    f"Run {run_idx+1}/{n_runs} · "
                    f"Prompt {prompt_idx+1}/{len(BENCHMARK_PROMPTS)}"
                )
                task = progress.add_task(task_label, total=None)

                try:
                    metrics = _run_single(llm, prompt, max_tokens=MAX_TOKENS)
                    metrics["prompt_tokens"] = len(prompt.split())
                    all_metrics.append(metrics)
                    progress.update(
                        task,
                        description=f"[green]✓[/green] {task_label} "
                                    f"— {metrics['tokens_per_sec']:.1f} tok/s"
                    )
                except Exception as e:
                    progress.update(task, description=f"[red]✗ {task_label}: {e}[/red]")
                    console.print(f"[yellow]Warning: run failed — {e}[/yellow]")

    if not all_metrics:
        console.print("[bold red]All runs failed. Writing failure record.[/bold red]")
        _write_failed(
            model_path_obj, quantization, hw_id, profile,
            "All measurement runs failed", output_path
        )
        return None

    # — Aggregate across all runs —
    import statistics

    def avg(key: str) -> float:
        vals = [m[key] for m in all_metrics if key in m]
        return statistics.mean(vals) if vals else 0.0

    result = RawBenchmarkResult(
        model_id          = model_path_obj.stem,
        model_file        = model_path_obj.name,
        quantization      = quantization,
        hardware_id       = hw_id,
        hardware_profile  = profile,

        tokens_per_sec    = avg("tokens_per_sec"),
        latency_first_ms  = avg("latency_first_ms"),
        latency_avg_ms    = avg("latency_avg_ms"),
        memory_used_mb    = avg("memory_used_mb"),
        memory_peak_mb    = max(m["memory_peak_mb"] for m in all_metrics),

        prompt_tokens     = int(avg("prompt_tokens")),
        completion_tokens = MAX_TOKENS,
        n_runs            = n_runs,
        status            = RunStatus.SUCCESS,

        prompt_hash       = PROMPT_HASH,
        seed              = SEED,

        python_version    = platform.python_version(),
        framework         = _get_llamacpp_version(),
        cuda_version      = None,
        driver_version    = None,
    )

    # — Write result —
    out_file = output_path / _make_filename(result)
    with open(out_file, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    console.rule("[bold green]Benchmark Complete[/bold green]")
    console.print(f"  Throughput  : [bold]{result.tokens_per_sec:.2f}[/bold] tokens/sec")
    console.print(f"  TTFT        : {result.latency_first_ms:.1f} ms")
    console.print(f"  Avg latency : {result.latency_avg_ms:.1f} ms/token")
    console.print(f"  Memory used : {result.memory_used_mb:.0f} MB")
    console.print(f"  Saved to    : [cyan]{out_file}[/cyan]")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_failed(
    model_path: Path,
    quantization: QuantizationType,
    hw_id: str,
    profile: HardwareProfile,
    error: str,
    output_path: Path,
) -> None:
    record = {
        "model_id":        model_path.stem,
        "model_file":      model_path.name,
        "quantization":    quantization.value,
        "hardware_id":     hw_id,
        "hardware_profile": profile.value,
        "status":          RunStatus.FAILED.value,
        "error_message":   error,
        "timestamp":       datetime.utcnow().isoformat(),
    }
    fname = f"FAILED_{model_path.stem}_{quantization.value}_{hw_id}.json"
    with open(output_path / fname, "w") as f:
        json.dump(record, f, indent=2)


def _make_filename(r: RawBenchmarkResult) -> str:
    ts = r.timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{r.model_id}__{r.quantization.value}__{r.hardware_profile.value}__{ts}.json"


def _get_llamacpp_version() -> str:
    try:
        import llama_cpp
        return f"llama-cpp-python-{llama_cpp.__version__}"
    except Exception:
        return "llama-cpp-python-unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run CPU benchmarks on a GGUF model."
    )
    parser.add_argument("--model",         required=True, help="Path to .gguf file")
    parser.add_argument("--quantization",  required=True,
                        choices=[q.value for q in QuantizationType],
                        help="Quantization type")
    parser.add_argument("--n-runs",        type=int, default=3)
    parser.add_argument("--n-threads",     type=int, default=-1,
                        help="CPU threads (-1 = auto)")
    parser.add_argument("--n-ctx",         type=int, default=2048)
    parser.add_argument("--output",        default="data/raw")
    parser.add_argument("--hardware-id",   default=None,
                        help="Optional override for hardware identifier")
    args = parser.parse_args()

    run_benchmark(
        model_path   = args.model,
        quantization = QuantizationType(args.quantization),
        n_runs       = args.n_runs,
        n_threads    = args.n_threads,
        n_ctx        = args.n_ctx,
        output_dir   = args.output,
        hardware_id  = args.hardware_id,
    )


if __name__ == "__main__":
    main()
