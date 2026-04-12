"""
GPU Benchmark Worker (Google Colab T4)
=======================================
Benchmarks GPTQ and AWQ quantized models using HuggingFace Transformers.
Designed to run in Google Colab with a T4 GPU (16 GB VRAM).

Usage in Colab:
    !git clone <your-repo>
    %cd llm-benchmark-platform
    !pip install -r requirements_colab.txt

    from src.workers.gpu_worker import run_benchmark
    from src.core.models import QuantizationType

    run_benchmark(
        model_id     = "TheBloke/Mistral-7B-v0.1-GPTQ",
        quantization = QuantizationType.GPTQ_4BIT,
        output_dir   = "data/raw",
    )

Supported quantizations:
    - GPTQ_4BIT, GPTQ_8BIT  → via auto-gptq + transformers
    - AWQ_4BIT               → via autoawq
    - FP16                   → baseline (requires full VRAM)
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics

# Rich is always available
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

# — Same standardized prompts as CPU worker (CRITICAL for comparability) —
BENCHMARK_PROMPTS = [
    "Explain the concept of gradient descent in machine learning.",
    "Write a Python function to perform binary search on a sorted list.",
    "What are the key differences between supervised and unsupervised learning?",
    "Describe the transformer architecture and why attention mechanisms are important.",
    "Summarize the main principles of object-oriented programming.",
]

PROMPT_HASH = hashlib.sha256(
    "\n".join(BENCHMARK_PROMPTS).encode()
).hexdigest()[:16]

MAX_TOKENS  = 256
WARMUP_RUNS = 1
SEED        = 42


# ---------------------------------------------------------------------------
# Model loader — handles GPTQ, AWQ, FP16
# ---------------------------------------------------------------------------

def _load_model(model_id: str, quantization: QuantizationType):
    """
    Loads the appropriate model + tokenizer based on quantization type.
    Returns (model, tokenizer, device).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"  Device: [cyan]{device}[/cyan]")

    if device == "cpu":
        console.print("[yellow]Warning: No GPU detected. Results will be CPU-speed.[/yellow]")

    if quantization in (QuantizationType.GPTQ_4BIT, QuantizationType.GPTQ_8BIT):
        return _load_gptq(model_id, device)

    elif quantization == QuantizationType.AWQ_4BIT:
        return _load_awq(model_id, device)

    elif quantization == QuantizationType.FP16:
        return _load_fp16(model_id, device)

    else:
        raise ValueError(f"GPU worker does not support: {quantization}")


def _load_gptq(model_id: str, device: str):
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        use_safetensors=True,
        trust_remote_code=False,
        device=device,
        quantize_config=None,
    )
    return model, tokenizer, device


def _load_awq(model_id: str, device: str):
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoAWQForCausalLM.from_quantized(
        model_id,
        fuse_layers=False,       # fuse_layers=True crashes in autoawq>=0.2.7
        trust_remote_code=False,
    )
    model = model.to(device)
    return model, tokenizer, device


def _load_fp16(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Single-run benchmark (GPU)
# ---------------------------------------------------------------------------

def _run_single_gpu(model, tokenizer, prompt: str, device: str) -> dict:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_token_count = inputs["input_ids"].shape[1]

    # GPU memory before
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    t_start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,         # deterministic
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    t_end = time.perf_counter()

    # GPU memory after
    if device == "cuda":
        mem_after   = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak    = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        import psutil
        rss = psutil.Process().memory_info().rss / (1024 ** 2)
        mem_after = mem_peak = rss

    completion_tokens = outputs.shape[1] - prompt_token_count
    total_time_s      = t_end - t_start
    tps               = completion_tokens / total_time_s if total_time_s > 0 else 0.0
    lat_avg_ms        = (total_time_s / completion_tokens * 1000) if completion_tokens > 0 else 0.0

    # TTFT approximation: run a single token generation
    t_ttft_start = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    ttft_ms = (time.perf_counter() - t_ttft_start) * 1000

    return {
        "tokens_per_sec":    tps,
        "latency_first_ms":  ttft_ms,
        "latency_avg_ms":    lat_avg_ms,
        "memory_used_mb":    mem_after,
        "memory_peak_mb":    mem_peak,
        "completion_tokens": completion_tokens,
        "prompt_tokens":     prompt_token_count,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    model_id:    str,
    quantization: QuantizationType,
    n_runs:      int  = 3,
    output_dir:  str  = "data/raw",
    hardware_id: Optional[str] = None,
) -> Optional[RawBenchmarkResult]:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spec    = autodetect_hardware()
    mapper  = HardwareMapper()
    profile = mapper.map(spec)
    hw_id   = hardware_id or f"colab_{profile.value.lower()}"

    console.rule("[bold cyan]LLM Benchmark Worker — GPU[/bold cyan]")
    console.print(f"  Model HF ID : [green]{model_id}[/green]")
    console.print(f"  Quantization: [yellow]{quantization.value}[/yellow]")
    console.print(f"  HW Profile  : [cyan]{profile.value}[/cyan]")
    console.print(f"  GPU         : {spec.gpu_name or 'N/A'}")
    console.print(f"  VRAM        : {spec.gpu_vram_gb or 'N/A'} GB")
    console.print()

    # — Load model —
    try:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as p:
            t = p.add_task("Loading model from HuggingFace...", total=None)
            model, tokenizer, device = _load_model(model_id, quantization)
            p.update(t, description="[green]Model loaded ✓[/green]")
    except Exception as e:
        console.print(f"[red]Model load failed: {e}[/red]")
        return None

    # — Warmup —
    console.print("[dim]Running warmup...[/dim]")
    try:
        _run_single_gpu(model, tokenizer, BENCHMARK_PROMPTS[0], device)
    except Exception:
        pass

    # — Measured runs —
    all_metrics: list[dict] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as p:
        for run_idx in range(n_runs):
            for pidx, prompt in enumerate(BENCHMARK_PROMPTS):
                label = f"Run {run_idx+1}/{n_runs} · Prompt {pidx+1}/{len(BENCHMARK_PROMPTS)}"
                t = p.add_task(label, total=None)
                try:
                    m = _run_single_gpu(model, tokenizer, prompt, device)
                    all_metrics.append(m)
                    p.update(t, description=f"[green]✓[/green] {label} — {m['tokens_per_sec']:.1f} tok/s")
                except Exception as e:
                    p.update(t, description=f"[red]✗ {label}: {e}[/red]")

    if not all_metrics:
        console.print("[red]All runs failed.[/red]")
        return None

    def avg(key): return statistics.mean(m[key] for m in all_metrics if key in m)

    # — Software stack metadata —
    fw = _get_framework_version(quantization)
    cuda_ver, driver_ver = _get_cuda_info()

    result = RawBenchmarkResult(
        model_id         = model_id.split("/")[-1],
        model_file       = model_id,
        quantization     = quantization,
        hardware_id      = hw_id,
        hardware_profile = profile,

        tokens_per_sec   = avg("tokens_per_sec"),
        latency_first_ms = avg("latency_first_ms"),
        latency_avg_ms   = avg("latency_avg_ms"),
        memory_used_mb   = avg("memory_used_mb"),
        memory_peak_mb   = max(m["memory_peak_mb"] for m in all_metrics),

        prompt_tokens    = int(avg("prompt_tokens")),
        completion_tokens= MAX_TOKENS,
        n_runs           = n_runs,
        status           = RunStatus.SUCCESS,

        prompt_hash      = PROMPT_HASH,
        seed             = SEED,
        python_version   = platform.python_version(),
        framework        = fw,
        cuda_version     = cuda_ver,
        driver_version   = driver_ver,
    )

    out_file = output_path / _make_filename(result)
    with open(out_file, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    console.rule("[bold green]GPU Benchmark Complete[/bold green]")
    console.print(f"  Throughput  : [bold]{result.tokens_per_sec:.2f}[/bold] tokens/sec")
    console.print(f"  TTFT        : {result.latency_first_ms:.1f} ms")
    console.print(f"  Memory peak : {result.memory_peak_mb:.0f} MB")
    console.print(f"  Saved to    : [cyan]{out_file}[/cyan]")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filename(r: RawBenchmarkResult) -> str:
    ts = r.timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{r.model_id}__{r.quantization.value}__{r.hardware_profile.value}__{ts}.json"


def _get_framework_version(q: QuantizationType) -> str:
    try:
        if q in (QuantizationType.GPTQ_4BIT, QuantizationType.GPTQ_8BIT):
            import auto_gptq
            return f"auto-gptq-{auto_gptq.__version__}"
        elif q == QuantizationType.AWQ_4BIT:
            import awq
            return f"autoawq-{awq.__version__}"
        import transformers
        return f"transformers-{transformers.__version__}"
    except Exception:
        return "unknown"


def _get_cuda_info() -> tuple[Optional[str], Optional[str]]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda, None
    except Exception:
        pass
    return None, None