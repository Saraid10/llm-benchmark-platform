"""
LLM Benchmark Platform — Gradio UI
Pure presentation layer. HuggingFace Spaces compatible.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.hardware_mapper import HardwareMapper
from src.core.models import HardwareSpec
from src.core.query_engine import QueryEngine
from src.db.repository import BenchmarkRepository
from src.versioning.versioning import VersionRegistry
from src.monitoring.monitor import DataMonitor

# — Initialize backend —
repo     = BenchmarkRepository()
engine   = QueryEngine(repo)
mapper   = HardwareMapper()
registry = VersionRegistry()
monitor  = DataMonitor(repo)

# Load seed data if DB is empty
if repo.count() == 0:
    from scripts.load_seed_data import load_seed_csv
    load_seed_csv(repo)

QUANT_COLORS = {
    "GGUF_Q4_K_M": "#3B82F6",
    "GGUF_Q4_0":   "#06B6D4",
    "GGUF_Q5_K_M": "#8B5CF6",
    "GGUF_Q8_0":   "#F59E0B",
    "GPTQ_4BIT":   "#10B981",
    "GPTQ_8BIT":   "#34D399",
    "AWQ_4BIT":    "#F97316",
    "FP16":        "#EF4444",
}


def make_empty_fig(msg="No data — run a query first"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False, font={"size": 16, "color": "gray"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 50, "b": 40}
    )
    return fig


def throughput_chart(df):
    if df.empty:
        return make_empty_fig()
    fig = px.bar(
        df, x="model_id", y="tokens_per_sec",
        color="quantization", barmode="group",
        title="⚡ Throughput — Tokens per Second",
        labels={"tokens_per_sec": "Tokens/sec", "model_id": "Model"},
        color_discrete_map=QUANT_COLORS,
        text_auto=".1f",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 50, "b": 40}
    )
    return fig


def memory_chart(df):
    if df.empty:
        return make_empty_fig()
    fig = px.bar(
        df, x="model_id", y="memory_used_mb",
        color="quantization", barmode="group",
        title="💾 Memory Usage (MB)",
        labels={"memory_used_mb": "Memory (MB)", "model_id": "Model"},
        color_discrete_map=QUANT_COLORS,
        text_auto=".0f",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 50, "b": 40}
    )
    return fig


def latency_chart(df):
    if df.empty:
        return make_empty_fig()
    fig = px.bar(
        df, x="model_id", y="latency_avg_ms",
        color="quantization", barmode="group",
        title="⏱ Avg Latency (ms/token)",
        labels={"latency_avg_ms": "Latency (ms)", "model_id": "Model"},
        color_discrete_map=QUANT_COLORS,
        text_auto=".1f",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 50, "b": 40}
    )
    return fig


def scatter_chart(df):
    if df.empty:
        return make_empty_fig()
    fig = px.scatter(
        df, x="memory_used_mb", y="tokens_per_sec",
        color="quantization", symbol="model_id",
        title="🎯 Speed vs Memory — Efficiency Frontier",
        labels={"memory_used_mb": "Memory (MB)", "tokens_per_sec": "Tokens/sec"},
        hover_data=["model_id", "quantization", "hardware_profile", "tokens_per_sec_per_gb"],
        color_discrete_map=QUANT_COLORS,
    )
    fig.update_traces(marker={"size": 12, "opacity": 0.85})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 20, "t": 50, "b": 40}
    )
    return fig


def run_query(hw_ram, hw_has_gpu, hw_gpu_vram, sel_quants, sel_models, max_mem):
    spec = HardwareSpec(
        ram_gb=hw_ram, cpu_cores=4,
        has_gpu=hw_has_gpu,
        gpu_vram_gb=hw_gpu_vram if hw_has_gpu else None,
    )
    profile = mapper.map(spec)

    df = engine.compare(
        hardware_profile   = profile.value,
        quantization_types = sel_quants if sel_quants else None,
        model_ids          = sel_models if sel_models else None,
        max_memory_mb      = max_mem if max_mem > 0 else None,
    )

    recs     = engine.recommend(profile.value)
    recs_md  = format_recs(recs, profile.value)

    display_cols = [
        "model_id", "quantization", "hardware_profile",
        "tokens_per_sec", "memory_used_mb", "latency_avg_ms",
        "tokens_per_sec_per_gb", "n_records",
    ]
    display_df = df[display_cols] if not df.empty else pd.DataFrame(columns=display_cols)

    return (
        display_df,
        throughput_chart(df),
        memory_chart(df),
        latency_chart(df),
        scatter_chart(df),
        recs_md,
    )


def format_recs(recs, profile):
    if "error" in recs:
        return f"⚠️ {recs['error']}"
    lines = [f"## 🔍 Recommendations for `{profile}`\n"]
    icons = {
        "best_throughput":        ("⚡", "Best Throughput"),
        "best_memory_efficiency": ("💾", "Best Memory Efficiency"),
        "best_latency":           ("⏱", "Best Latency"),
    }
    for key, (icon, label) in icons.items():
        r = recs[key]
        lines.append(f"### {icon} {label}")
        lines.append(f"**Model:** `{r['model']}` | **Quant:** `{r['quant']}`")
        lines.append(f"> {r['reason']}\n")
    return "\n".join(lines)


def build_app():
    all_models = engine.get_available_models() or [
        "mistral-7b-v0.1", "llama-3.2-3b", "phi-3-mini", "gemma-2b"
    ]
    all_quants = engine.get_available_quantization_types() or [
        "GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q5_K_M", "GGUF_Q8_0",
        "GPTQ_4BIT", "GPTQ_8BIT", "AWQ_4BIT",
    ]
    stats = engine.get_db_stats()

    with gr.Blocks(title="LLM Benchmark Platform") as app:

        gr.Markdown(f"""
# 🔬 LLM Benchmark Platform
### Compare quantized LLM performance across real hardware
GGUF · GPTQ · AWQ · CPU · GPU · Edge devices

**{stats['total_records']} benchmark records** · **{stats['models']} models** · **{stats['hardware_profiles']} hardware profiles** · **{stats['quantizations']} quantization types**
---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Your Hardware")
                hw_ram = gr.Slider(
                    minimum=4, maximum=64, value=16, step=4,
                    label="RAM (GB)"
                )
                hw_has_gpu = gr.Checkbox(label="I have an NVIDIA GPU", value=False)
                hw_gpu_vram = gr.Slider(
                    minimum=4, maximum=48, value=16, step=2,
                    label="GPU VRAM (GB)", visible=False
                )
                hw_has_gpu.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=hw_has_gpu, outputs=hw_gpu_vram
                )

                gr.Markdown("## 🔍 Filters")
                sel_quants = gr.CheckboxGroup(
                    choices=all_quants, value=[],
                    label="Quantization (all if none selected)"
                )
                sel_models = gr.CheckboxGroup(
                    choices=all_models, value=[],
                    label="Models (all if none selected)"
                )
                max_mem = gr.Slider(
                    minimum=0, maximum=16000, value=0, step=500,
                    label="Max Memory MB (0 = no limit)"
                )
                run_btn = gr.Button("🚀 Run Comparison", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("## 📊 Results")
                with gr.Tabs():
                    with gr.Tab("⚡ Throughput"):
                        tput_chart = gr.Plot()
                    with gr.Tab("💾 Memory"):
                        mem_chart = gr.Plot()
                    with gr.Tab("⏱ Latency"):
                        lat_chart = gr.Plot()
                    with gr.Tab("🎯 Efficiency"):
                        scat_chart = gr.Plot()
                    with gr.Tab("📋 Table"):
                        result_table = gr.DataFrame(wrap=True)

        gr.Markdown("---")
        recs_panel = gr.Markdown("*Run a query to see recommendations.*")

        with gr.Accordion("📖 Methodology", open=False):
            gr.Markdown("""
**Benchmark methodology:**
- 5 standardized prompts, SHA256-hashed for traceability
- 1 warmup run (discarded) + 3 measured runs averaged
- 256 completion tokens per prompt
- Temperature 0.0 (deterministic), seed 42
- Outlier detection: IQR method (1.5×IQR)

**Hardware tiers:** `CPU_LOW` ≤8GB | `CPU_MEDIUM` 16GB | `CPU_HIGH` 32GB+ | `GPU_T4` 16GB VRAM | `GPU_A10` 24GB VRAM
            """)

        inputs  = [hw_ram, hw_has_gpu, hw_gpu_vram, sel_quants, sel_models, max_mem]
        outputs = [result_table, tput_chart, mem_chart, lat_chart, scat_chart, recs_panel]

        run_btn.click(fn=run_query, inputs=inputs, outputs=outputs)
        app.load(fn=run_query, inputs=inputs, outputs=outputs)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
