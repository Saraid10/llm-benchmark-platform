"""
LLM Benchmark Platform — Gradio UI
====================================
Pure presentation layer. No business logic here.
All data flows through QueryEngine → BenchmarkRepository → DuckDB.

Deployable on Hugging Face Spaces with zero modification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# — Color palette (consistent across all charts) —
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

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="monospace", size=12),
    margin       =dict(l=40, r=20, t=50, b=40),
)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _throughput_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: tokens/sec by model, colored by quantization."""
    if df.empty:
        return _empty_chart("No data — run a query first")

    fig = px.bar(
        df,
        x          = "model_id",
        y          = "tokens_per_sec",
        color      = "quantization",
        barmode    = "group",
        title      = "⚡ Throughput — Tokens per Second",
        labels     = {"tokens_per_sec": "Tokens / sec", "model_id": "Model"},
        color_discrete_map = QUANT_COLORS,
        text_auto  = ".1f",
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(textposition="outside")
    return fig


def _memory_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: memory usage."""
    if df.empty:
        return _empty_chart("No data")

    fig = px.bar(
        df,
        x         = "model_id",
        y         = "memory_used_mb",
        color     = "quantization",
        barmode   = "group",
        title     = "💾 Memory Usage (MB)",
        labels    = {"memory_used_mb": "Memory (MB)", "model_id": "Model"},
        color_discrete_map = QUANT_COLORS,
        text_auto = ".0f",
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


def _latency_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: average latency per token."""
    if df.empty:
        return _empty_chart("No data")

    fig = px.bar(
        df,
        x         = "model_id",
        y         = "latency_avg_ms",
        color     = "quantization",
        barmode   = "group",
        title     = "⏱ Average Latency (ms / token)",
        labels    = {"latency_avg_ms": "Latency (ms)", "model_id": "Model"},
        color_discrete_map = QUANT_COLORS,
        text_auto = ".1f",
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


def _efficiency_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: throughput vs memory.
    Points in the top-left are best (fast + low memory).
    This is the most informative chart for deployment decisions.
    """
    if df.empty:
        return _empty_chart("No data")

    fig = px.scatter(
        df,
        x          = "memory_used_mb",
        y          = "tokens_per_sec",
        color      = "quantization",
        symbol     = "model_id",
        size_max   = 18,
        title      = "🎯 Efficiency Frontier — Speed vs Memory",
        labels     = {
            "memory_used_mb": "Memory Used (MB)",
            "tokens_per_sec": "Tokens / sec",
        },
        hover_data = ["model_id", "quantization", "hardware_profile",
                      "tokens_per_sec_per_gb"],
        color_discrete_map = QUANT_COLORS,
    )
    # Add quadrant annotation
    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text="← Better",
        showarrow=False, font=dict(size=11, color="gray"),
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(marker=dict(size=12, opacity=0.85))
    return fig


def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper",
                       yref="paper", showarrow=False,
                       font=dict(size=16, color="gray"))
    fig.update_layout(**CHART_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Core query handler
# ---------------------------------------------------------------------------

def run_query(
    hw_ram_gb:       float,
    hw_has_gpu:      bool,
    hw_gpu_vram:     float,
    selected_quants: list[str],
    selected_models: list[str],
    max_memory_mb:   float,
) -> tuple:
    """
    Called on every UI interaction.
    Returns: (table_df, throughput_fig, memory_fig, latency_fig, scatter_fig, recommendations_md)
    """
    # Build hardware spec from UI inputs
    spec = HardwareSpec(
        ram_gb      = hw_ram_gb,
        cpu_cores   = 4,          # approximation — doesn't affect tier mapping
        has_gpu     = hw_has_gpu,
        gpu_vram_gb = hw_gpu_vram if hw_has_gpu else None,
    )
    profile = mapper.map(spec)

    # Query
    df = engine.compare(
        hardware_profile   = profile.value,
        quantization_types = selected_quants if selected_quants else None,
        model_ids          = selected_models  if selected_models  else None,
        max_memory_mb      = max_memory_mb    if max_memory_mb > 0 else None,
    )

    # Recommendations
    recs_raw = engine.recommend(
        hardware_profile = profile.value,
        max_memory_mb    = max_memory_mb if max_memory_mb > 0 else None,
    )
    recs_md  = _format_recommendations(recs_raw, profile.value)

    # Display columns for table
    display_cols = [
        "model_id", "quantization", "hardware_profile",
        "tokens_per_sec", "memory_used_mb", "latency_avg_ms",
        "tokens_per_sec_per_gb", "n_records",
    ]
    display_df = df[display_cols] if not df.empty else pd.DataFrame(columns=display_cols)

    return (
        display_df,
        _throughput_chart(df),
        _memory_chart(df),
        _latency_chart(df),
        _efficiency_scatter(df),
        recs_md,
    )


def _format_recommendations(recs: dict, profile: str) -> str:
    if "error" in recs:
        return f"⚠️ {recs['error']}\n\nRun some benchmarks on this hardware tier first."

    lines = [f"## 🔍 Recommendations for `{profile}`\n"]

    icons = {
        "best_throughput":         ("⚡", "Best Throughput"),
        "best_memory_efficiency":  ("💾", "Best Memory Efficiency"),
        "best_latency":            ("⏱", "Best Latency"),
    }

    for key, (icon, label) in icons.items():
        r = recs[key]
        lines.append(f"### {icon} {label}")
        lines.append(f"**Model:** `{r['model']}`  |  **Quant:** `{r['quant']}`")
        lines.append(f"> {r['reason']}\n")

    lines.append("---")
    lines.append(
        "_These are data-driven suggestions, not absolute rankings. "
        "Always validate on your specific workload._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app definition
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:

    # Load dynamic options from DB
    all_models = engine.get_available_models() or [
        "mistral-7b-v0.1", "llama-3.2-3b", "phi-3-mini", "gemma-2b"
    ]
    all_quants = engine.get_available_quantization_types() or [
        "GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q5_K_M", "GGUF_Q8_0",
        "GPTQ_4BIT", "GPTQ_8BIT", "AWQ_4BIT",
    ]
    stats = engine.get_db_stats()

    with gr.Blocks(
        title="LLM Benchmark Platform",
        theme=gr.themes.Base(
            primary_hue   = gr.themes.colors.blue,
            secondary_hue = gr.themes.colors.slate,
        ),
        css="""
        .stat-box { text-align: center; padding: 12px; border-radius: 8px;
                    background: #1e293b; color: #e2e8f0; }
        .stat-num  { font-size: 2em; font-weight: bold; color: #3b82f6; }
        #rec-panel { background: #0f172a; border-radius: 8px; padding: 16px; }
        """,
    ) as app:

        # — Header —
        gr.Markdown("""
# 🔬 LLM Benchmark Platform
### Quantized Model Performance on Real Hardware
Compare GGUF · GPTQ · AWQ inference speed, memory, and latency
across CPU and GPU environments. Data from controlled runs — not synthetic.
---
        """)

        # — Stats row —
        with gr.Row():
            gr.HTML(f'<div class="stat-box"><div class="stat-num">{stats["total_records"]}</div>Benchmark Records</div>')
            gr.HTML(f'<div class="stat-box"><div class="stat-num">{stats["models"]}</div>Models</div>')
            gr.HTML(f'<div class="stat-box"><div class="stat-num">{stats["hardware_profiles"]}</div>HW Profiles</div>')
            gr.HTML(f'<div class="stat-box"><div class="stat-num">{stats["quantizations"]}</div>Quantization Types</div>')

        gr.Markdown("---")

        with gr.Row():
            # — Left panel: filters —
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Hardware Configuration")

                hw_ram = gr.Slider(
                    minimum=4, maximum=64, value=16, step=4,
                    label="Your RAM (GB)",
                    info="Used to determine hardware tier"
                )
                hw_has_gpu = gr.Checkbox(
                    label="I have an NVIDIA GPU",
                    value=False,
                )
                hw_gpu_vram = gr.Slider(
                    minimum=4, maximum=48, value=16, step=2,
                    label="GPU VRAM (GB)",
                    visible=False,
                )

                # Show/hide VRAM slider based on GPU checkbox
                hw_has_gpu.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=hw_has_gpu,
                    outputs=hw_gpu_vram,
                )

                gr.Markdown("## 🔍 Filters")

                sel_quants = gr.CheckboxGroup(
                    choices = all_quants,
                    value   = [],
                    label   = "Quantization Types (all if none selected)",
                )
                sel_models = gr.CheckboxGroup(
                    choices = all_models,
                    value   = [],
                    label   = "Models (all if none selected)",
                )
                max_mem = gr.Slider(
                    minimum=0, maximum=16000, value=0, step=500,
                    label="Max Memory (MB) — 0 = no limit",
                )

                run_btn = gr.Button("🚀 Run Comparison", variant="primary", size="lg")

            # — Right panel: results —
            with gr.Column(scale=3):
                gr.Markdown("## 📊 Results")
                with gr.Tabs():
                    with gr.Tab("📈 Throughput"):
                        throughput_chart = gr.Plot()
                    with gr.Tab("💾 Memory"):
                        memory_chart = gr.Plot()
                    with gr.Tab("⏱ Latency"):
                        latency_chart = gr.Plot()
                    with gr.Tab("🎯 Efficiency Frontier"):
                        scatter_chart = gr.Plot()
                    with gr.Tab("📋 Data Table"):
                        result_table = gr.DataFrame(
                            wrap=True,
                            column_widths=["15%", "12%", "13%",
                                           "12%", "12%", "12%", "12%", "8%"],
                        )

        # — Recommendations panel —
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                recs_panel = gr.Markdown(
                    value="*Run a query above to see recommendations.*",
                    elem_id="rec-panel",
                )

        # — Data health + version history —
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🏥 Data Health")
                health_btn = gr.Button("Refresh Health Report", size="sm")
                quality_display = gr.Markdown("*Click Refresh to load.*")
                gaps_display    = gr.Markdown()

            with gr.Column(scale=2):
                gr.Markdown("## 📜 Version History")
                version_display = gr.Markdown(
                    value = registry.format_changelog_table()
                )

        def refresh_health():
            score = monitor.quality_score()
            gaps  = monitor.coverage_gaps()
            color = "🟢" if score["score"] >= 75 else "🟡" if score["score"] >= 50 else "🔴"

            quality_md = f"""
**{color} Quality Score: {score['score']}/100**
{score['message']}

| Dimension | Score |
|---|---|
| Freshness  | {score['breakdown'].get('freshness', 0):.0f}/100 |
| Coverage   | {score['breakdown'].get('coverage', 0):.0f}/100 |
| Outliers   | {score['breakdown'].get('outlier', 0):.0f}/100 |
| Volume     | {score['breakdown'].get('volume', 0):.0f}/100 |
            """
            gaps_md = f"""
**Coverage:** {gaps['covered']}/{gaps['total_possible']} cells ({gaps['coverage_pct']}%)

**Next benchmarks to run:**
""" + "\n".join(
                f"- `{g['model']}` on `{g['hardware']}` ({g['quantization']})"
                for g in gaps["gaps"][:5]
            ) if gaps["gaps"] else "\n✅ All cells covered!"

            return quality_md, gaps_md

        health_btn.click(
            fn      = refresh_health,
            inputs  = [],
            outputs = [quality_display, gaps_display],
        )

        # — How to use —
        with gr.Accordion("📖 How to Use / Methodology", open=False):
            gr.Markdown("""
### Benchmark Methodology

- **Prompt set**: 5 standardized prompts (same across all runs), SHA256-hashed for traceability
- **Warm-up**: 1 discarded warm-up run before measurement
- **Runs**: 3 measured runs, results averaged
- **Metrics**:
  - `tokens_per_sec` — completion tokens ÷ total generation time
  - `latency_first_ms` — time to first token (TTFT)
  - `latency_avg_ms` — average per-token latency
  - `memory_used_mb` — RSS delta during generation
  - `tokens_per_sec_per_gb` — throughput ÷ memory in GB (efficiency)
- **Outlier detection**: IQR method, flagged records excluded from display
- **Hardware tiers**:
  - `CPU_LOW` ≤ 8 GB RAM | `CPU_MEDIUM` 16 GB | `CPU_HIGH` 32+ GB
  - `GPU_T4` 16 GB VRAM (Colab) | `GPU_A10` 24 GB VRAM

### Reproduce These Results

```bash
git clone https://github.com/YOUR_USERNAME/llm-benchmark-platform
pip install -r requirements.txt

# CPU benchmark
python -m src.workers.cpu_worker \\
    --model models/mistral-7b-v0.1.Q4_K_M.gguf \\
    --quantization GGUF_Q4_K_M \\
    --n-runs 3

# Load results into DB
python scripts/load_seed_data.py --raw-dir data/raw/
```
            """)

        # — Wire up the button —
        run_btn.click(
            fn      = run_query,
            inputs  = [hw_ram, hw_has_gpu, hw_gpu_vram,
                       sel_quants, sel_models, max_mem],
            outputs = [result_table, throughput_chart, memory_chart,
                       latency_chart, scatter_chart, recs_panel],
        )

        # Auto-run on load with defaults
        app.load(
            fn      = run_query,
            inputs  = [hw_ram, hw_has_gpu, hw_gpu_vram,
                       sel_quants, sel_models, max_mem],
            outputs = [result_table, throughput_chart, memory_chart,
                       latency_chart, scatter_chart, recs_panel],
        )

    return app


if __name__ == "__main__":
    import os
    if os.path.exists("data/benchmarks.duckdb"):
        os.remove("data/benchmarks.duckdb")
        try:
            os.remove("data/benchmarks.duckdb.wal")
        except FileNotFoundError:
            pass

    from scripts.load_seed_data import load_seed_csv
    load_seed_csv(repo)

    app = build_app()
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
    )
