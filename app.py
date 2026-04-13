"""HuggingFace Spaces entrypoint — Dark, technical dashboard."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.db.repository import BenchmarkRepository
from scripts.load_seed_data import load_seed_csv

repo = BenchmarkRepository()
if repo.count() == 0:
    load_seed_csv(repo)
repo.close()

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.core.query_engine import QueryEngine
from src.db.repository import BenchmarkRepository
from src.core.hardware_mapper import HardwareMapper
from src.core.models import HardwareSpec

repo2  = BenchmarkRepository()
engine = QueryEngine(repo2)
mapper = HardwareMapper()

# ── Dark theme palette ─────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN    = "#3fb950"
BLUE     = "#58a6ff"
ORANGE   = "#f78166"
PURPLE   = "#bc8cff"
YELLOW   = "#e3b341"

QUANT_COLORS = {
    "GGUF_Q4_K_M": "#58a6ff",
    "GGUF_Q4_0":   "#79c0ff",
    "GGUF_Q5_K_M": "#bc8cff",
    "GGUF_Q8_0":   "#e3b341",
    "GPTQ_4BIT":   "#3fb950",
    "GPTQ_8BIT":   "#56d364",
    "AWQ_4BIT":    "#f78166",
    "FP16":        "#ff7b72",
}

CHART_STYLE = dict(
    paper_bgcolor = BG2,
    plot_bgcolor  = BG,
    font          = dict(family="monospace", color=TEXT, size=12),
    margin        = dict(l=50, r=20, t=50, b=50),
    legend        = dict(bgcolor=BG2, bordercolor=BORDER, borderwidth=1),
    xaxis         = dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_DIM)),
    yaxis         = dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(color=TEXT_DIM)),
)

CSS = """
body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace !important; }
.gr-button-primary { background: #238636 !important; border: 1px solid #2ea043 !important; color: #fff !important; font-family: monospace !important; }
.gr-button-primary:hover { background: #2ea043 !important; }
.gr-panel, .gr-box, .gr-form { background: #161b22 !important; border: 1px solid #30363d !important; }
label, .gr-block-label { color: #8b949e !important; font-family: monospace !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
.gr-input, .gr-slider input { background: #0d1117 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }
.stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px 20px; text-align: center; }
.stat-num { font-size: 2.2em; font-weight: 700; color: #58a6ff; font-family: monospace; }
.stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.rec-card { background: #161b22; border: 1px solid #30363d; border-left: 3px solid #58a6ff; border-radius: 6px; padding: 14px 18px; margin: 8px 0; }
.rec-best { border-left-color: #3fb950; }
.rec-mem { border-left-color: #bc8cff; }
.rec-lat { border-left-color: #e3b341; }
footer { display: none !important; }
"""

# ── Chart builders ─────────────────────────────────────────────────────────

def empty_fig(msg="No data — click Compare"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=TEXT_DIM))
    fig.update_layout(**CHART_STYLE)
    return fig

def throughput_fig(df):
    if df.empty: return empty_fig()
    fig = px.bar(df, x="model_id", y="tokens_per_sec", color="quantization",
                 barmode="group", color_discrete_map=QUANT_COLORS,
                 text_auto=".1f")
    fig.update_layout(**CHART_STYLE, title=dict(text="⚡ Throughput — tokens / sec", font=dict(color=TEXT)))
    fig.update_traces(textfont_color=TEXT)
    return fig

def memory_fig(df):
    if df.empty: return empty_fig()
    fig = px.bar(df, x="model_id", y="memory_used_mb", color="quantization",
                 barmode="group", color_discrete_map=QUANT_COLORS, text_auto=".0f")
    fig.update_layout(**CHART_STYLE, title=dict(text="💾 Memory — MB used", font=dict(color=TEXT)))
    fig.update_traces(textfont_color=TEXT)
    return fig

def latency_fig(df):
    if df.empty: return empty_fig()
    fig = px.bar(df, x="model_id", y="latency_avg_ms", color="quantization",
                 barmode="group", color_discrete_map=QUANT_COLORS, text_auto=".1f")
    fig.update_layout(**CHART_STYLE, title=dict(text="⏱ Latency — ms / token", font=dict(color=TEXT)))
    fig.update_traces(textfont_color=TEXT)
    return fig

def scatter_fig(df):
    if df.empty: return empty_fig()
    fig = px.scatter(df, x="memory_used_mb", y="tokens_per_sec",
                     color="quantization", symbol="model_id",
                     color_discrete_map=QUANT_COLORS,
                     hover_data=["model_id","quantization","hardware_profile","tokens_per_sec_per_gb"])
    fig.update_traces(marker=dict(size=14, opacity=0.9,
                                  line=dict(width=1, color=BORDER)))
    fig.add_annotation(text="← Better (fast + low memory)", x=0.05, y=0.95,
                       xref="paper", yref="paper", showarrow=False,
                       font=dict(size=11, color=TEXT_DIM))
    fig.update_layout(**CHART_STYLE,
                      title=dict(text="🎯 Efficiency Frontier — speed vs memory", font=dict(color=TEXT)))
    return fig

def format_recs(recs, profile):
    if "error" in recs:
        return f"<div class='rec-card'>⚠️ {recs['error']}</div>"
    html = f"<p style='color:{TEXT_DIM};font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px'>Recommendations for {profile}</p>"
    data = [
        ("best_throughput",        "rec-best", "⚡", "Best Throughput",         "tokens_per_sec",   "tok/s"),
        ("best_memory_efficiency", "rec-mem",  "💾", "Best Memory Efficiency",  "tps_per_gb",       "tok/s/GB"),
        ("best_latency",           "rec-lat",  "⏱", "Lowest Latency",          "latency_ms",       "ms"),
    ]
    for key, cls, icon, label, metric_key, unit in data:
        r = recs[key]
        val = r.get(metric_key, r.get("tokens_per_sec", ""))
        val_str = f"{val:.2f} {unit}" if isinstance(val, float) else str(val)
        html += f"""
        <div class='rec-card {cls}'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <span style='color:{TEXT};font-weight:600'>{icon} {label}</span>
                <span style='color:{BLUE};font-family:monospace;font-size:13px'>{val_str}</span>
            </div>
            <div style='margin-top:6px;font-size:12px'>
                <span style='color:{GREEN}'>{r['model']}</span>
                <span style='color:{TEXT_DIM}'> · {r['quant']}</span>
            </div>
            <div style='margin-top:4px;color:{TEXT_DIM};font-size:11px'>{r['reason']}</div>
        </div>"""
    return html

# ── Main query function ────────────────────────────────────────────────────

def run_query(hw_ram, hw_has_gpu, hw_gpu_vram, sel_quants, sel_models, max_mem):
    spec    = HardwareSpec(ram_gb=hw_ram, cpu_cores=4, has_gpu=hw_has_gpu,
                           gpu_vram_gb=hw_gpu_vram if hw_has_gpu else None)
    profile = mapper.map(spec)
    df      = engine.compare(
        hardware_profile   = profile.value,
        quantization_types = sel_quants or None,
        model_ids          = sel_models or None,
        max_memory_mb      = max_mem if max_mem > 0 else None,
    )
    recs     = engine.recommend(profile.value)
    recs_html = format_recs(recs, profile.value)

    if df.empty:
        empty = pd.DataFrame(columns=["model_id","quantization","hardware_profile",
                                       "tokens_per_sec","memory_used_mb","latency_avg_ms"])
        return empty, empty_fig(), empty_fig(), empty_fig(), empty_fig(), recs_html

    cols = ["model_id","quantization","hardware_profile",
            "tokens_per_sec","memory_used_mb","latency_avg_ms","tokens_per_sec_per_gb"]
    return (df[cols], throughput_fig(df), memory_fig(df),
            latency_fig(df), scatter_fig(df), recs_html)

# ── UI ─────────────────────────────────────────────────────────────────────

stats      = engine.get_db_stats()
all_models = engine.get_available_models() or ["mistral-7b-v0.1","phi-3-mini","gemma-2b","llama-3.2-3b"]
all_quants = engine.get_available_quantization_types() or ["GGUF_Q4_K_M","GGUF_Q4_0","AWQ_4BIT","GPTQ_4BIT"]

with gr.Blocks(title="LLM Benchmark Platform", css=CSS) as demo:

    # ── Header ──
    gr.HTML(f"""
    <div style='padding:24px 0 8px;border-bottom:1px solid #30363d;margin-bottom:20px'>
        <div style='font-size:22px;font-weight:700;color:#e6edf3;font-family:monospace'>
            🔬 LLM Benchmark Platform
        </div>
        <div style='font-size:13px;color:#8b949e;margin-top:4px;font-family:monospace'>
            Quantized model performance on real hardware · GGUF · GPTQ · AWQ
        </div>
    </div>
    """)

    # ── Stats bar ──
    gr.HTML(f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px'>
        <div class='stat-card'><div class='stat-num'>{stats['total_records']}</div><div class='stat-label'>Benchmark Records</div></div>
        <div class='stat-card'><div class='stat-num'>{stats['models']}</div><div class='stat-label'>Models Tested</div></div>
        <div class='stat-card'><div class='stat-num'>{stats['hardware_profiles']}</div><div class='stat-label'>Hardware Tiers</div></div>
        <div class='stat-card'><div class='stat-num'>{stats['quantizations']}</div><div class='stat-label'>Quantization Types</div></div>
    </div>
    """)

    with gr.Row():
        # ── Left panel ──
        with gr.Column(scale=1, min_width=260):
            gr.HTML("<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>⚙ Hardware Config</div>")
            hw_ram  = gr.Slider(4, 64, value=16, step=4, label="RAM (GB)")
            hw_gpu  = gr.Checkbox(label="NVIDIA GPU available", value=False)
            hw_vram = gr.Slider(4, 48, value=16, step=2, label="GPU VRAM (GB)", visible=False)
            hw_gpu.change(lambda x: gr.update(visible=x), hw_gpu, hw_vram)

            gr.HTML("<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px'>🔍 Filters</div>")
            sel_quants = gr.CheckboxGroup(choices=all_quants, value=[], label="Quantization Types")
            sel_models = gr.CheckboxGroup(choices=all_models, value=[], label="Models")
            max_mem    = gr.Slider(0, 16000, value=0, step=500, label="Max Memory (MB) · 0 = no limit")
            btn        = gr.Button("▶  RUN COMPARISON", variant="primary")

            gr.HTML("<div style='font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px'>🏆 Recommendations</div>")
            recs_panel = gr.HTML("<div style='color:#8b949e;font-size:12px'>Run a comparison to see recommendations.</div>")

        # ── Right panel ──
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("⚡ Throughput"):
                    chart_tput = gr.Plot()
                with gr.Tab("💾 Memory"):
                    chart_mem  = gr.Plot()
                with gr.Tab("⏱ Latency"):
                    chart_lat  = gr.Plot()
                with gr.Tab("🎯 Efficiency"):
                    chart_scat = gr.Plot()
                with gr.Tab("📋 Data Table"):
                    table = gr.DataFrame(wrap=True)

    # ── Data Provenance ──
    with gr.Accordion("🔍 Data Provenance", open=False):
        gr.HTML(f"""
        <div style='font-family:monospace;font-size:12px;color:#8b949e;line-height:1.8'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>
                <div>
                    <div style='color:#e6edf3;margin-bottom:8px;font-weight:600'>Dataset</div>
                    <div>Version: <span style='color:#58a6ff'>2026.04.13.1</span></div>
                    <div>Prompt hash: <span style='color:#58a6ff'>e3efec0e5fcd0b22</span></div>
                    <div>Seed records: <span style='color:#3fb950'>25</span> (synthetic baseline)</div>
                    <div>Real records: <span style='color:#3fb950'>{stats['total_records'] - 25}</span> (collected runs)</div>
                </div>
                <div>
                    <div style='color:#e6edf3;margin-bottom:8px;font-weight:600'>Methodology</div>
                    <div>Warmup: 1 run (discarded)</div>
                    <div>Measured: 3 runs (averaged)</div>
                    <div>Tokens: 256 completion</div>
                    <div>Temp: 0.0 · Seed: 42</div>
                </div>
                <div>
                    <div style='color:#e6edf3;margin-bottom:8px;font-weight:600'>Data Sources</div>
                    <div><span style='color:#e3b341'>seed</span> — baseline synthetic data</div>
                    <div><span style='color:#3fb950'>real_colab_t4</span> — Colab T4 GPU runs</div>
                    <div><span style='color:#58a6ff'>real_cpu</span> — local CPU runs</div>
                </div>
                <div>
                    <div style='color:#e6edf3;margin-bottom:8px;font-weight:600'>Outlier Detection</div>
                    <div>Method: IQR (1.5× rule)</div>
                    <div>Grouping: model × quant × hw</div>
                    <div>Flagged rows excluded from UI</div>
                </div>
            </div>
            <div style='margin-top:16px;padding-top:12px;border-top:1px solid #30363d'>
                Reproduce: <a href='https://github.com/Saraid10/llm-benchmark-platform' style='color:#58a6ff'>github.com/Saraid10/llm-benchmark-platform</a>
            </div>
        </div>
        """)
        gr.Markdown(f"""
```
Prompt set    : 5 standardized prompts · SHA256: e3efec0e5fcd0b22
Warmup runs   : 1 discarded
Measured runs : 3 averaged
Max tokens    : 256 completion tokens
Temperature   : 0.0 (deterministic) · Seed: 42
Outlier det.  : IQR method (1.5×IQR) per (model, quant, hw) group

Hardware tiers:
  CPU_LOW    ≤ 8 GB RAM
  CPU_MEDIUM   16 GB RAM
  CPU_HIGH   ≥ 32 GB RAM
  GPU_T4      16 GB VRAM  (Google Colab free tier)
  GPU_A10     24 GB VRAM  (Colab Pro+)

Derived metrics:
  tokens_per_sec_per_gb = throughput ÷ memory_gb
  memory_efficiency     = throughput ÷ memory_mb
  latency_per_token_ms  = latency_avg_ms ÷ completion_tokens
```
To reproduce: github.com/Saraid10/llm-benchmark-platform
        """)

    # ── Wire up ──
    inputs  = [hw_ram, hw_gpu, hw_vram, sel_quants, sel_models, max_mem]
    outputs = [table, chart_tput, chart_mem, chart_lat, chart_scat, recs_panel]

    btn.click(fn=run_query, inputs=inputs, outputs=outputs)
    demo.load(fn=run_query, inputs=inputs, outputs=outputs)

demo.launch()
