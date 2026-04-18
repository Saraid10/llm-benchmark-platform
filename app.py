"""HuggingFace Spaces entrypoint — Dark technical dashboard v3."""
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

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#0d1117"
BG2     = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
DIM     = "#8b949e"
GREEN   = "#3fb950"
BLUE    = "#58a6ff"
ORANGE  = "#f78166"
PURPLE  = "#bc8cff"
YELLOW  = "#e3b341"

QUANT_COLORS = {
    "GGUF_Q4_K_M": "#58a6ff", "GGUF_Q4_0": "#79c0ff",
    "GGUF_Q5_K_M": "#bc8cff", "GGUF_Q8_0": "#e3b341",
    "GPTQ_4BIT":   "#3fb950", "GPTQ_8BIT": "#56d364",
    "AWQ_4BIT":    "#f78166", "FP16":      "#ff7b72",
}

CHART_BASE = dict(
    paper_bgcolor=BG2, plot_bgcolor=BG,
    font=dict(family="monospace", color=TEXT, size=12),
    margin=dict(l=10, r=20, t=50, b=20),
    legend=dict(bgcolor=BG2, bordercolor=BORDER, borderwidth=1),
)

CSS = """
body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace !important; }
.gr-button-primary { background: #238636 !important; border: 1px solid #2ea043 !important; color: #fff !important; font-family: monospace !important; font-size: 13px !important; }
.gr-button-secondary { background: #161b22 !important; border: 1px solid #30363d !important; color: #8b949e !important; font-family: monospace !important; font-size: 11px !important; }
.gr-panel, .gr-box, .gr-form { background: #161b22 !important; border: 1px solid #30363d !important; }
label, .gr-block-label { color: #8b949e !important; font-family: monospace !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
.gr-input, .gr-slider input { background: #0d1117 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }
.sticky-recs { position: sticky; top: 16px; }
footer { display: none !important; }
"""

# ── Helpers ────────────────────────────────────────────────────────────────

def empty_fig(msg="No data for this selection"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=DIM))
    fig.update_layout(**CHART_BASE)
    return fig

def label(row):
    return f"{row['model_id']} · {row['quantization']}"

def add_methodology_note(fig):
    fig.add_annotation(
        text="3 measured runs · 1 warmup · temp 0.0 · seed 42",
        x=1, y=-0.08, xref="paper", yref="paper",
        showarrow=False, font=dict(size=9, color=DIM),
        xanchor="right"
    )
    return fig

# ── Chart builders ─────────────────────────────────────────────────────────

def throughput_fig(df):
    if df.empty: return empty_fig()
    df = df.copy()
    df["label"] = df.apply(label, axis=1)
    df = df.sort_values("tokens_per_sec", ascending=True)  # ascending for horizontal bar
    fig = go.Figure(go.Bar(
        x=df["tokens_per_sec"], y=df["label"],
        orientation="h",
        marker_color=[QUANT_COLORS.get(q, BLUE) for q in df["quantization"]],
        text=[f"{v:.1f}" for v in df["tokens_per_sec"]],
        textposition="outside", textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f} tok/s<extra></extra>",
    ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="⚡ Throughput — tokens / sec  (higher is better →)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER, tickfont=dict(color=DIM), title="tokens / second"),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT)),
        height=max(300, len(df) * 42 + 80),
    )
    add_methodology_note(fig)
    return fig

def memory_fig(df):
    if df.empty: return empty_fig()
    df = df.copy()
    df["label"] = df.apply(label, axis=1)
    df = df.sort_values("memory_used_mb", ascending=False)  # ascending for horizontal
    fig = go.Figure(go.Bar(
        x=df["memory_used_mb"], y=df["label"],
        orientation="h",
        marker_color=[QUANT_COLORS.get(q, BLUE) for q in df["quantization"]],
        text=[f"{v:.0f} MB" for v in df["memory_used_mb"]],
        textposition="outside", textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.0f} MB<extra></extra>",
    ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="💾 Memory — MB used  (lower is better ←)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER, tickfont=dict(color=DIM), title="Memory used (MB)", autorange="reversed"),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT)),
        height=max(300, len(df) * 42 + 80),
    )
    add_methodology_note(fig)
    return fig

def latency_fig(df):
    if df.empty: return empty_fig()
    df = df.copy()
    df["label"] = df.apply(label, axis=1)
    df = df.sort_values("latency_avg_ms", ascending=False)
    fig = go.Figure(go.Bar(
        x=df["latency_avg_ms"], y=df["label"],
        orientation="h",
        marker_color=[QUANT_COLORS.get(q, BLUE) for q in df["quantization"]],
        text=[f"{v:.1f} ms" for v in df["latency_avg_ms"]],
        textposition="outside", textfont=dict(color=TEXT, size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f} ms/token<extra></extra>",
    ))
    fig.update_layout(**CHART_BASE,
        title=dict(text="⏱ Latency — ms/token  (lower is better ←)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER, tickfont=dict(color=DIM), title="Latency (ms/token)", autorange="reversed"),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT)),
        height=max(300, len(df) * 42 + 80),
    )
    add_methodology_note(fig)
    return fig

def scatter_fig(df):
    if df.empty: return empty_fig()
    df = df.copy()
    df["label"] = df.apply(label, axis=1)

    # Pareto frontier
    pareto = []
    sorted_df = df.sort_values("memory_used_mb")
    best_tps = -1
    for _, row in sorted_df.iterrows():
        if row["tokens_per_sec"] > best_tps:
            pareto.append(row)
            best_tps = row["tokens_per_sec"]
    pareto_df = pd.DataFrame(pareto)

    fig = go.Figure()

    # All points
    for quant in df["quantization"].unique():
        mask = df["quantization"] == quant
        sub  = df[mask]
        fig.add_trace(go.Scatter(
            x=sub["memory_used_mb"], y=sub["tokens_per_sec"],
            mode="markers", name=quant,
            marker=dict(size=14, color=QUANT_COLORS.get(quant, BLUE),
                        opacity=0.85, line=dict(width=1, color=BORDER)),
            text=sub["label"],
            hovertemplate="<b>%{text}</b><br>%{y:.1f} tok/s · %{x:.0f} MB<extra></extra>",
        ))

    # Pareto line
    if len(pareto_df) >= 2:
        fig.add_trace(go.Scatter(
            x=pareto_df["memory_used_mb"], y=pareto_df["tokens_per_sec"],
            mode="lines", name="Pareto frontier",
            line=dict(color=YELLOW, width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

    # Label top 3 by tok/s
    top3 = df.nlargest(3, "tokens_per_sec")
    for _, row in top3.iterrows():
        fig.add_annotation(
            x=row["memory_used_mb"], y=row["tokens_per_sec"],
            text=f"  {row['model_id']}", showarrow=False,
            font=dict(size=10, color=TEXT), xanchor="left",
        )

    fig.add_annotation(text="← top-left = best (fast + low memory)",
                       x=0.02, y=0.97, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=10, color=DIM))
    fig.update_layout(**CHART_BASE,
        title=dict(text="🎯 Efficiency Frontier — speed vs memory", font=dict(color=TEXT)),
        xaxis=dict(gridcolor=BORDER, tickfont=dict(color=DIM), title="Memory used (MB)"),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=DIM), title="Tokens / second"),
        height=420,
    )
    add_methodology_note(fig)
    return fig

# ── Recommendations from filtered df ──────────────────────────────────────

def compute_recs(df, profile):
    if df.empty:
        return "<div style='color:#8b949e;font-size:12px'>No data for this selection.</div>"

    best_tps = df.loc[df["tokens_per_sec"].idxmax()]
    best_mem = df.loc[df["tokens_per_sec_per_gb"].idxmax()]
    best_lat = df.loc[df["latency_avg_ms"].idxmin()]

    # Best overall = highest combined rank
    df2 = df.copy()
    df2["rank_tps"] = df2["tokens_per_sec"].rank(ascending=True)
    df2["rank_mem"] = df2["tokens_per_sec_per_gb"].rank(ascending=True)
    df2["rank_lat"] = df2["latency_avg_ms"].rank(ascending=False)
    df2["combined"] = df2["rank_tps"] + df2["rank_mem"] + df2["rank_lat"]
    best_overall = df2.loc[df2["combined"].idxmax()]

    conf = lambda row: "single run · low confidence" if row.get("n_records", 3) == 1 else f"{row.get('n_records',3)} runs · higher confidence"

    html = f"""
    <div style='font-family:monospace'>
    <div style='font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>
        Tier: {profile}
    </div>

    <div style='background:#161b22;border:1px solid #2ea043;border-left:3px solid #2ea043;
                border-radius:6px;padding:12px;margin-bottom:10px'>
        <div style='font-size:10px;color:{GREEN};text-transform:uppercase;letter-spacing:1px'>🏆 Best Overall</div>
        <div style='color:{TEXT};font-weight:700;margin:4px 0'>{best_overall['model_id']} · {best_overall['quantization']}</div>
        <div style='color:{DIM};font-size:11px'>
            {best_overall['tokens_per_sec']:.1f} tok/s &nbsp;·&nbsp;
            {best_overall['memory_used_mb']:.0f} MB &nbsp;·&nbsp;
            {best_overall['latency_avg_ms']:.1f} ms/tok
        </div>
    </div>

    <div style='background:{BG2};border:1px solid {BORDER};border-left:3px solid {BLUE};
                border-radius:6px;padding:10px;margin-bottom:6px'>
        <div style='font-size:10px;color:{BLUE};text-transform:uppercase;letter-spacing:1px'>⚡ Best Throughput</div>
        <div style='color:{TEXT};font-weight:600;font-size:12px;margin:3px 0'>{best_tps['model_id']} · {best_tps['quantization']}</div>
        <div style='color:{DIM};font-size:11px'>{best_tps['tokens_per_sec']:.1f} tok/s</div>
    </div>

    <div style='background:{BG2};border:1px solid {BORDER};border-left:3px solid {PURPLE};
                border-radius:6px;padding:10px;margin-bottom:6px'>
        <div style='font-size:10px;color:{PURPLE};text-transform:uppercase;letter-spacing:1px'>💾 Best Memory Efficiency</div>
        <div style='color:{TEXT};font-weight:600;font-size:12px;margin:3px 0'>{best_mem['model_id']} · {best_mem['quantization']}</div>
        <div style='color:{DIM};font-size:11px'>{best_mem['tokens_per_sec_per_gb']:.2f} tok/s/GB</div>
    </div>

    <div style='background:{BG2};border:1px solid {BORDER};border-left:3px solid {YELLOW};
                border-radius:6px;padding:10px;margin-bottom:6px'>
        <div style='font-size:10px;color:{YELLOW};text-transform:uppercase;letter-spacing:1px'>⏱ Lowest Latency</div>
        <div style='color:{TEXT};font-weight:600;font-size:12px;margin:3px 0'>{best_lat['model_id']} · {best_lat['quantization']}</div>
        <div style='color:{DIM};font-size:11px'>{best_lat['latency_avg_ms']:.1f} ms/token</div>
    </div>
    </div>"""
    return html

# ── Main query ─────────────────────────────────────────────────────────────

def run_query(hw_preset, hw_ram, hw_has_gpu, hw_gpu_vram,
              sel_quants, sel_models, max_mem, data_source_filter):

    # Apply preset
    if hw_preset == "8GB Laptop":
        hw_ram, hw_has_gpu, hw_gpu_vram = 8, False, 16
    elif hw_preset == "16GB Laptop":
        hw_ram, hw_has_gpu, hw_gpu_vram = 16, False, 16
    elif hw_preset == "32GB Workstation":
        hw_ram, hw_has_gpu, hw_gpu_vram = 32, False, 16
    elif hw_preset == "Colab T4":
        hw_ram, hw_has_gpu, hw_gpu_vram = 12, True, 16
    elif hw_preset == "A10 GPU":
        hw_ram, hw_has_gpu, hw_gpu_vram = 64, True, 24

    spec    = HardwareSpec(ram_gb=hw_ram, cpu_cores=4, has_gpu=hw_has_gpu,
                           gpu_vram_gb=hw_gpu_vram if hw_has_gpu else None)
    profile = mapper.map(spec)

    # Smart quant defaults by hardware
    effective_quants = sel_quants or None
    if not sel_quants:
        if hw_has_gpu:
            effective_quants = ["GPTQ_4BIT", "GPTQ_8BIT", "AWQ_4BIT", "FP16"]
        else:
            effective_quants = ["GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q5_K_M", "GGUF_Q8_0"]

    df = engine.compare(
        hardware_profile   = profile.value,
        quantization_types = effective_quants,
        model_ids          = sel_models or None,
        max_memory_mb      = max_mem if max_mem > 0 else None,
    )

    # Data source filter
    if data_source_filter != "all" and not df.empty and "data_source" in df.columns:
        df = df[df["data_source"] == data_source_filter]

    # Current query strip
    src_label = data_source_filter if data_source_filter != "all" else "all sources"
    query_strip = f"""
    <div style='background:{BG2};border:1px solid {BORDER};border-radius:4px;
                padding:8px 14px;font-family:monospace;font-size:11px;color:{DIM};
                margin-bottom:10px'>
        Tier: <span style='color:{BLUE}'>{profile.value}</span> &nbsp;·&nbsp;
        Source: <span style='color:{GREEN}'>{src_label}</span> &nbsp;·&nbsp;
        Rows: <span style='color:{TEXT}'>{len(df)}</span> &nbsp;·&nbsp;
        Version: <span style='color:{DIM}'>2026.04.13.1</span>
        &nbsp;&nbsp;
        <span style='color:{DIM}'>Mapped tier: </span>
        <span style='color:{YELLOW}'>{profile.value}</span>
    </div>"""

    # Hardware tier badge + compatibility hint
    compat_hint = "GPU mode: AWQ / GPTQ recommended" if hw_has_gpu else "CPU mode: GGUF quantizations recommended"
    compat_color = ORANGE if hw_has_gpu else BLUE
    tier_badge = f"""
    <div style='font-family:monospace;font-size:11px;color:{DIM};margin-top:6px'>
        Mapped tier: <span style='background:{BG2};border:1px solid {BORDER};
        border-radius:3px;padding:2px 8px;color:{BLUE}'>{profile.value}</span>
        <span style='margin-left:10px;color:{compat_color}'>↳ {compat_hint}</span>
    </div>"""

    # Recommendations from filtered df
    recs_html = compute_recs(df, profile.value)

    if df.empty:
        empty = pd.DataFrame(columns=["model_id","quantization","hardware_profile",
                                       "tokens_per_sec","memory_used_mb","latency_avg_ms","data_source"])
        empty_preview = "<div style='color:#8b949e;font-size:12px;font-family:monospace'>No data for this selection.</div>"
        return (empty, empty_fig(), empty_fig(), empty_fig(), empty_fig(),
                recs_html, query_strip, tier_badge, empty_preview)

    # Display columns
    display_cols = ["model_id","quantization","hardware_profile",
                    "tokens_per_sec","memory_used_mb","latency_avg_ms",
                    "tokens_per_sec_per_gb","n_records","data_source"]
    available = [c for c in display_cols if c in df.columns]
    display_df = df[available].copy()

    # Add confidence hints
    if "n_records" in display_df.columns:
        display_df["confidence"] = display_df["n_records"].apply(
            lambda n: "⚠ single run" if n == 1 else f"✓ {n} runs"
        )

    # Mini preview (top 5)
    preview_cols = ["model_id","quantization","tokens_per_sec","memory_used_mb","latency_avg_ms"]
    preview_df = df[preview_cols].head(5).copy()
    preview_df.columns = ["Model","Quant","tok/s","Memory MB","Latency ms/tok"]
    preview_df["tok/s"] = preview_df["tok/s"].round(1)
    preview_df["Memory MB"] = preview_df["Memory MB"].round(0).astype(int)
    preview_df["Latency ms/tok"] = preview_df["Latency ms/tok"].round(1)
    preview_html = preview_df.to_html(index=False, border=0,
        classes="preview-table").replace("<table",
        f"<table style='font-family:monospace;font-size:11px;color:{TEXT};width:100%;border-collapse:collapse'").replace(
        "<th>", f"<th style='color:{DIM};text-align:left;padding:4px 8px;border-bottom:1px solid {BORDER}'>").replace(
        "<td>", f"<td style='padding:3px 8px;border-bottom:1px solid {BG2}'>")

    return (display_df, throughput_fig(df), memory_fig(df), latency_fig(df),
            scatter_fig(df), recs_html, query_strip, tier_badge,
            f"<div style='font-size:11px'>{preview_html}</div>")

# ── Build UI ───────────────────────────────────────────────────────────────

stats        = engine.get_db_stats()
source_counts = repo2.get_source_counts()
seed_count   = source_counts.get("seed", 0)
real_count   = sum(v for k, v in source_counts.items() if k != "seed")
all_models   = engine.get_available_models() or ["mistral-7b-v0.1","phi-3-mini","gemma-2b","llama-3.2-3b"]
all_quants   = engine.get_available_quantization_types() or ["GGUF_Q4_K_M","GGUF_Q4_0","AWQ_4BIT","GPTQ_4BIT"]

with gr.Blocks(title="LLM Benchmark Platform", css=CSS) as demo:

    # Header
    gr.HTML(f"""
    <div style='padding:20px 0 12px;border-bottom:1px solid #30363d;margin-bottom:16px'>
        <div style='font-size:20px;font-weight:700;color:#e6edf3;font-family:monospace'>
            🔬 LLM Benchmark Platform
        </div>
        <div style='font-size:12px;color:#8b949e;margin-top:4px;font-family:monospace'>
            What should I run on my hardware? · GGUF · GPTQ · AWQ · CPU · GPU
        </div>
    </div>
    """)

    # Stats bar
    gr.HTML(f"""
    <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px'>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center'>
            <div style='font-size:2em;font-weight:700;color:#58a6ff;font-family:monospace'>{stats['total_records']}</div>
            <div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>Total Records</div>
            <div style='font-size:10px;color:#8b949e;margin-top:2px'>{seed_count} seed · {real_count} real</div>
        </div>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center'>
            <div style='font-size:2em;font-weight:700;color:#3fb950;font-family:monospace'>{real_count}</div>
            <div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>Real Runs</div>
            <div style='font-size:10px;color:#3fb950;margin-top:2px'>measured · verified</div>
        </div>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center'>
            <div style='font-size:2em;font-weight:700;color:#e6edf3;font-family:monospace'>{stats['models']}</div>
            <div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>Models</div>
        </div>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center'>
            <div style='font-size:2em;font-weight:700;color:#bc8cff;font-family:monospace'>{stats['hardware_profiles']}</div>
            <div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>HW Tiers</div>
        </div>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center'>
            <div style='font-size:2em;font-weight:700;color:#e3b341;font-family:monospace'>{stats['quantizations']}</div>
            <div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px'>Quant Types</div>
        </div>
    </div>
    """)

    with gr.Row():
        # ── Left: Controls (24%) ──
        with gr.Column(scale=3, min_width=240):
            gr.HTML(f"<div style='font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>⚡ Quick Presets</div>")
            hw_preset = gr.Radio(
                choices=["Custom", "8GB Laptop", "16GB Laptop", "32GB Workstation", "Colab T4", "A10 GPU"],
                value="16GB Laptop", label="Hardware Preset", container=False,
            )
            tier_badge = gr.HTML()

            gr.HTML(f"<div style='font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px'>⚙ Manual Config</div>")
            hw_ram    = gr.Slider(4, 64, value=16, step=4, label="RAM (GB)")
            hw_gpu    = gr.Checkbox(label="NVIDIA GPU", value=False)
            hw_vram   = gr.Slider(4, 48, value=16, step=2, label="GPU VRAM (GB)", visible=False)
            hw_gpu.change(lambda x: gr.update(visible=x), hw_gpu, hw_vram)

            gr.HTML(f"<div style='font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px'>🔍 Filters</div>")
            data_src = gr.Radio(
                choices=["all", "seed", "real_cpu", "real_colab_t4"],
                value="all", label="Data Source",
            )
            sel_quants = gr.CheckboxGroup(choices=all_quants, value=[], label="Quantization (auto if empty)")
            sel_models = gr.CheckboxGroup(choices=all_models, value=[], label="Models (all if empty)")
            max_mem    = gr.Slider(0, 16000, value=0, step=500, label="Max Memory MB (0=no limit)")

            with gr.Row():
                btn       = gr.Button("▶ COMPARE", variant="primary")
                reset_btn = gr.Button("↺ Reset", variant="secondary")

        # ── Middle: Charts + Table (52%) ──
        with gr.Column(scale=6):
            query_strip = gr.HTML()
            with gr.Tabs():
                with gr.Tab("⚡ Throughput"):
                    chart_tput = gr.Plot()
                with gr.Tab("💾 Memory"):
                    chart_mem  = gr.Plot()
                with gr.Tab("⏱ Latency"):
                    chart_lat  = gr.Plot()
                with gr.Tab("🎯 Efficiency"):
                    chart_scat = gr.Plot()
                with gr.Tab("📋 Full Table"):
                    table = gr.DataFrame(wrap=True)
            gr.HTML(f"<div style='font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px'>Top 5 Preview</div>")
            mini_preview = gr.HTML()

        # ── Right: Recommendations (24%) ──
        with gr.Column(scale=3, min_width=240):
            gr.HTML(f"""
            <div class='sticky-recs'>
            <div style='font-size:10px;color:{DIM};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>🏆 Recommendations</div>
            </div>
            """)
            recs_panel = gr.HTML(
                f"<div style='color:{DIM};font-size:12px;font-family:monospace;position:sticky;top:16px'>"
                "Run a comparison to see recommendations.</div>"
            )

    # Methodology
    with gr.Accordion("📖 Methodology & Data Provenance", open=False):
        gr.Markdown(f"""
**Benchmark protocol:** 5 standardized prompts · SHA256 `e3efec0e5fcd0b22` · 1 warmup + 3 measured runs · 256 tokens · temp 0.0 · seed 42

**Data sources:** `seed` = synthetic baseline · `real_cpu` = measured on CPU · `real_colab_t4` = measured on Colab T4

**Outlier detection:** IQR method (1.5×) per (model, quantization, hardware) group — flagged rows excluded

**Limitations:** Seed data is synthetic. AWQ results use `fuse_layers=False` (3–5× slower than fused). Validate on your workload.

**Reproduce:** [github.com/Saraid10/llm-benchmark-platform](https://github.com/Saraid10/llm-benchmark-platform)
        """)

    # ── Wire up ──
    inputs  = [hw_preset, hw_ram, hw_gpu, hw_vram, sel_quants, sel_models, max_mem, data_src]
    outputs = [table, chart_tput, chart_mem, chart_lat, chart_scat,
               recs_panel, query_strip, tier_badge, mini_preview]

    btn.click(fn=run_query, inputs=inputs, outputs=outputs)
    demo.load(fn=run_query, inputs=inputs, outputs=outputs)

    def reset_filters():
        return ("Custom", 16, False, 16, [], [], 0, "all")
    reset_btn.click(fn=reset_filters, inputs=[],
                    outputs=[hw_preset, hw_ram, hw_gpu, hw_vram, sel_quants, sel_models, max_mem, data_src])

demo.launch()
