"""HuggingFace Spaces entrypoint."""
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
from src.core.query_engine import QueryEngine
from src.db.repository import BenchmarkRepository
from src.core.hardware_mapper import HardwareMapper
from src.core.models import HardwareSpec

repo2  = BenchmarkRepository()
engine = QueryEngine(repo2)
mapper = HardwareMapper()

def query(hw_ram, hw_has_gpu, hw_gpu_vram, sel_quants, sel_models, max_mem):
    spec    = HardwareSpec(ram_gb=hw_ram, cpu_cores=4, has_gpu=hw_has_gpu,
                           gpu_vram_gb=hw_gpu_vram if hw_has_gpu else None)
    profile = mapper.map(spec)
    df      = engine.compare(
        hardware_profile   = profile.value,
        quantization_types = sel_quants or None,
        model_ids          = sel_models or None,
        max_memory_mb      = max_mem if max_mem > 0 else None,
    )
    if df.empty:
        fig = px.bar(title="No data for this selection")
        return pd.DataFrame(), fig, fig
    cols = ["model_id","quantization","hardware_profile",
            "tokens_per_sec","memory_used_mb","latency_avg_ms"]
    fig1 = px.bar(df, x="model_id", y="tokens_per_sec",
                  color="quantization", barmode="group",
                  title="Throughput (tokens/sec)")
    fig2 = px.bar(df, x="model_id", y="memory_used_mb",
                  color="quantization", barmode="group",
                  title="Memory Usage (MB)")
    return df[cols], fig1, fig2

stats      = engine.get_db_stats()
all_models = engine.get_available_models() or ["mistral-7b-v0.1","phi-3-mini","gemma-2b"]
all_quants = engine.get_available_quantization_types() or ["GGUF_Q4_K_M","AWQ_4BIT","GPTQ_4BIT"]

with gr.Blocks(title="LLM Benchmark Platform") as demo:
    gr.Markdown(f"""
# 🔬 LLM Benchmark Platform
Compare quantized LLM performance · **{stats['total_records']} records · {stats['models']} models**
""")
    with gr.Row():
        with gr.Column(scale=1):
            hw_ram     = gr.Slider(4, 64, value=16, step=4, label="RAM (GB)")
            hw_gpu     = gr.Checkbox(label="I have a GPU", value=False)
            hw_vram    = gr.Slider(4, 48, value=16, step=2, label="GPU VRAM (GB)", visible=False)
            hw_gpu.change(lambda x: gr.update(visible=x), hw_gpu, hw_vram)
            sel_quants = gr.CheckboxGroup(choices=all_quants, label="Quantization")
            sel_models = gr.CheckboxGroup(choices=all_models, label="Models")
            max_mem    = gr.Slider(0, 16000, value=0, step=500, label="Max Memory MB")
            btn        = gr.Button("🚀 Compare", variant="primary")
        with gr.Column(scale=3):
            table  = gr.DataFrame(label="Results")
            chart1 = gr.Plot(label="Throughput")
            chart2 = gr.Plot(label="Memory")

    btn.click(fn=query,
              inputs=[hw_ram, hw_gpu, hw_vram, sel_quants, sel_models, max_mem],
              outputs=[table, chart1, chart2])

demo.launch()
