"""
Data Loader
===========
Loads the seed CSV dataset into DuckDB.
Run this once before starting the UI.

Usage:
    python scripts/load_seed_data.py

After real benchmark runs, re-run with --raw-dir to ingest JSON files:
    python scripts/load_seed_data.py --raw-dir data/raw/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console

from src.db.repository import BenchmarkRepository
from src.processing.pipeline import Pipeline

console = Console()


def load_seed_csv(repo: BenchmarkRepository, csv_path: str = "data/seed_benchmarks.csv"):
    console.rule("[bold cyan]Loading Seed Dataset[/bold cyan]")
    path = Path(csv_path)
    if not path.exists():
        console.print(f"[red]Seed CSV not found: {csv_path}[/red]")
        return 0

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    console.print(f"  Loaded [green]{len(df)}[/green] rows from {csv_path}")

    # Validate columns match schema
    required_cols = {
        "model_id", "model_file", "quantization", "hardware_id",
        "hardware_profile", "tokens_per_sec", "latency_first_ms",
        "latency_avg_ms", "memory_used_mb", "memory_peak_mb",
        "tokens_per_sec_per_gb", "memory_efficiency", "latency_per_token_ms",
        "prompt_tokens", "completion_tokens", "n_runs", "status",
        "timestamp", "prompt_hash", "seed", "python_version",
        "framework", "cuda_version", "driver_version",
        "is_outlier", "pipeline_version"
    }
    missing = required_cols - set(df.columns)
    if missing:
        console.print(f"[red]Missing columns in CSV: {missing}[/red]")
        return 0

    n = repo.insert_from_dataframe(df)
    console.print(f"[green]✓ Inserted {n} seed records into DuckDB[/green]")
    return n


def load_raw_json_dir(repo: BenchmarkRepository, raw_dir: str):
    console.rule("[bold cyan]Processing Raw JSON Files[/bold cyan]")
    pipeline = Pipeline()
    records  = pipeline.process_directory(raw_dir)
    if records:
        n = repo.insert_batch(records)
        console.print(f"[green]✓ Inserted {n} processed records[/green]")
    else:
        console.print("[yellow]No valid records found in raw directory.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Load benchmark data into DuckDB")
    parser.add_argument("--seed-csv", default="data/seed_benchmarks.csv")
    parser.add_argument("--raw-dir",  default=None,
                        help="Directory of raw JSON benchmark files to process")
    parser.add_argument("--clear",    action="store_true",
                        help="Clear existing data before loading")
    args = parser.parse_args()

    with BenchmarkRepository() as repo:
        if args.clear:
            console.print("[yellow]Clearing existing data...[/yellow]")
            repo.clear()

        if args.raw_dir:
            load_raw_json_dir(repo, args.raw_dir)
        else:
            load_seed_csv(repo, args.seed_csv)

        stats = {
            "total_records":     repo.count(),
            "models":            len(repo.get_model_list()),
            "hardware_profiles": len(repo.get_hardware_profiles()),
            "quantizations":     len(repo.get_quantization_types()),
        }

        console.rule("[bold green]Database Summary[/bold green]")
        for k, v in stats.items():
            console.print(f"  {k:20s}: [bold]{v}[/bold]")


if __name__ == "__main__":
    main()
