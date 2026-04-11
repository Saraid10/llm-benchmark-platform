"""
Submit benchmark results to the ingestion API.
Run this after collecting raw JSON files from workers.

Usage:
    # Submit all files from data/raw/
    python scripts/submit_results.py --raw-dir data/raw/

    # Submit a single file
    python scripts/submit_results.py --file data/raw/mistral__GPTQ_4BIT__GPU_T4__20260409.json

    # Offline mode (processes files directly into DB without API)
    python scripts/submit_results.py --raw-dir data/raw/ --offline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from src.api.client import BenchmarkClient

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Submit benchmark results")
    parser.add_argument("--raw-dir",   default=None, help="Directory of JSON files")
    parser.add_argument("--file",      default=None, help="Single JSON file")
    parser.add_argument("--api-url",   default="http://localhost:8000")
    parser.add_argument("--api-key",   default="dev")
    parser.add_argument("--offline",   action="store_true",
                        help="Skip API, process directly into DB")
    parser.add_argument("--changelog", default="Manual submission",
                        help="Describe what changed in this batch")
    args = parser.parse_args()

    if args.offline:
        # Bypass API — process files directly into DuckDB
        from src.db.repository import BenchmarkRepository
        from src.processing.pipeline import Pipeline
        from src.versioning.versioning import VersionRegistry

        repo     = BenchmarkRepository()
        pipeline = Pipeline()
        registry = VersionRegistry()

        raw_dir = args.raw_dir or Path(args.file).parent
        records = pipeline.process_directory(raw_dir)
        n       = repo.insert_batch(records)

        if n > 0:
            registry.create_version(
                record_count      = repo.count(),
                model_ids         = repo.get_model_list(),
                hardware_profiles = repo.get_hardware_profiles(),
                quantizations     = repo.get_quantization_types(),
                changelog         = args.changelog,
                prompt_hash       = records[0].prompt_hash if records else "",
                record_ids        = [f"{r.model_id}_{r.timestamp}" for r in records],
            )
        repo.close()
        return

    client = BenchmarkClient(api_url=args.api_url, api_key=args.api_key)

    if args.file:
        success = client.submit_file(args.file, changelog=args.changelog)
        console.print("[green]✓ Submitted[/green]" if success else "[red]✗ Failed[/red]")

    elif args.raw_dir:
        result = client.submit_directory(args.raw_dir, changelog=args.changelog)
        console.print(f"[green]Done: {result['success']} submitted, {result['failed']} failed[/green]")

    else:
        console.print("[red]Provide --file or --raw-dir[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
