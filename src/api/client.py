"""
API Client
==========
Used by workers to submit results to the ingestion API.
Workers can either write JSON files directly (offline mode)
or POST to the API (online mode).

This client handles both modes transparently.

Usage:
    client = BenchmarkClient(api_url="http://localhost:8000", api_key="dev")

    # Submit a single result
    client.submit(result)

    # Or in offline mode (writes to file)
    client = BenchmarkClient(offline=True, output_dir="data/raw")
    client.submit(result)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.core.models import RawBenchmarkResult

console = Console()


class BenchmarkClient:

    def __init__(
        self,
        api_url:    str  = "http://localhost:8000",
        api_key:    str  = "dev",
        offline:    bool = False,
        output_dir: str  = "data/raw",
    ):
        self.api_url    = api_url.rstrip("/")
        self.api_key    = api_key
        self.offline    = offline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        result:    RawBenchmarkResult,
        changelog: str = "Worker submission",
    ) -> bool:
        """
        Submit a benchmark result.
        Falls back to offline mode if API is unreachable.
        """
        if self.offline:
            return self._write_file(result)

        try:
            return self._post_to_api(result, changelog)
        except Exception as e:
            console.print(f"[yellow]API unreachable ({e}). Falling back to file mode.[/yellow]")
            return self._write_file(result)

    def _post_to_api(self, result: RawBenchmarkResult, changelog: str) -> bool:
        import urllib.request
        import urllib.error

        payload = result.model_dump(mode="json")
        payload["changelog"] = changelog

        data = json.dumps(payload, default=str).encode()
        req  = urllib.request.Request(
            url     = f"{self.api_url}/benchmark",
            data    = data,
            headers = {
                "Content-Type": "application/json",
                "X-API-Key":    self.api_key,
            },
            method  = "POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
                if body.get("success"):
                    console.print(
                        f"[green]✓ Submitted to API[/green] — "
                        f"version: [cyan]{body.get('version_id')}[/cyan]"
                    )
                    return True
                console.print(f"[red]API rejected: {body}[/red]")
                return False
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            console.print(f"[red]API error {e.code}: {body}[/red]")
            # Fall back to file
            return self._write_file(result)

    def _write_file(self, result: RawBenchmarkResult) -> bool:
        ts    = result.timestamp.strftime("%Y%m%d_%H%M%S")
        fname = f"{result.model_id}__{result.quantization.value}__{result.hardware_profile.value}__{ts}.json"
        path  = self.output_dir / fname

        with open(path, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        console.print(f"[cyan]Saved to file: {path}[/cyan]")
        return True

    def submit_file(self, filepath: str, changelog: str = "File submission") -> bool:
        """Submit a previously saved raw JSON file to the API."""
        with open(filepath) as f:
            data = json.load(f)

        import urllib.request
        data["changelog"] = changelog
        encoded = json.dumps(data, default=str).encode()
        req = urllib.request.Request(
            url     = f"{self.api_url}/benchmark",
            data    = encoded,
            headers = {"Content-Type": "application/json", "X-API-Key": self.api_key},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
                return body.get("success", False)
        except Exception as e:
            console.print(f"[red]Submit failed: {e}[/red]")
            return False

    def submit_directory(self, raw_dir: str, changelog: str = "Batch file submission") -> dict:
        """Submit all JSON files in a directory to the API."""
        files   = list(Path(raw_dir).glob("*.json"))
        success = 0
        failed  = 0

        console.print(f"[cyan]Submitting {len(files)} files...[/cyan]")
        for f in files:
            if f.name.startswith("FAILED_"):
                continue
            if self.submit_file(str(f), changelog=changelog):
                success += 1
            else:
                failed += 1

        console.print(f"[green]Done: {success} submitted, {failed} failed[/green]")
        return {"success": success, "failed": failed}
