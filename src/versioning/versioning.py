"""
Dataset Versioning System
==========================
Every time new benchmark data is ingested, a new dataset version
is created — immutable, timestamped, with a full changelog.

Why this matters for a recruiter-facing project:
- Shows you understand data engineering, not just ML
- Makes the dataset citable and reproducible
- Lets the UI show "data freshness" per hardware tier
- Enables rollback if bad data is ingested

Version format: YYYY.MM.DD.N  (e.g. 2026.04.09.1)
Each version is a manifest JSON pointing to the records it contains.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()

VERSIONS_DIR = Path("data/versions")
VERSION_FILE  = Path("data/current_version.json")


# ---------------------------------------------------------------------------
# Version manifest schema
# ---------------------------------------------------------------------------

class DatasetVersion(BaseModel):
    version_id:      str                        # e.g. "2026.04.09.1"
    created_at:      datetime
    record_count:    int
    model_ids:       list[str]
    hardware_profiles: list[str]
    quantizations:   list[str]
    changelog:       str                        # what changed in this version
    prompt_hash:     str                        # hash of prompt set used
    data_hash:       str                        # SHA256 of all record IDs
    parent_version:  Optional[str]   = None     # previous version ID
    schema_version:  int             = 1
    is_stable:       bool            = True


class VersionRegistry:
    """
    Manages the history of dataset versions.
    Reads/writes JSON manifests to data/versions/.
    """

    def __init__(self, versions_dir: str = str(VERSIONS_DIR)):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        record_count:      int,
        model_ids:         list[str],
        hardware_profiles: list[str],
        quantizations:     list[str],
        changelog:         str,
        prompt_hash:       str,
        record_ids:        list[str],
    ) -> DatasetVersion:
        """
        Create a new immutable version manifest.
        Called after every successful batch ingest.
        """
        parent = self._get_current_version_id()
        version_id = self._next_version_id()

        # Deterministic hash of all record IDs in this version
        data_hash = hashlib.sha256(
            "|".join(sorted(record_ids)).encode()
        ).hexdigest()[:16]

        version = DatasetVersion(
            version_id        = version_id,
            created_at        = datetime.utcnow(),
            record_count      = record_count,
            model_ids         = sorted(set(model_ids)),
            hardware_profiles = sorted(set(hardware_profiles)),
            quantizations     = sorted(set(quantizations)),
            changelog         = changelog,
            prompt_hash       = prompt_hash,
            data_hash         = data_hash,
            parent_version    = parent,
        )

        # Write manifest
        manifest_path = self.versions_dir / f"{version_id}.json"
        with open(manifest_path, "w") as f:
            json.dump(version.model_dump(mode="json"), f, indent=2, default=str)

        # Update current pointer
        with open(VERSION_FILE, "w") as f:
            json.dump({"current": version_id, "updated_at": datetime.utcnow().isoformat()}, f)

        console.print(f"[green]✓ Version created: [bold]{version_id}[/bold][/green]")
        console.print(f"  Records   : {record_count}")
        console.print(f"  Data hash : {data_hash}")
        console.print(f"  Parent    : {parent or 'none (initial)'}")

        return version

    def get_current(self) -> Optional[DatasetVersion]:
        """Return the current (latest) version manifest."""
        vid = self._get_current_version_id()
        if not vid:
            return None
        return self.get_version(vid)

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        path = self.versions_dir / f"{version_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return DatasetVersion(**json.load(f))

    def list_versions(self) -> list[DatasetVersion]:
        """Return all versions, newest first."""
        versions = []
        for f in sorted(self.versions_dir.glob("*.json"), reverse=True):
            try:
                with open(f) as fp:
                    versions.append(DatasetVersion(**json.load(fp)))
            except Exception:
                pass
        return versions

    def _get_current_version_id(self) -> Optional[str]:
        if not VERSION_FILE.exists():
            return None
        with open(VERSION_FILE) as f:
            return json.load(f).get("current")

    def _next_version_id(self) -> str:
        today    = datetime.utcnow().strftime("%Y.%m.%d")
        existing = [
            v.version_id for v in self.list_versions()
            if v.version_id.startswith(today)
        ]
        n = len(existing) + 1
        return f"{today}.{n}"

    def format_changelog_table(self) -> str:
        """Markdown table of all versions — for the UI."""
        versions = self.list_versions()
        if not versions:
            return "_No versions yet._"

        lines = [
            "| Version | Date | Records | Models | Changelog |",
            "|---|---|---|---|---|",
        ]
        for v in versions[:10]:   # last 10
            date  = v.created_at.strftime("%Y-%m-%d %H:%M")
            mods  = ", ".join(v.model_ids[:3])
            if len(v.model_ids) > 3:
                mods += f" +{len(v.model_ids)-3}"
            lines.append(
                f"| `{v.version_id}` | {date} | {v.record_count} "
                f"| {mods} | {v.changelog[:60]} |"
            )
        return "\n".join(lines)
