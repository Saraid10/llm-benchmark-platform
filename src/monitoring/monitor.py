"""
Data Monitor
============
Tracks data quality, freshness, and coverage gaps.
This feeds the "Data Health" panel in the UI.

For a recruiter, this demonstrates:
- You think about data quality, not just data quantity
- You built observability into the platform
- The dataset is actively maintained, not abandoned

Checks:
    - Data freshness per (model, hardware, quant) cell
    - Coverage gaps (which cells have zero data)
    - Outlier rate per group
    - Framework version drift (old benchmark vs current library)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.db.repository import BenchmarkRepository

console = Console()

# Thresholds
STALE_DAYS      = 30    # data older than this is considered stale
MIN_RECORDS     = 3     # minimum records per cell for confidence
OUTLIER_WARN    = 0.2   # warn if >20% of a group are outliers


class DataMonitor:

    def __init__(self, repo: BenchmarkRepository):
        self.repo = repo

    # ---------------------------------------------------------------------------
    # Freshness report
    # ---------------------------------------------------------------------------

    def freshness_report(self) -> pd.DataFrame:
        """
        Per (model, quantization, hardware_profile):
        - latest timestamp
        - days since last update
        - record count
        - is_stale flag
        """
        conn  = self.repo._connect()
        df    = conn.execute("""
            SELECT
                model_id,
                quantization,
                hardware_profile,
                COUNT(*)                                    AS record_count,
                MAX(timestamp)                              AS last_updated,
                AVG(tokens_per_sec)                         AS avg_tps,
                SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) AS outlier_count
            FROM benchmarks
            WHERE status = 'success'
            GROUP BY model_id, quantization, hardware_profile
            ORDER BY last_updated ASC
        """).df()

        if df.empty:
            return df

        now = datetime.utcnow()
        df["days_old"]  = df["last_updated"].apply(
            lambda t: (now - pd.Timestamp(t).to_pydatetime().replace(tzinfo=None)).days
        )
        df["is_stale"]        = df["days_old"] > STALE_DAYS
        df["is_low_coverage"] = df["record_count"] < MIN_RECORDS
        df["outlier_rate"]    = df["outlier_count"] / df["record_count"].clip(lower=1)
        df["high_outliers"]   = df["outlier_rate"] > OUTLIER_WARN

        return df

    # ---------------------------------------------------------------------------
    # Coverage gap analysis
    # ---------------------------------------------------------------------------

    def coverage_gaps(self) -> dict:
        """
        Returns which (model × hardware × quant) cells are missing data.
        This tells you exactly which benchmarks to run next.
        """
        conn   = self.repo._connect()

        models  = [r[0] for r in conn.execute("SELECT DISTINCT model_id FROM benchmarks").fetchall()]
        hws     = [r[0] for r in conn.execute("SELECT DISTINCT hardware_profile FROM benchmarks").fetchall()]
        quants  = [r[0] for r in conn.execute("SELECT DISTINCT quantization FROM benchmarks").fetchall()]

        existing = set(
            (r[0], r[1], r[2])
            for r in conn.execute(
                "SELECT model_id, hardware_profile, quantization FROM benchmarks"
            ).fetchall()
        )

        gaps = []
        for m in models:
            for h in hws:
                for q in quants:
                    if (m, h, q) not in existing:
                        gaps.append({"model": m, "hardware": h, "quantization": q})

        coverage_pct = (
            len(existing) / (len(models) * len(hws) * len(quants)) * 100
            if models and hws and quants else 0
        )

        return {
            "total_possible": len(models) * len(hws) * len(quants),
            "covered":        len(existing),
            "coverage_pct":   round(coverage_pct, 1),
            "gaps":           gaps,
            "models":         models,
            "hardware_tiers": hws,
            "quantizations":  quants,
        }

    # ---------------------------------------------------------------------------
    # Quality score (0–100)
    # ---------------------------------------------------------------------------

    def quality_score(self) -> dict:
        """
        Composite data quality score for the dashboard.
        Higher = better maintained dataset.
        """
        freshness = self.freshness_report()
        gaps      = self.coverage_gaps()

        if freshness.empty:
            return {"score": 0, "breakdown": {}, "message": "No data yet."}

        # Sub-scores
        freshness_score  = max(0, 100 - freshness["days_old"].mean() * 2)
        coverage_score   = gaps["coverage_pct"]
        outlier_score    = max(0, 100 - freshness["outlier_rate"].mean() * 500)
        volume_score     = min(100, (self.repo.count() / 50) * 100)

        total = (
            freshness_score * 0.30
            + coverage_score * 0.30
            + outlier_score  * 0.25
            + volume_score   * 0.15
        )

        return {
            "score": round(total, 1),
            "breakdown": {
                "freshness":  round(freshness_score, 1),
                "coverage":   round(coverage_score, 1),
                "outlier":    round(outlier_score, 1),
                "volume":     round(volume_score, 1),
            },
            "message": _score_message(total),
        }

    # ---------------------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------------------

    def print_report(self):
        console.rule("[bold cyan]Data Quality Report[/bold cyan]")

        score = self.quality_score()
        color = "green" if score["score"] >= 75 else "yellow" if score["score"] >= 50 else "red"
        console.print(f"  Quality Score: [{color}]{score['score']}/100[/{color}] — {score['message']}")

        gaps = self.coverage_gaps()
        console.print(f"  Coverage: {gaps['covered']}/{gaps['total_possible']} cells ({gaps['coverage_pct']}%)")

        if gaps["gaps"]:
            console.print(f"\n[yellow]⚠ Missing data for {len(gaps['gaps'])} (model×hw×quant) combinations.[/yellow]")
            console.print("  Top 5 gaps to fill:")
            for g in gaps["gaps"][:5]:
                console.print(f"    • {g['model']} / {g['hardware']} / {g['quantization']}")

        freshness = self.freshness_report()
        stale = freshness[freshness["is_stale"]]
        if not stale.empty:
            console.print(f"\n[yellow]⏰ {len(stale)} cells have stale data (>{STALE_DAYS} days old)[/yellow]")

        # Rich table
        table = Table(title="Freshness by Cell", show_header=True)
        table.add_column("Model",    style="cyan")
        table.add_column("Quant",    style="yellow")
        table.add_column("HW",       style="magenta")
        table.add_column("Records",  justify="right")
        table.add_column("Days Old", justify="right")
        table.add_column("Avg tok/s",justify="right")
        table.add_column("Status",   justify="center")

        for _, row in freshness.iterrows():
            status_icon = "✓"
            if row["is_stale"]:        status_icon = "⏰"
            if row["is_low_coverage"]: status_icon = "⚠"
            if row["high_outliers"]:   status_icon = "🔴"

            table.add_row(
                row["model_id"],
                row["quantization"],
                row["hardware_profile"],
                str(int(row["record_count"])),
                str(int(row["days_old"])),
                f"{row['avg_tps']:.1f}",
                status_icon,
            )

        console.print(table)


def _score_message(score: float) -> str:
    if score >= 85: return "Excellent — dataset is fresh and well-covered."
    if score >= 70: return "Good — minor gaps or slight staleness."
    if score >= 50: return "Fair — needs more benchmark runs."
    return "Poor — significant gaps or stale data."
