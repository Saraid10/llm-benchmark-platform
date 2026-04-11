"""
Hardware Abstraction Layer.

Problem: two users both say "I have 16GB RAM" but one has a laptop
with a slow DDR4 single-channel and the other has a desktop with
fast DDR5. Raw specs don't tell the whole story.

Solution: map to standardized capability tiers that align with
what the benchmark data was collected on. This is the right level
of abstraction for a comparison tool.
"""

from __future__ import annotations
import platform
import psutil
from typing import Optional

from src.core.models import HardwareProfile, HardwareSpec


# ---------------------------------------------------------------------------
# Tier definitions — must match benchmark collection environments exactly
# ---------------------------------------------------------------------------

TIER_DEFINITIONS = {
    HardwareProfile.CPU_LOW: {
        "label":       "CPU — Low (≤8 GB RAM)",
        "description": "Entry-level laptops, older machines, budget VMs.",
        "ram_max_gb":  8,
        "has_gpu":     False,
    },
    HardwareProfile.CPU_MEDIUM: {
        "label":       "CPU — Medium (16 GB RAM)",
        "description": "Standard developer laptops, mid-range desktops.",
        "ram_max_gb":  16,
        "has_gpu":     False,
    },
    HardwareProfile.CPU_HIGH: {
        "label":       "CPU — High (32+ GB RAM)",
        "description": "High-end workstations, Mac Studio, server-grade CPUs.",
        "ram_max_gb":  float("inf"),
        "has_gpu":     False,
    },
    HardwareProfile.GPU_T4: {
        "label":       "GPU — NVIDIA T4 (16 GB VRAM)",
        "description": "Google Colab free tier, GCP n1 instances.",
        "vram_gb":     16,
    },
    HardwareProfile.GPU_A10: {
        "label":       "GPU — NVIDIA A10 (24 GB VRAM)",
        "description": "Colab Pro+, Lambda Labs, RunPod.",
        "vram_gb":     24,
    },
    HardwareProfile.EDGE: {
        "label":       "Edge — NVIDIA Jetson / ARM",
        "description": "Jetson Nano, Orin, Raspberry Pi 5 class devices.",
    },
}


class HardwareMapper:
    """
    Maps a HardwareSpec (raw user input or auto-detected)
    to a standardized HardwareProfile tier.

    Design decision: when in doubt, map DOWN to the lower tier.
    It's better to show slightly pessimistic estimates than to
    set expectations the hardware can't meet.
    """

    def map(self, spec: HardwareSpec) -> HardwareProfile:
        """Primary mapping function."""

        if spec.is_edge:
            return HardwareProfile.EDGE

        if spec.has_gpu and spec.gpu_vram_gb is not None:
            return self._map_gpu(spec.gpu_vram_gb, spec.gpu_name)

        return self._map_cpu(spec.ram_gb)

    def _map_gpu(self, vram_gb: float, gpu_name: Optional[str]) -> HardwareProfile:
        # Named GPU overrides — most reliable signal
        if gpu_name:
            name_upper = gpu_name.upper()
            if "T4" in name_upper:
                return HardwareProfile.GPU_T4
            if "A10" in name_upper:
                return HardwareProfile.GPU_A10

        # Fall back to VRAM thresholds
        if vram_gb <= 16:
            return HardwareProfile.GPU_T4
        return HardwareProfile.GPU_A10

    def _map_cpu(self, ram_gb: float) -> HardwareProfile:
        if ram_gb <= 8:
            return HardwareProfile.CPU_LOW
        if ram_gb <= 20:          # 16GB + small headroom
            return HardwareProfile.CPU_MEDIUM
        return HardwareProfile.CPU_HIGH

    def get_tier_info(self, profile: HardwareProfile) -> dict:
        """Returns human-readable description of a tier."""
        return TIER_DEFINITIONS.get(profile, {})

    def list_all_tiers(self) -> list[dict]:
        """For UI dropdowns."""
        return [
            {"profile": p.value, **info}
            for p, info in TIER_DEFINITIONS.items()
        ]


def autodetect_hardware() -> HardwareSpec:
    """
    Detect hardware specs of the current machine.
    Used by workers to self-tag their benchmark results.

    Note: This runs SERVER-SIDE. In the Gradio UI, we use
    manual selection for the user's machine instead.
    """
    import subprocess

    ram_gb    = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False) or 1

    has_gpu      = False
    gpu_vram_gb  = None
    gpu_name     = None
    cuda_version = None

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts      = lines[0].split(",")
                gpu_name   = parts[0].strip()
                gpu_vram_gb = float(parts[1].strip()) / 1024   # MiB → GiB
                has_gpu    = True

        # cuda version
        ver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if ver_result.returncode == 0:
            cuda_version = ver_result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass    # No GPU — that's fine

    # Jetson detection
    is_edge = False
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_name = f.read()
            if "Jetson" in model_name or "Raspberry" in model_name:
                is_edge = True
    except FileNotFoundError:
        pass

    return HardwareSpec(
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        has_gpu=has_gpu,
        gpu_vram_gb=round(gpu_vram_gb, 1) if gpu_vram_gb else None,
        gpu_name=gpu_name,
        is_edge=is_edge,
    )
