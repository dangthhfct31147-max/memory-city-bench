"""Resource and latency instrumentation."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceSnapshot:
    timestamp: float = 0.0
    rss_mb: float = 0.0
    cpu_percent: float = 0.0
    vram_mb: float = 0.0


@dataclass
class LatencyStats:
    samples: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def _pct(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    @property
    def p50(self) -> float:
        return self._pct(50)

    @property
    def p90(self) -> float:
        return self._pct(90)

    @property
    def p95(self) -> float:
        return self._pct(95)

    @property
    def p99(self) -> float:
        return self._pct(99)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def qps(self) -> float:
        if not self.samples:
            return 0.0
        total_s = sum(self.samples) / 1000.0
        return len(self.samples) / total_s if total_s > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "count": len(self.samples),
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.p50, 2),
            "p90_ms": round(self.p90, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "qps": round(self.qps, 2),
        }


def get_process_rss_mb() -> float:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_vram_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        return 0.0


def collect_environment() -> dict[str, Any]:
    """Gather reproducibility metadata for a run."""
    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        env["ram_total_gb"] = round(vm.total / 1024**3, 1)
        env["ram_available_gb"] = round(vm.available / 1024**3, 1)
    except Exception:
        pass

    # GPU
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({
                "name": name if isinstance(name, str) else name.decode(),
                "vram_total_gb": round(mem.total / 1024**3, 1),
            })
        env["gpus"] = gpus
    except Exception:
        env["gpus"] = []

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        env["git_commit"] = result.stdout.strip()
    except Exception:
        env["git_commit"] = "unknown"

    # Package versions
    packages = ["pydantic", "rank_bm25", "networkx", "numpy", "scikit-learn",
                "sentence_transformers", "rich", "typer", "psutil"]
    installed: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata
            installed[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except Exception:
            installed[pkg] = "not_installed"
    env["packages"] = installed

    return env
