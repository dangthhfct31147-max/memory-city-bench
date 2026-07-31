"""Utility helpers: stable hashing, IDs, seeds, and JSON I/O."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any


def stable_hash(data: str | bytes | dict | list, length: int = 16) -> str:
    """Return a stable, truncated SHA-256 hex digest."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def make_id(prefix: str = "node", text: str = "", ts: float | None = None) -> str:
    """Create a stable deterministic ID from prefix + text, or a random UUID fallback."""
    if text:
        digest = stable_hash(text, 12)
        return f"{prefix}_{digest}"
    if ts is not None:
        return f"{prefix}_{int(ts * 1000)}_{uuid.uuid4().hex[:6]}"
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def run_id() -> str:
    """Generate a time-sortable run identifier."""
    t = time.strftime("%Y%m%d_%H%M%S")
    return f"run_{t}_{uuid.uuid4().hex[:6]}"


def seed_everything(seed: int) -> None:
    """Seed Python random and numpy."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def safe_json_loads(text: str) -> dict | None:
    """Parse JSON; return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def count_tokens_approx(text: str) -> int:
    """Approximate token count (whitespace split / 0.75 ratio heuristic)."""
    return max(1, int(len(text.split()) / 0.75))


def get_project_root() -> Path:
    """Return the package root (two levels up from this file)."""
    return Path(__file__).parent.parent.parent.parent


def data_dir() -> Path:
    root = get_project_root()
    d = root / "data"
    d.mkdir(exist_ok=True)
    return d


def runs_dir() -> Path:
    root = get_project_root()
    d = root / "runs"
    d.mkdir(exist_ok=True)
    return d
