"""Shared dataset selection and official-source download helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from memcity.datasets.locomo import LoCoMoDataset
from memcity.datasets.longmemeval import LongMemEvalDataset
from memcity.datasets.protocol import BaseDataset, QASample
from memcity.datasets.synthetic import SyntheticDataset

DATASET_URLS = {
    "longmemeval-s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "longmemeval-oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "locomo": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
}


def get_dataset(name: str, *, seed: int = 42, limit: int | None = None) -> BaseDataset:
    normalised = name.lower().strip()
    if normalised.startswith("synthetic-"):
        return SyntheticDataset(scale=normalised.removeprefix("synthetic-"), seed=seed)
    if normalised in {"longmemeval", "longmemeval-s", "longmemeval-s-cleaned"}:
        return LongMemEvalDataset(variant="s", limit=limit)
    if normalised == "longmemeval-oracle":
        return LongMemEvalDataset(variant="oracle", limit=limit)
    if normalised == "locomo":
        return LoCoMoDataset(limit=limit)
    raise ValueError(f"Unknown dataset '{name}'")


def load_dataset(
    name: str, data_dir: str | Path, *, seed: int = 42, limit: int | None = None
) -> tuple[BaseDataset, list[QASample]]:
    dataset = get_dataset(name, seed=seed, limit=limit)
    samples = dataset.load(Path(data_dir))
    if limit is not None and name.startswith("synthetic-"):
        samples = samples[:limit]
    return dataset, samples


def fetch_dataset(name: str, data_dir: str | Path, *, variant: str = "s") -> Path:
    normalised = name.lower().strip()
    if normalised == "longmemeval":
        normalised = f"longmemeval-{variant}"
    if normalised not in DATASET_URLS:
        raise ValueError(f"No official fetch source for '{name}'")

    root = Path(data_dir)
    if normalised.startswith("longmemeval-"):
        suffix = normalised.removeprefix("longmemeval-")
        filename = (
            "longmemeval_oracle.json"
            if suffix == "oracle"
            else f"longmemeval_{suffix}_cleaned.json"
        )
        destination = root / "longmemeval" / filename
    else:
        destination = root / "locomo" / "locomo10.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(DATASET_URLS[normalised], headers={"User-Agent": "memory-city-bench/0.1"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination
