"""Deterministic synthetic Memory City benchmark dataset generator."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from memcity.datasets.protocol import (
    BaseDataset, DatasetManifest, EpisodeTurn, QASample, QuestionCategory
)
from memcity.utils.helpers import stable_hash


SCALES = {
    "tiny": {"sessions": 4, "turns_per_session": 3, "queries": 20},
    "small": {"sessions": 20, "turns_per_session": 5, "queries": 100},
    "medium": {"sessions": 100, "turns_per_session": 5, "queries": 500},
    "stress": {"sessions": 500, "turns_per_session": 8, "queries": 2000},
}


# Knowledge facts used to build episodes
TOPICS = [
    {
        "entity": "Alice",
        "facts": [
            ("Alice's favorite language", "Python"),
            ("Alice's role", "senior engineer"),
            ("Alice's project", "data pipeline"),
        ],
        "update": ("Alice's role", "tech lead"),
    },
    {
        "entity": "Bob",
        "facts": [
            ("Bob's framework", "FastAPI"),
            ("Bob's team", "platform"),
            ("Bob's location", "Berlin"),
        ],
        "update": ("Bob's location", "Amsterdam"),
    },
    {
        "entity": "Project Alpha",
        "facts": [
            ("Project Alpha's status", "in planning"),
            ("Project Alpha's deadline", "Q3 2025"),
            ("Project Alpha's lead", "Alice"),
        ],
        "update": ("Project Alpha's status", "in development"),
    },
    {
        "entity": "Database",
        "facts": [
            ("Database type", "PostgreSQL"),
            ("Database version", "14"),
            ("Database host", "db.internal"),
        ],
        "update": ("Database version", "16"),
    },
]


def _ts(base: float, offset_days: float) -> float:
    return base + offset_days * 86400.0


class SyntheticDataset(BaseDataset):
    name = "synthetic"

    def __init__(self, scale: str = "tiny", seed: int = 42):
        self.scale = scale
        self.seed = seed
        self._manifest: DatasetManifest | None = None
        self._samples: list[QASample] = []

    def load(self, data_dir: Path, **kwargs) -> list[QASample]:
        cache_path = data_dir / "synthetic" / f"{self.scale}_s{self.seed}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self._samples = [QASample.model_validate(s) for s in data["samples"]]
            self._manifest = DatasetManifest.model_validate(data["manifest"])
            return self._samples

        self._samples = self._generate()
        self._manifest = self._build_manifest()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "manifest": self._manifest.model_dump(),
                    "samples": [s.model_dump() for s in self._samples],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._samples

    def manifest(self) -> DatasetManifest:
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    def _generate(self) -> list[QASample]:
        rng = random.Random(self.seed)
        cfg = SCALES[self.scale]
        base_time = 1_700_000_000.0  # deterministic epoch

        # ── Build episode turns ──────────────────────────────────────────────
        # Each "session" covers one topic; builds initial facts, then an update
        episodes: list[EpisodeTurn] = []
        # Map: fact_key → (value, episode_id, timestamp)
        fact_log: list[dict] = []
        ep_counter = 0

        topics_used = TOPICS[: cfg["sessions"]] if cfg["sessions"] <= len(TOPICS) else (
            TOPICS * (cfg["sessions"] // len(TOPICS) + 1)
        )[: cfg["sessions"]]

        for sess_idx, topic in enumerate(topics_used):
            session_id = f"sess_{sess_idx:03d}"
            turn_offset = 0.0

            # Intro turn
            ep_id = f"ep_{ep_counter:04d}"
            ep_ts = _ts(base_time, sess_idx * 3 + turn_offset)
            episodes.append(EpisodeTurn(
                episode_id=ep_id,
                session_id=session_id,
                turn_index=0,
                user_text=f"Tell me about {topic['entity']}.",
                assistant_text=f"Sure! {topic['entity']} is a key component.",
                timestamp=ep_ts,
            ))
            ep_counter += 1

            # Fact establishment turns
            for fi, (fact_key, fact_val) in enumerate(topic["facts"]):
                ep_id = f"ep_{ep_counter:04d}"
                ep_ts = _ts(base_time, sess_idx * 3 + 0.1 * (fi + 1))
                episodes.append(EpisodeTurn(
                    episode_id=ep_id,
                    session_id=session_id,
                    turn_index=fi + 1,
                    user_text=f"What is {fact_key}?",
                    assistant_text=f"{fact_key} is {fact_val}.",
                    timestamp=ep_ts,
                ))
                fact_log.append({
                    "key": fact_key,
                    "value": fact_val,
                    "episode_id": ep_id,
                    "timestamp": ep_ts,
                    "session_id": session_id,
                    "is_stale": False,
                })
                ep_counter += 1

            # Extra turns for variety
            for extra in range(cfg["turns_per_session"] - len(topic["facts"]) - 1):
                ep_id = f"ep_{ep_counter:04d}"
                ep_ts = _ts(base_time, sess_idx * 3 + 0.5 + extra * 0.1)
                episodes.append(EpisodeTurn(
                    episode_id=ep_id,
                    session_id=session_id,
                    turn_index=len(topic["facts"]) + 1 + extra,
                    user_text=f"Any updates about {topic['entity']}?",
                    assistant_text=f"No new updates on {topic['entity']} yet.",
                    timestamp=ep_ts,
                ))
                ep_counter += 1

        # Update turns (later sessions)
        update_episodes: dict[str, list[dict]] = {}
        for sess_idx, topic in enumerate(topics_used):
            upd_key, upd_val = topic["update"]
            ep_id = f"ep_{ep_counter:04d}"
            ep_ts = _ts(base_time, sess_idx * 3 + 10)  # 10 days later
            session_id = f"sess_upd_{sess_idx:03d}"
            episodes.append(EpisodeTurn(
                episode_id=ep_id,
                session_id=session_id,
                turn_index=0,
                user_text=f"Update: {upd_key} has changed.",
                assistant_text=f"{upd_key} is now {upd_val}.",
                timestamp=ep_ts,
            ))
            # Mark stale
            for f in fact_log:
                if f["key"] == upd_key:
                    f["is_stale"] = True
            fact_log.append({
                "key": upd_key,
                "value": upd_val,
                "episode_id": ep_id,
                "timestamp": ep_ts,
                "session_id": session_id,
                "is_stale": False,
            })
            update_episodes.setdefault(upd_key, []).append({
                "episode_id": ep_id, "value": upd_val, "timestamp": ep_ts
            })
            ep_counter += 1

        # ── Build QA samples ─────────────────────────────────────────────────
        samples: list[QASample] = []
        sample_counter = 0

        def make_sample(
            query: str,
            answer: str,
            evidence_ids: list[str],
            category: QuestionCategory,
            **kwargs,
        ) -> QASample:
            nonlocal sample_counter
            sid = f"s_{sample_counter:04d}"
            sample_counter += 1
            kwargs.setdefault("required_evidence_count", len(evidence_ids))
            return QASample(
                sample_id=sid,
                query=query,
                answer=answer,
                history=episodes,
                evidence_episode_ids=evidence_ids,
                category=category,
                **kwargs,
            )

        n = cfg["queries"]
        # Current fact queries
        for topic in topics_used[:n // 4]:
            for fact_key, fact_val in topic["facts"][:1]:
                current_facts = [f for f in fact_log if f["key"] == fact_key and not f["is_stale"]]
                if not current_facts:
                    continue
                cf = current_facts[-1]
                samples.append(make_sample(
                    query=f"What is {fact_key}?",
                    answer=cf["value"],
                    evidence_ids=[cf["episode_id"]],
                    category=QuestionCategory.DIRECT_FACT,
                    expected_current_fact=cf["value"],
                ))
                if len(samples) >= n:
                    break
            if len(samples) >= n:
                break

        # Paraphrase queries
        for topic in topics_used[:n // 4]:
            for fact_key, _ in topic["facts"][:1]:
                current_facts = [f for f in fact_log if f["key"] == fact_key and not f["is_stale"]]
                if not current_facts:
                    continue
                cf = current_facts[-1]
                samples.append(make_sample(
                    query=f"Can you tell me the current value of {fact_key}?",
                    answer=cf["value"],
                    evidence_ids=[cf["episode_id"]],
                    category=QuestionCategory.PARAPHRASE,
                ))
                if len(samples) >= n:
                    break
            if len(samples) >= n:
                break

        # Fact update queries: should return NEW value not old
        for topic in topics_used:
            upd_key, upd_val = topic["update"]
            current = [f for f in fact_log if f["key"] == upd_key and not f["is_stale"]]
            stale = [f for f in fact_log if f["key"] == upd_key and f["is_stale"]]
            if not current:
                continue
            cf = current[-1]
            stale_ids = [f["episode_id"] for f in stale]
            samples.append(make_sample(
                query=f"What is the current {upd_key}?",
                answer=upd_val,
                evidence_ids=[cf["episode_id"]],
                category=QuestionCategory.FACT_UPDATE,
                expected_current_fact=upd_val,
                stale_fact_ids=stale_ids,
            ))
            if len(samples) >= n:
                break

        # Temporal: what was X before the update?
        for topic in topics_used:
            upd_key, _ = topic["update"]
            stale = [f for f in fact_log if f["key"] == upd_key and f["is_stale"]]
            if not stale:
                continue
            sf = stale[0]
            samples.append(make_sample(
                query=f"What was {upd_key} originally?",
                answer=sf["value"],
                evidence_ids=[sf["episode_id"]],
                category=QuestionCategory.TEMPORAL,
            ))
            if len(samples) >= n:
                break

        # Multi-hop: what is the project lead's language?
        if len(topics_used) >= 3:
            # Alice leads Project Alpha; Alice's language is Python
            samples.append(make_sample(
                query="What programming language does the lead of Project Alpha use?",
                answer="Python",
                evidence_ids=[
                    f["episode_id"]
                    for f in fact_log
                    if f["key"] in ("Project Alpha's lead", "Alice's favorite language")
                    and not f["is_stale"]
                ],
                category=QuestionCategory.MULTI_HOP,
                required_evidence_count=2,
            ))

        # Abstention: ask about something not in corpus
        samples.append(make_sample(
            query="What is Charlie's email address?",
            answer="INSUFFICIENT_EVIDENCE",
            evidence_ids=[],
            category=QuestionCategory.ABSTENTION,
        ))

        # Community/global: what topics are discussed?
        all_ep_ids = list({ep.episode_id for ep in episodes})[:5]
        samples.append(make_sample(
            query="What are the main topics discussed in the conversation history?",
            answer="people, projects, and technical details",
            evidence_ids=all_ep_ids,
            category=QuestionCategory.GLOBAL_COMMUNITY,
        ))

        # Distractor: lots of noise
        distractor_ep_ids = [ep.episode_id for ep in episodes if "No new updates" in ep.assistant_text]
        if distractor_ep_ids and fact_log:
            cf = fact_log[0]
            samples.append(make_sample(
                query=f"Amid all updates, what is {cf['key']}?",
                answer=cf["value"] if not cf["is_stale"] else "INSUFFICIENT_EVIDENCE",
                evidence_ids=[cf["episode_id"]],
                category=QuestionCategory.DISTRACTOR,
            ))

        return samples[:n]

    def _build_manifest(self) -> DatasetManifest:
        cat_counts: dict[str, int] = {}
        for s in self._samples:
            cat_counts[s.category.value] = cat_counts.get(s.category.value, 0) + 1
        return DatasetManifest(
            name=f"synthetic-{self.scale}",
            version="1.0",
            description=f"Deterministic synthetic dataset (scale={self.scale}, seed={self.seed})",
            scale=self.scale,
            seed=self.seed,
            sample_count=len(self._samples),
            category_counts=cat_counts,
            provenance="generated",
        )
