"""OpenAI-compatible LLM reader for end-to-end evaluation.

All LLM calls are cached by (model, prompt_hash).
Never used for retrieval metrics or graph construction.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

# ── Response schema ────────────────────────────────────────────────────────────


class ReaderAnswer(BaseModel):
    answer: str
    evidence_ids: list[str] = []
    abstained: bool = False
    confidence: float = 0.0


class ReaderResult(BaseModel):
    sample_id: str
    retriever: str
    answer: ReaderAnswer | None = None
    raw_response: str = ""
    schema_ok: bool = False
    schema_repaired: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


# ── Cache ──────────────────────────────────────────────────────────────────────


class ResponseCache:
    """Persistent JSON-lines cache keyed by hash(model + prompt + context)."""

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    self._data[entry["key"]] = entry

    def _make_key(self, model: str, prompt: str) -> str:
        raw = f"{model}\x00{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, prompt: str) -> dict | None:
        return self._data.get(self._make_key(model, prompt))

    def put(self, model: str, prompt: str, response: dict) -> None:
        key = self._make_key(model, prompt)
        entry = {"key": key, "model": model, "response": response}
        self._data[key] = entry
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Prompt builder ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise evidence-based answering system.
Rules:
1. Answer ONLY from the provided evidence passages.
2. Cite every passage you use by its episode_id in the evidence_ids list.
3. If the evidence is insufficient to answer, set abstained to true and answer "INSUFFICIENT_EVIDENCE".
4. Do NOT use any external knowledge or assumptions.
5. Respond with valid JSON matching this schema exactly:
   {"answer": "...", "evidence_ids": ["ep-..."], "abstained": false, "confidence": 0.0}
"""


def build_prompt(query: str, evidence: list[dict], token_budget: int) -> str:
    """Build a prompt string staying within token_budget (approximate)."""
    # A conservative word-to-token estimate avoids overflowing small local contexts.
    approx_tokens = int(len(query.split()) * 1.5) + 250
    passages: list[str] = []
    for index, ep in enumerate(evidence):
        ep_id = ep.get("id", "unknown")
        text = ep.get("text", "")
        remaining = token_budget - approx_tokens
        if remaining <= 8:
            break
        words = text.split()
        passages_left = max(1, len(evidence) - index)
        passage_token_budget = max(8, remaining // passages_left)
        word_budget = max(1, int((passage_token_budget - 4) / 1.5))
        snippet = f"[{ep_id}] {' '.join(words[:word_budget])}"
        approx_tokens += int(len(snippet.split()) * 1.5)
        passages.append(snippet)

    evidence_block = "\n".join(passages) if passages else "(no evidence provided)"
    return f"{_SYSTEM_PROMPT}\n\nEVIDENCE:\n{evidence_block}\n\nQUESTION: {query}\n\nJSON ANSWER:"


# ── JSON repair ────────────────────────────────────────────────────────────────


def _try_repair_json(raw: str) -> str | None:
    """Single deterministic repair: extract the first JSON object found."""
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        return m.group(0)
    return None


def parse_reader_response(raw: str) -> tuple[ReaderAnswer | None, bool, bool]:
    """Parse raw model text into ReaderAnswer.

    Returns (answer, schema_ok, schema_repaired).
    """
    raw = raw.strip()
    # Direct parse
    try:
        data = json.loads(raw)
        return ReaderAnswer.model_validate(data), True, False
    except (json.JSONDecodeError, ValidationError):
        pass

    # One repair attempt
    repaired = _try_repair_json(raw)
    if repaired:
        try:
            data = json.loads(repaired)
            return ReaderAnswer.model_validate(data), True, True
        except (json.JSONDecodeError, ValidationError):
            pass

    return None, False, False


# ── OpenAI-compatible client ───────────────────────────────────────────────────


class OpenAICompatibleReader:
    """Calls an OpenAI-compatible local inference server (e.g. llama.cpp)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3-0.6b",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        context_token_budget: int = 4096,
        cache_path: Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.context_token_budget = context_token_budget
        self._cache = ResponseCache(cache_path or Path("data/reader_cache.jsonl"))
        # Lazy import httpx
        self._httpx: Any = None

    def _get_httpx(self) -> Any:
        if self._httpx is None:
            try:
                import httpx

                self._httpx = httpx
            except ImportError as exc:
                raise ImportError(
                    "httpx is required for the reader. Install with: uv sync --extra readers"
                ) from exc
        return self._httpx

    def answer(
        self,
        sample_id: str,
        query: str,
        evidence: list[dict],
        retriever_name: str = "",
    ) -> ReaderResult:
        prompt = build_prompt(query, evidence, self.context_token_budget)
        cached = self._cache.get(self.model, prompt)

        t0 = time.perf_counter()
        if cached:
            raw_response = cached["response"].get("raw", "")
            prompt_tokens = cached["response"].get("prompt_tokens", 0)
            completion_tokens = cached["response"].get("completion_tokens", 0)
            latency_ms = cached["response"].get("latency_ms", 0.0)
        else:
            httpx = self._get_httpx()
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            }
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                raw_response = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            except Exception as exc:
                return ReaderResult(
                    sample_id=sample_id,
                    retriever=retriever_name,
                    error=str(exc),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            self._cache.put(
                self.model,
                prompt,
                {
                    "raw": raw_response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                },
            )

        answer, schema_ok, repaired = parse_reader_response(raw_response)
        return ReaderResult(
            sample_id=sample_id,
            retriever=retriever_name,
            answer=answer,
            raw_response=raw_response,
            schema_ok=schema_ok,
            schema_repaired=repaired,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# ── Oracle reader (retrieval ceiling) ─────────────────────────────────────────


class OracleReader:
    """Simulates a perfect reader that always uses ground-truth evidence."""

    def answer(
        self,
        sample_id: str,
        query: str,
        evidence: list[dict],
        retriever_name: str = "oracle",
        ground_truth_answer: str = "",
        evidence_ids: list[str] | None = None,
    ) -> ReaderResult:
        answer = ReaderAnswer(
            answer=ground_truth_answer or "(oracle answer)",
            evidence_ids=evidence_ids or [ep.get("id", "") for ep in evidence],
            abstained=False,
            confidence=1.0,
        )
        return ReaderResult(
            sample_id=sample_id,
            retriever=retriever_name,
            answer=answer,
            schema_ok=True,
        )
