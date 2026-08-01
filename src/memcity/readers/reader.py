"""OpenAI-compatible LLM reader for end-to-end evaluation.

All LLM calls are cached by a composite key covering every field that can change
the response (base_url, model, temperature, seed, max_tokens, prompt, and a hash
of the relevant request-header configuration).

The reader optionally talks to an OmniRoute-style gateway. That support is
entirely opt-in: retrieval-only benchmarks never construct this reader, and even
end-to-end runs default to the offline OracleReader. When a gateway is used the
reader:
  * sends ``Authorization: Bearer <api key>`` (key read from the environment,
    never committed or printed);
  * can forward arbitrary extra headers (e.g. gateway routing controls);
  * sends ``X-OmniRoute-No-Cache: true`` for benchmark runs so the gateway does
    not serve a stale cached generation;
  * records the requested model alongside the effective model/provider the
    gateway actually served, and flags any fallback/provider switch as a warning
    so published runs can be audited for reproducibility.

Never used for retrieval metrics or graph construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

# Environment-variable names for the optional gateway configuration.
ENV_BASE_URL = "MEMCITY_LLM_BASE_URL"
ENV_API_KEY = "MEMCITY_LLM_API_KEY"
ENV_MODEL = "MEMCITY_LLM_MODEL"

# Defaults used only when the corresponding env var is unset. The model has no
# safe default (it must be an explicit pinned id for reproducibility), so callers
# are expected to pin it; this fallback exists purely so construction never
# crashes in offline unit tests.
_DEFAULT_BASE_URL = "http://localhost:20128/v1"
_DEFAULT_MODEL = "qwen3-0.6b"

# Header that instructs an OmniRoute-style gateway to bypass its own response
# cache for this request (benchmark correctness).
GATEWAY_NO_CACHE_HEADER = "X-OmniRoute-No-Cache"

# Local response-cache modes.
#   read-write : read hits, write misses (default; dev smoke tests)
#   read-only  : read hits, never write (frozen cache)
#   off        : never read, never write (mandatory for latency benchmarks)
_CACHE_MODES = frozenset({"read-write", "read-only", "off"})


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
    # Reproducibility / provenance fields (populated for gateway calls).
    requested_model: str = ""
    effective_model: str = ""
    effective_provider: str = ""
    fallback_warning: bool = False
    # Cache / latency provenance. upstream_latency_ms is the real network round
    # trip (0.0 on a local cache hit); local_lookup_latency_ms is the time spent
    # reading the local cache. They must never be mixed in latency stats.
    cache_hit: bool = False
    upstream_latency_ms: float = 0.0
    local_lookup_latency_ms: float = 0.0
    # ── Phase 5: KV / prefix-cache metrics ────────────────────────────────────
    # All fields are "unknown" (None) when the backend does not return the data,
    # so callers must never assume 0 means "no cache hit" vs "not reported".
    # prompt_eval_ms  — time the backend spent processing the prompt tokens
    #                   (cold = full prefill, warm ≈ 0 on a KV-cache hit).
    # decode_ms       — time the backend spent generating completion tokens.
    # cached_prompt_tokens — tokens served from the KV cache (OpenAI: usage
    #                        .prompt_tokens_details.cached_tokens; llama.cpp:
    #                        timings.prompt_n_cached or similar).
    # prefix_cache_hit — True when cached_prompt_tokens > 0, False when 0 and
    #                    reported, None when not reported by the backend.
    reader_backend: str = ""
    prefix_cache_enabled: bool | None = None
    prefix_cache_hit: bool | None = None
    cached_prompt_tokens: int | None = None
    prompt_eval_ms: float | None = None
    decode_ms: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    # SHA-256 (hex) of the stable prompt prefix (system prompt + schema +
    # instructions). Identical across queries of the same run; enables auditing
    # that the prefix was not changed mid-run.
    prompt_prefix_hash: str = ""


# ── Cache ──────────────────────────────────────────────────────────────────────


class ResponseCache:
    """Persistent JSON-lines cache keyed by a composite of request-defining fields.

    The key is a SHA-256 over the JSON-serialised ``key_fields`` dict. Callers
    supply every field that can change the model output: base_url, model,
    temperature, seed, max_tokens, prompt, and a non-secret hash of the header
    configuration. The API key itself is NEVER part of the key fields — only a
    boolean "auth enabled" flag folded into the header hash — so the cache file
    can never leak a credential.
    """

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

    @staticmethod
    def make_key(key_fields: dict[str, Any]) -> str:
        raw = json.dumps(key_fields, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key_fields: dict[str, Any]) -> dict | None:
        return self._data.get(self.make_key(key_fields))

    def put(self, key_fields: dict[str, Any], response: dict) -> None:
        key = self.make_key(key_fields)
        # Store the model for human readability but never the api key or headers.
        entry = {"key": key, "model": key_fields.get("model", ""), "response": response}
        self._data[key] = entry
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def header_config_hash(
    extra_headers: dict[str, str] | None,
    auth_enabled: bool,
    gateway_no_cache: bool,
) -> str:
    """Deterministic, secret-free hash of the request header configuration.

    Includes the extra headers verbatim (these are routing controls, not
    secrets), plus whether auth is enabled and whether the gateway cache is
    bypassed. The API key value is deliberately excluded.
    """
    payload = {
        "extra_headers": dict(sorted((extra_headers or {}).items())),
        "auth_enabled": bool(auth_enabled),
        "gateway_no_cache": bool(gateway_no_cache),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Prompt builder ─────────────────────────────────────────────────────────────

# Stable prefix: system instructions + JSON schema.  These sections are fixed
# for the lifetime of a run and are placed first so backends with KV-cache
# reuse can cache them across queries.  The prefix hash lets us audit that no
# part of this section changed mid-run.
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

# SHA-256 of the stable prefix, computed once at import time.
PROMPT_PREFIX_HASH: str = hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest()


def build_prompt(query: str, evidence: list[dict], token_budget: int) -> str:
    """Build a prompt string staying within token_budget (approximate).

    Section order is stable for KV-cache reuse:
      1. System instructions + schema  (stable — same hash every run)
      2. Evidence passages             (variable per query)
      3. Question                      (variable per query)
    """
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


def _extract_cache_metrics(usage: dict) -> dict[str, int | float | bool | None]:
    """Parse backend-specific KV/prefix-cache fields from the ``usage`` object.

    Returns a dict with keys: ``cached_prompt_tokens``, ``prompt_eval_ms``,
    ``decode_ms``, ``prompt_tokens_per_second``, ``generation_tokens_per_second``,
    ``prefix_cache_hit``.  All values are None when the backend did not report
    them; callers MUST NOT treat None as zero.

    Supported backends:
    * OpenAI-compatible (``usage.prompt_tokens_details.cached_tokens``)
    * llama.cpp HTTP server (``usage.timings.*``)
    """
    cached: int | None = None
    prompt_eval_ms: float | None = None
    decode_ms_val: float | None = None
    prompt_tps: float | None = None
    gen_tps: float | None = None

    # OpenAI-style: usage.prompt_tokens_details.cached_tokens
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        raw = details.get("cached_tokens")
        if raw is not None:
            cached = int(raw)

    # llama.cpp HTTP server: usage.timings (non-standard extension)
    timings = usage.get("timings")
    if isinstance(timings, dict):
        # prompt_eval_ms / prompt_n_cached
        # Prefer the first key; fall back to the alias only when first is absent.
        p_ms = timings.get("prompt_ms")
        if p_ms is None:
            p_ms = timings.get("prompt_eval_ms")
        if p_ms is not None:
            prompt_eval_ms = float(p_ms)

        p_cached = timings.get("prompt_n_cached")
        if p_cached is None:
            p_cached = timings.get("cached_tokens")
        if p_cached is not None and cached is None:
            cached = int(p_cached)

        # decode / generation timing
        d_ms = timings.get("predicted_ms")
        if d_ms is None:
            d_ms = timings.get("eval_ms")
        if d_ms is not None:
            decode_ms_val = float(d_ms)

        # tokens/s
        p_tps = timings.get("prompt_per_second")
        if p_tps is None:
            p_tps = timings.get("prompt_tps")
        if p_tps is not None:
            prompt_tps = float(p_tps)

        g_tps = timings.get("predicted_per_second")
        if g_tps is None:
            g_tps = timings.get("gen_tps")
        if g_tps is not None:
            gen_tps = float(g_tps)

    # Derive prefix_cache_hit from cached tokens when reported
    prefix_hit: bool | None
    if cached is not None:
        prefix_hit = cached > 0
    else:
        prefix_hit = None

    return {
        "cached_prompt_tokens": cached,
        "prompt_eval_ms": prompt_eval_ms,
        "decode_ms": decode_ms_val,
        "prompt_tokens_per_second": prompt_tps,
        "generation_tokens_per_second": gen_tps,
        "prefix_cache_hit": prefix_hit,
    }


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
    """Calls an OpenAI-compatible inference server or OmniRoute-style gateway.

    Works unchanged against a plain local server (llama.cpp, vLLM, …). Gateway
    features (auth header, extra headers, no-cache, model/provider auditing) are
    only exercised when the corresponding arguments are supplied, so the plain
    local path stays exactly as before.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        context_token_budget: int = 4096,
        cache_path: Path | None = None,
        *,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        gateway_no_cache: bool = False,
        cache_mode: str = "read-write",
        omniroute_provider: str | None = None,
        transport: Any = None,
    ) -> None:
        # Resolve base_url/model from the environment when not passed explicitly.
        resolved_base = base_url or os.environ.get(ENV_BASE_URL) or _DEFAULT_BASE_URL
        resolved_model = model or os.environ.get(ENV_MODEL) or _DEFAULT_MODEL

        self.base_url = resolved_base.rstrip("/")
        self.model = resolved_model
        self.requested_model = resolved_model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.context_token_budget = context_token_budget
        self._cache = ResponseCache(cache_path or Path("data/reader_cache.jsonl"))

        if cache_mode not in _CACHE_MODES:
            raise ValueError(
                f"cache_mode must be one of {sorted(_CACHE_MODES)}, got {cache_mode!r}"
            )
        self.cache_mode = cache_mode

        # OmniRoute provider-locked route: when set, chat completions go to
        # /v1/providers/<id>/chat/completions, which returns 400 (rather than
        # silently falling back) if the model is not served by that provider.
        # /models health check always stays on the base URL.
        self.omniroute_provider = omniroute_provider
        self.models_url = f"{self.base_url}/models"
        if omniroute_provider:
            self.chat_completions_url = (
                f"{self.base_url}/providers/{omniroute_provider}/chat/completions"
            )
        else:
            self.chat_completions_url = f"{self.base_url}/chat/completions"

        # Secret handling: the key is stored on the instance only to build the
        # Authorization header. It is never placed in the cache key, cache file,
        # trace, repr, or error messages.
        self._api_key = api_key
        self._extra_headers = dict(extra_headers or {})
        self._gateway_no_cache = gateway_no_cache
        self._transport = transport
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

    def _build_headers(self) -> dict[str, str]:
        """Assemble request headers, adding auth and no-cache when configured."""
        headers = dict(self._extra_headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._gateway_no_cache:
            headers[GATEWAY_NO_CACHE_HEADER] = "true"
        return headers

    def _client(self) -> Any:
        httpx = self._get_httpx()
        if self._transport is not None:
            return httpx.Client(timeout=self.timeout_s, transport=self._transport)
        return httpx.Client(timeout=self.timeout_s)

    def _redact(self, message: str) -> str:
        """Strip the API key from any string before it can be surfaced."""
        if self._api_key:
            return message.replace(self._api_key, "***REDACTED***")
        return message

    def _cache_key_fields(self, prompt: str) -> dict[str, Any]:
        """Every field that can change the generation (review requirement #4)."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
            "header_config_hash": header_config_hash(
                self._extra_headers,
                auth_enabled=bool(self._api_key),
                gateway_no_cache=self._gateway_no_cache,
            ),
        }

    def health_check(self) -> dict[str, Any]:
        """GET /models to verify the endpoint is reachable and the model is served.

        Returns ``{"ok": bool, "models": [...], "model_available": bool, "error": str}``.
        Never raises and never leaks the API key.
        """
        try:
            with self._client() as client:
                resp = client.get(self.models_url, headers=self._build_headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return {"ok": False, "models": [], "model_available": False,
                    "error": self._redact(str(exc))}
        items = data.get("data", data) if isinstance(data, dict) else data
        model_ids = [
            m.get("id", "") for m in items if isinstance(m, dict)
        ] if isinstance(items, list) else []
        return {
            "ok": True,
            "models": model_ids,
            "model_available": self.model in model_ids,
            "error": "",
        }

    def answer(
        self,
        sample_id: str,
        query: str,
        evidence: list[dict],
        retriever_name: str = "",
    ) -> ReaderResult:
        prompt = build_prompt(query, evidence, self.context_token_budget)
        key_fields = self._cache_key_fields(prompt)

        # Local cache read (only when the mode permits it). We measure the lookup
        # time separately so it is never conflated with upstream latency.
        cache_hit = False
        cache_hit_latency_ms = 0.0
        cached = None
        if self.cache_mode in ("read-write", "read-only"):
            t_lookup = time.perf_counter()
            cached = self._cache.get(key_fields)
            cache_hit_latency_ms = (time.perf_counter() - t_lookup) * 1000

        effective_model = self.model
        effective_provider = ""
        # Phase 5: KV/prefix-cache metrics (all None = "not reported by backend")
        cache_metrics: dict[str, Any] = {}

        if cached:
            cache_hit = True
            resp_data = cached["response"]
            raw_response = resp_data.get("raw", "")
            prompt_tokens = resp_data.get("prompt_tokens", 0)
            completion_tokens = resp_data.get("completion_tokens", 0)
            # A cache hit performs no network call: upstream latency is 0 and the
            # reported latency is the local lookup, not a stale prior round trip.
            upstream_latency_ms = 0.0
            local_lookup_latency_ms = cache_hit_latency_ms
            latency_ms = local_lookup_latency_ms
            effective_model = resp_data.get("effective_model", self.model)
            effective_provider = resp_data.get("effective_provider", "")
            # Restore persisted cache metrics when available.
            cache_metrics = resp_data.get("cache_metrics") or {}
        else:
            # Ensure httpx is importable (raises a helpful error otherwise).
            self._get_httpx()
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            }
            t_upstream = time.perf_counter()
            try:
                with self._client() as client:
                    resp = client.post(
                        self.chat_completions_url,
                        json=payload,
                        headers=self._build_headers(),
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Provider is surfaced via a response header by some
                    # OmniRoute-style gateways, but it is NOT part of the
                    # documented public header set, so treat it as best-effort:
                    # fall back to the pinned provider-locked route id, else "".
                    effective_provider = (
                        resp.headers.get("X-OmniRoute-Provider", "")
                        or (self.omniroute_provider or "")
                    )
                raw_response = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                # Phase 5: extract KV-cache and timing fields from the usage object.
                # Values are None when the backend does not report them.
                cache_metrics = _extract_cache_metrics(usage)
                # The gateway echoes the model it actually served; a mismatch with
                # the requested model means a fallback/switch occurred.
                effective_model = data.get("model", self.model) or self.model
            except Exception as exc:
                upstream_latency_ms = (time.perf_counter() - t_upstream) * 1000
                return ReaderResult(
                    sample_id=sample_id,
                    retriever=retriever_name,
                    error=self._redact(str(exc)),
                    latency_ms=upstream_latency_ms,
                    requested_model=self.requested_model,
                    cache_hit=False,
                    upstream_latency_ms=upstream_latency_ms,
                    local_lookup_latency_ms=cache_hit_latency_ms,
                    prompt_prefix_hash=PROMPT_PREFIX_HASH,
                )
            upstream_latency_ms = (time.perf_counter() - t_upstream) * 1000
            local_lookup_latency_ms = cache_hit_latency_ms
            latency_ms = upstream_latency_ms
            if self.cache_mode == "read-write":
                self._cache.put(
                    key_fields,
                    {
                        "raw": raw_response,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_ms": upstream_latency_ms,
                        "effective_model": effective_model,
                        "effective_provider": effective_provider,
                        "cache_metrics": cache_metrics,
                    },
                )

        fallback_warning = (
            effective_model != "" and effective_model != self.requested_model
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
            requested_model=self.requested_model,
            effective_model=effective_model,
            effective_provider=effective_provider,
            fallback_warning=fallback_warning,
            cache_hit=cache_hit,
            upstream_latency_ms=upstream_latency_ms,
            local_lookup_latency_ms=local_lookup_latency_ms,
            # Phase 5 fields
            prefix_cache_hit=cache_metrics.get("prefix_cache_hit"),
            cached_prompt_tokens=cache_metrics.get("cached_prompt_tokens"),
            prompt_eval_ms=cache_metrics.get("prompt_eval_ms"),
            decode_ms=cache_metrics.get("decode_ms"),
            prompt_tokens_per_second=cache_metrics.get("prompt_tokens_per_second"),
            generation_tokens_per_second=cache_metrics.get("generation_tokens_per_second"),
            prompt_prefix_hash=PROMPT_PREFIX_HASH,
        )

    def __repr__(self) -> str:
        # Never expose the api key.
        return (
            f"OpenAICompatibleReader(base_url={self.base_url!r}, model={self.model!r}, "
            f"provider={self.omniroute_provider!r}, cache_mode={self.cache_mode!r}, "
            f"auth={'yes' if self._api_key else 'no'})"
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
