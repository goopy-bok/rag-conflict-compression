"""
LLM client for CCD-G1 (local Ollama or OpenRouter, per model `provider` in configs/models.yaml), three modes:

- mock     : deterministic offline answers from mock_llm.respond. No cache read/write, no network.
- dry_run  : cache hit -> cached text; miss -> no call, the call is recorded as planned and
             ("", {"dry_run": True, ...}) is returned. Used for gate cost estimates.
- execute  : real OpenRouter call. Needs OPENROUTER_API_KEY and max_cost_usd; raises
             BudgetExceeded once actual spend passes max_cost_usd.

Caches
- cache/llm_cache.jsonl (append-only). key = sha256(json[model, system, prompt, temperature, max_tokens]).
- cache/llm_cache.sqlite (pilot, READ-ONLY, opened immutable). key = sha256(f"{model}:::{system}:::{prompt}:::{temperature:.4f}"),
  max_tokens not part of the key -> hits are marked meta["legacy_cache"]=True. Model aliases from
  configs/models.yaml legacy_aliases are tried in both directions.

Cost: prices in configs/models.yaml are USD per 1M tokens. 'TO_CONFIRM' -> cost None in dry_run and
execute refuses the model. Spend uses OpenRouter's usage.cost when returned, else price x tokens.
Structure ported from v0.0.1-Confilt-Agent-RAG/experiments/llm_api.py (chat, chat_many).

Ollama (provider: ollama; preregistration 2026-10-09 sections 2-3)
- POST {url}/api/chat, stream=false, options {num_ctx (default 8192), seed, temperature, num_predict=max_tokens};
  "format": "json" when the model is assigned to a role with role_options.<role>.json_mode (judge roles).
  A model that also holds a non-JSON role gets no JSON format (warning printed).
- Price 0 by definition (never TO_CONFIRM); no API key; the cost cap is not checked, so --max-cost 0 works.
- Cache key = the OpenRouter key fields + provider options (num_ctx, json_mode, seed); OpenRouter keys unchanged.
- execute: GET {url}/api/tags once -> model digests in meta["model_digest"] and results/models_manifest.json
  (next to the cost log). meta["ollama"] = prompt_eval_count, eval_count, total_duration_ns, done_reason, num_ctx.
- Concurrency per provider: ollama calls are serialized by a semaphore (ollama.concurrency, default 1).
  Timeout 900 s; connection errors / HTTP 5xx retried 3 times.

Splash (provider: splash; preregistration 2.1, incoai/Qwen3.8-27B-Splash served by `splash serve`)
- POST {base_url}/chat/completions (OpenAI compatible; base_url default http://127.0.0.1:8010/v1, per model in
  configs/models.yaml) with model, messages, temperature, max_tokens, seed, reasoning_effort ("none" by default;
  Splash's frontend validates it against none|minimal|...|max and falls back to --default-reasoning-effort when
  absent; set `reasoning_effort: null` on the model to omit the field), and response_format {"type": "json_object"}
  for JSON (judge) roles, with the same judge/non-judge conflict rule as Ollama.
- meta["splash"] = prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens (server usage), finish_reason.
- Price 0, --max-cost 0 works; cache key = OpenRouter fields + provider options (reasoning_effort, json_mode, seed).
- execute: model_digest() = artifact_set_sha256 of the Splash manifest (manifest_path glob, must match exactly one
  file) after checking GET {base_url}/models serves the model id; both go to results/models_manifest.json.
- Calls are serialized (splash concurrency, default 1). Same timeout / retry rule as Ollama.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from budget import count_tokens  # noqa: E402

MODELS_CONFIG = ROOT_DIR / "configs" / "models.yaml"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
UNPRICED = "TO_CONFIRM"
MAX_ATTEMPTS = 5
SAME_AS_COMPRESSOR = "same_as_compressor"
OLLAMA_DEFAULTS = {"url": "http://localhost:11434", "num_ctx": 8192, "seed": 0, "concurrency": 1,
                   "timeout_s": 900, "max_retries": 3}
OLLAMA_REPEAT_ABORT = "token repeat limit reached"  # Ollama >= 0.40 HTTP 500 body for a degenerate repetition
SPLASH_DEFAULTS = {"base_url": "http://127.0.0.1:8010/v1", "reasoning_effort": "none", "seed": 0, "concurrency": 1,
                   "timeout_s": 900, "max_retries": 3, "manifest_path": None}
LOCAL_PROVIDERS = ("ollama", "splash")


class BudgetExceeded(RuntimeError):
    pass


class RetryableHTTPError(RuntimeError):
    pass


def resolve_roles(roles: dict | None) -> dict:
    """'same_as_compressor' -> the compressor model id."""
    roles = dict(roles or {})
    return {k: roles.get("compressor") if v == SAME_AS_COMPRESSOR else v for k, v in roles.items()}


def load_models_config(path: Path | None = None) -> dict:
    """Parsed models.yaml with roles resolved (resolve_roles). Read the file directly for raw roles."""
    if path is None:
        env_cfg = os.environ.get("CCD_MODELS_CONFIG")
        path = Path(env_cfg) if env_cfg else MODELS_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["roles"] = resolve_roles(cfg.get("roles"))
    return cfg


def cache_key(model: str, system: str, prompt: str, temperature: float, max_tokens: int,
              salt: str | None = None, provider_opts: dict | None = None) -> str:
    """salt (e.g. a DSPy rollout_id) and provider_opts (Ollama only) are appended only when given,
    so OpenRouter keys are unchanged."""
    parts = [model, system, prompt, float(temperature), int(max_tokens)] + ([salt] if salt is not None else [])
    if provider_opts is not None:
        parts.append(json.dumps(provider_opts, sort_keys=True))
    blob = json.dumps(parts, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def local_transport(url: str, payload: dict | None, timeout: float = OLLAMA_DEFAULTS["timeout_s"],
                    label: str = "Ollama") -> dict:
    """One HTTP request to a local server (Ollama or Splash; GET when payload is None). Connection errors and
    HTTP 5xx -> RetryableHTTPError; a timeout is raised as TimeoutError (not retried)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        if e.code >= 500:
            raise RetryableHTTPError(f"{label} HTTP {e.code}: {body}") from e
        raise RuntimeError(f"{label} HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise TimeoutError(f"{label} request timed out after {timeout}s") from e
        raise RetryableHTTPError(f"{label} connection error: {e.reason}") from e
    except ConnectionError as e:
        raise RetryableHTTPError(f"{label} connection error: {e}") from e


def read_splash_manifest(pattern: str) -> tuple[str, str]:
    """(path, artifact_set_sha256) of the Splash package manifest. The glob must match exactly one file
    (several snapshots -> which one is served is ambiguous, so refuse)."""
    paths = sorted(glob.glob(os.path.expanduser(pattern)))
    if len(paths) != 1:
        raise RuntimeError(f"Splash manifest pattern {pattern!r} matched {len(paths)} files: {paths}")
    sha = json.loads(Path(paths[0]).read_text(encoding="utf-8")).get("artifact_set_sha256")
    if not sha:
        raise RuntimeError(f"{paths[0]} has no artifact_set_sha256")
    return paths[0], sha


def legacy_cache_key(model: str, system: str, prompt: str, temperature: float) -> str:
    raw = f"{model}:::{system}:::{prompt}:::{temperature:.4f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_api_key() -> str:
    env_file = ROOT_DIR / ".env"
    if "OPENROUTER_API_KEY" not in os.environ and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == "OPENROUTER_API_KEY" and v.strip():
                os.environ["OPENROUTER_API_KEY"] = v.strip().strip('"').strip("'")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (copy .env.example to .env and fill it in)")
    return key


def openrouter_transport(payload: dict, api_key: str) -> dict:
    """One HTTP POST to OpenRouter. Raises RetryableHTTPError on 429/5xx/network errors."""
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Thesis-CCD-G1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429 or e.code >= 500:
            raise RetryableHTTPError(f"HTTP {e.code}") from e
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RetryableHTTPError(str(e)) from e
    err = data.get("error")
    if err:
        code = int(err["code"]) if str(err.get("code", "")).isdigit() else 0
        if code == 429 or code >= 500:
            raise RetryableHTTPError(f"OpenRouter error {code}: {err.get('message')}")
        raise RuntimeError(f"OpenRouter error: {err}")
    return data


class LLMClient:
    def __init__(
        self,
        mode: str = "dry_run",
        max_cost_usd: float | None = None,
        cache_dir: Path | None = None,   # default: $CCD_CACHE_DIR or cache/ (the 2.2-h rerun uses its own cache)
        cost_log: Path | None = None,
        concurrency: int = 8,
        models_config: Path | dict | None = None,
        transport: Callable[[dict, str], dict] = openrouter_transport,
        out_ratio: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        ollama_transport_fn: Callable[[str, dict | None], dict] | None = None,
        splash_transport_fn: Callable[[str, dict | None], dict] | None = None,
    ):
        if mode not in ("mock", "dry_run", "execute"):
            raise ValueError(f"unknown mode: {mode}")
        if mode == "execute" and max_cost_usd is None:
            raise ValueError("execute mode requires max_cost_usd")
        self.mode = mode
        self.max_cost_usd = max_cost_usd
        self.cache_dir = Path(cache_dir or os.environ.get("CCD_CACHE_DIR") or ROOT_DIR / "cache")
        self.cache_path = self.cache_dir / "llm_cache.jsonl"
        self.legacy_path = self.cache_dir / "llm_cache.sqlite"
        self.cost_log = Path(cost_log or os.environ.get("CCD_COST_LOG") or (
            Path(os.environ.get("CCD_RESULTS_DIR")) / "api_cost_log.jsonl"
            if os.environ.get("CCD_RESULTS_DIR")
            else ROOT_DIR / "results" / "api_cost_log.jsonl"
        ))
        self.concurrency = concurrency
        self.transport = transport
        self._sleep = sleep
        cfg = models_config if isinstance(models_config, dict) else load_models_config(models_config)
        self.prices = cfg.get("prices") or {}
        self.aliases = cfg.get("legacy_aliases") or {}
        self.out_ratio = out_ratio if out_ratio is not None else cfg.get("dry_run_output_ratio", 0.5)
        self.models = cfg.get("models") or {}
        self.ollama = {**OLLAMA_DEFAULTS, **(cfg.get("ollama") or {})}
        self.ollama_transport = ollama_transport_fn or (lambda path, payload: local_transport(
            self.ollama["url"].rstrip("/") + path, payload, self.ollama["timeout_s"]))
        self._ollama_sem = threading.Semaphore(int(self.ollama["concurrency"]))
        self.splash = {**SPLASH_DEFAULTS, **(cfg.get("splash") or {})}
        # splash_transport_fn(url, payload): url = base_url + path, so per-model base_urls stay visible to fakes
        self.splash_transport = splash_transport_fn or (
            lambda url, payload: local_transport(url, payload, self.splash["timeout_s"], "Splash"))
        self._splash_sem = threading.Semaphore(int(self.splash["concurrency"]))
        roles = resolve_roles(cfg.get("roles"))
        json_roles = {r for r, o in (cfg.get("role_options") or {}).items() if (o or {}).get("json_mode")}
        json_models = {m for r, m in roles.items() if r in json_roles and m}
        plain_models = {m for r, m in roles.items() if r not in json_roles and m}
        self.json_models = json_models - plain_models
        if json_models & plain_models:
            print(f"[llm_client] WARNING: {sorted(json_models & plain_models)} hold both a JSON (judge) role and a "
                  f"non-JSON role; JSON format is OFF for them", file=sys.stderr)
        self.manifest_path = self.cost_log.parent / "models_manifest.json"
        self._digests: dict[str, dict] | None = None
        self._digests_splash: dict[str, dict] = {}

        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if mode != "mock" and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._cache[rec["k"]] = rec
        self._planned: set[str] = set()
        self._stats = {"calls_total": 0, "calls_cached": 0, "calls_uncached": 0, "calls_legacy_cached": 0}
        self._by_model: dict[str, dict] = {}
        self._unpriced: set[str] = set()
        self.usd_spent = 0.0

    # ---------- pricing / aliases ----------
    def _alias_names(self, model: str) -> list[str]:
        names = [model]
        for old, new in self.aliases.items():
            if new == model and old not in names:
                names.append(old)
            if old == model and new not in names:
                names.append(new)
        return names

    def provider(self, model: str) -> str:
        """models.<id>.provider; models not listed are OpenRouter (pre-Ollama behaviour)."""
        return (self.models.get(model) or {}).get("provider", "openrouter")

    def splash_opts(self, model: str) -> dict:
        """Splash settings for a model: SPLASH_DEFAULTS < top-level `splash:` < models.<id> keys."""
        return {**self.splash, **{k: v for k, v in (self.models.get(model) or {}).items() if k in SPLASH_DEFAULTS}}

    def provider_opts(self, model: str) -> dict | None:
        """Local-provider options that change the output (part of the cache key); None for OpenRouter."""
        provider = self.provider(model)
        if provider == "ollama":
            opts = {"num_ctx": int((self.models.get(model) or {}).get("num_ctx", self.ollama["num_ctx"])),
                    "json_mode": model in self.json_models, "seed": int(self.ollama["seed"])}
            think = (self.models.get(model) or {}).get("think")
            if think is not None:  # only models that configure it get the key, so every other cache key is unchanged
                opts["think"] = think
            return opts
        if provider == "splash":
            o = self.splash_opts(model)
            return {"provider": "splash", "reasoning_effort": o["reasoning_effort"],
                    "json_mode": model in self.json_models, "seed": int(o["seed"])}
        return None

    def _key(self, model, system, prompt, temperature, max_tokens, cache_salt=None) -> str:
        return cache_key(model, system, prompt, temperature, max_tokens, cache_salt, self.provider_opts(model))

    def model_digest(self, model: str) -> str | None:
        """Ollama digest from GET /api/tags, or the Splash manifest artifact_set_sha256 (splash_version), fetched
        once per client in execute mode only and recorded in the models manifest. None for OpenRouter models and
        in mock/dry_run."""
        provider = self.provider(model)
        if self.mode != "execute" or provider not in LOCAL_PROVIDERS:
            return None
        with self._lock:
            if provider == "splash":
                info = self._digests_splash.get(model)
                if info is None:
                    info = self._digests_splash[model] = self.splash_version(model)
                    self._record_manifest(model, info)
                return info["digest"]
            if self._digests is None:
                tags = self.ollama_transport("/api/tags", None)
                self._digests = {}
                for m in tags.get("models", []):
                    info = {"digest": m.get("digest"), "size": m.get("size"), "modified_at": m.get("modified_at")}
                    self._digests[m["name"]] = info
                    if m["name"].endswith(":latest"):
                        self._digests.setdefault(m["name"][: -len(":latest")], info)
            info = self._digests.get(model)
            if info is None:
                raise RuntimeError(f"Ollama model {model!r} is not installed (ollama pull {model})")
            if not info.get("recorded"):
                self._record_manifest(model, info)
                info["recorded"] = True
            return info["digest"]

    def splash_version(self, model: str) -> dict:
        """Version record of a Splash model: GET {base_url}/models must list the model id, and the package manifest
        (models.<id>.manifest_path) gives artifact_set_sha256. Real server + real file: execute mode only."""
        o = self.splash_opts(model)
        if not o["manifest_path"]:
            raise RuntimeError(f"models.{model}.manifest_path is not set in configs/models.yaml")
        ids = [m.get("id") for m in self.splash_transport(o["base_url"].rstrip("/") + "/models", None).get("data", [])]
        if model not in ids:
            raise RuntimeError(f"Splash server {o['base_url']} does not serve {model!r} (serves {ids})")
        path, sha = read_splash_manifest(o["manifest_path"])
        return {"digest": sha, "provider": "splash", "artifact_set_sha256": sha, "manifest_path": path,
                "served_model_ids": ids, "base_url": o["base_url"]}

    def _record_manifest(self, model: str, info: dict) -> None:
        """results/models_manifest.json: {model: {digest, <provider fields>, last_seen, previous_digests}}.
        Written once per model per client. Caller holds self._lock."""
        man = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        old = man.get(model) or {}
        prev = old.get("previous_digests", []) + ([old["digest"]] if old.get("digest") not in (None, info["digest"]) else [])
        man[model] = {**{k: v for k, v in info.items() if k != "recorded"},
                      "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "previous_digests": prev}
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def price(self, model: str) -> tuple[float, float] | None:
        if self.provider(model) in LOCAL_PROVIDERS:
            return 0.0, 0.0  # local model: no per-token cost by definition
        for name in self._alias_names(model):
            p = self.prices.get(name)
            if p:
                if UNPRICED in (p.get("in"), p.get("out")):
                    return None
                return float(p["in"]), float(p["out"])
        return None

    def _cost(self, model: str, in_tok: int, out_tok: int) -> float | None:
        p = self.price(model)
        if p is None:
            self._unpriced.add(model)
            return None
        return (in_tok * p[0] + out_tok * p[1]) / 1e6

    # ---------- caches ----------
    def _legacy_lookup(self, model, system, prompt, temperature) -> tuple[str, str] | None:
        if not self.legacy_path.exists():
            return None
        uri = self.legacy_path.resolve().as_uri() + "?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        try:
            for name in self._alias_names(model):
                row = conn.execute(
                    "SELECT response FROM api_cache WHERE cache_key = ?",
                    (legacy_cache_key(name, system, prompt, temperature),),
                ).fetchone()
                if row:
                    return row[0], name
        finally:
            conn.close()
        return None

    def _account(self, model: str, in_tok: int, out_tok: int, usd: float | None) -> None:
        m = self._by_model.setdefault(
            model, {"calls": 0, "in_tokens_est": 0, "out_tokens_est": 0, "usd_est": 0.0}
        )
        m["calls"] += 1
        m["in_tokens_est"] += in_tok
        m["out_tokens_est"] += out_tok
        if usd is None or m["usd_est"] is None:
            m["usd_est"] = None
        else:
            m["usd_est"] += usd

    def _log_cost(self, model: str, in_tok: int, out_tok: int, usd: float, cached: bool) -> None:
        rec = {"ts": time.time(), "model": model, "in_tokens": in_tok, "out_tokens": out_tok,
               "usd": usd, "cached": cached}
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
        with self.cost_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------- public API ----------
    def call(self, model: str, system: str, prompt: str, max_tokens: int,
             temperature: float = 0.0, cache_salt: str | None = None) -> tuple[str, dict]:
        in_tok = count_tokens(system) + count_tokens(prompt)
        key = self._key(model, system, prompt, temperature, max_tokens, cache_salt)
        provider = self.provider(model)
        meta = {"model": model, "cached": False, "dry_run": False, "mock": False,
                "legacy_cache": False, "input_tokens_est": in_tok, "cache_key": key, "provider": provider,
                "model_digest": self.model_digest(model)}

        if self.mode == "mock":
            from mock_llm import respond
            text = respond(model, system, prompt, max_tokens)
            out_tok = count_tokens(text)
            with self._lock:
                self._stats["calls_total"] += 1
                self._stats["calls_uncached"] += 1
                self._account(model, in_tok, out_tok, self._cost(model, in_tok, out_tok))
            return text, {**meta, "mock": True, "output_tokens": out_tok}

        hit = self._cache.get(key)
        legacy = (None if hit or cache_salt is not None or provider in LOCAL_PROVIDERS
                  else self._legacy_lookup(model, system, prompt, temperature))
        if hit and hit.get(provider):
            meta[provider] = hit[provider]  # stats of the original call (durations are not this call's)
        if hit or legacy:
            text = hit["text"] if hit else legacy[0]
            with self._lock:
                self._stats["calls_total"] += 1
                self._stats["calls_cached"] += 1
                if legacy:
                    self._stats["calls_legacy_cached"] += 1
                if self.mode == "execute":
                    self._log_cost(model, in_tok, 0, 0.0, True)
            if legacy:
                meta.update(legacy_cache=True, legacy_model=legacy[1])
            return text, {**meta, "cached": True, "output_tokens": count_tokens(text), "cost_usd": 0.0}

        if self.mode == "dry_run":
            out_est = int(max_tokens * self.out_ratio)
            with self._lock:
                usd = self._cost(model, in_tok, out_est)
                self._stats["calls_total"] += 1
                if key in self._planned:  # a real run would hit the cache the 2nd time
                    self._stats["calls_cached"] += 1
                else:
                    self._planned.add(key)
                    self._stats["calls_uncached"] += 1
                    self._account(model, in_tok, out_est, usd)
            return "", {**meta, "dry_run": True, "output_tokens_est": out_est, "cost_usd_est": usd}

        return self._execute(model, system, prompt, max_tokens, temperature, key, meta)

    @staticmethod
    def _messages(system: str, prompt: str) -> list[dict]:
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    def _send_ollama(self, model, system, prompt, max_tokens, temperature, meta):
        opts = self.provider_opts(model)
        payload = {"model": model, "messages": self._messages(system, prompt), "stream": False,
                   "options": {"num_ctx": opts["num_ctx"], "seed": opts["seed"], "temperature": temperature,
                               "num_predict": max_tokens}}
        if opts["json_mode"]:
            payload["format"] = "json"
        if "think" in opts:
            payload["think"] = opts["think"]
        def send() -> dict:
            try:
                return self.ollama_transport("/api/chat", payload)
            except RetryableHTTPError as e:
                # Ollama >= 0.40 aborts a degenerate repetition with HTTP 500. At temperature 0 a retry repeats it,
                # so it is recorded as an empty (failed) output, like a pre-0.40 run that hit num_predict and failed
                # the budget (amendment 2.2-f). Every other 5xx is still retried.
                if OLLAMA_REPEAT_ABORT in str(e):
                    self._unload_ollama(model)
                    return {"message": {"role": "assistant", "content": ""}, "done_reason": "repeat_limit"}
                raise
        with self._ollama_sem:
            data = self._retry(send, int(self.ollama["max_retries"]))
        text = (data.get("message") or {}).get("content") or ""
        stats = {"prompt_eval_count": data.get("prompt_eval_count"), "eval_count": data.get("eval_count"),
                 "total_duration_ns": data.get("total_duration"), "load_duration_ns": data.get("load_duration"),
                 "done_reason": data.get("done_reason"), "num_ctx": opts["num_ctx"], "json_mode": opts["json_mode"],
                 "seed": opts["seed"], "think": opts.get("think"),
                 "thinking_chars": len((data.get("message") or {}).get("thinking") or "")}
        return text, meta["input_tokens_est"], count_tokens(text), 0.0, stats

    def _unload_ollama(self, model: str) -> None:
        """Amendment 2.2-h: after repeat aborts the llama-server runner degrades until every later output is
        degenerate text, whatever the prompt (seen 2026-10-10/11 on one 14-hour runner; a fresh runner gave clean
        output for the same inputs). Unload it so the next call loads a fresh runner."""
        try:
            self.ollama_transport("/api/generate", {"model": model, "keep_alive": 0})
        except Exception:  # the abort is already recorded; a failed unload must not fail the call
            pass

    def _retry(self, send: Callable[[], dict], retries: int) -> dict:
        for attempt in range(retries + 1):
            try:
                return send()
            except RetryableHTTPError:
                if attempt == retries:
                    raise
                self._sleep(2 ** attempt)

    def _send_splash(self, model, system, prompt, max_tokens, temperature, meta):
        o, opts = self.splash_opts(model), self.provider_opts(model)
        payload = {"model": model, "messages": self._messages(system, prompt), "temperature": temperature,
                   "max_tokens": max_tokens, "seed": opts["seed"]}
        if opts["reasoning_effort"] is not None:
            payload["reasoning_effort"] = opts["reasoning_effort"]
        if opts["json_mode"]:
            payload["response_format"] = {"type": "json_object"}
        url = o["base_url"].rstrip("/") + "/chat/completions"
        with self._splash_sem:
            data = self._retry(lambda: self.splash_transport(url, payload), int(o["max_retries"]))
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        stats = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                 "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                 "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                 "finish_reason": choice.get("finish_reason"), "reasoning_effort": opts["reasoning_effort"],
                 "json_mode": opts["json_mode"], "seed": opts["seed"]}
        return text, meta["input_tokens_est"], count_tokens(text), 0.0, stats

    def _send_openrouter(self, model, system, prompt, max_tokens, temperature, meta):
        api_key = get_api_key()
        if self.price(model) is None:
            raise RuntimeError(f"price for {model} is {UNPRICED} in configs/models.yaml; confirm it before execute")
        with self._lock:
            if self.usd_spent >= self.max_cost_usd:
                raise BudgetExceeded(f"spent ${self.usd_spent:.4f} >= max ${self.max_cost_usd}")
        payload = {"model": model, "messages": self._messages(system, prompt), "temperature": temperature,
                   "max_tokens": max_tokens, "usage": {"include": True}}
        for attempt in range(MAX_ATTEMPTS):
            try:
                data = self.transport(payload, api_key)
                break
            except RetryableHTTPError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                self._sleep(2 ** attempt)
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        in_tok = int(usage["prompt_tokens"] if usage.get("prompt_tokens") is not None else meta["input_tokens_est"])
        out_tok = int(usage["completion_tokens"] if usage.get("completion_tokens") is not None else count_tokens(text))
        usd = usage.get("cost")
        usd = float(usd) if usd is not None else self._cost(model, in_tok, out_tok)
        return text, in_tok, out_tok, usd, None

    def _execute(self, model, system, prompt, max_tokens, temperature, key, meta):
        send = {"ollama": self._send_ollama, "splash": self._send_splash}.get(meta["provider"], self._send_openrouter)
        t0 = time.time()
        text, in_tok, out_tok, usd, stats = send(model, system, prompt, max_tokens, temperature, meta)
        latency_ms = (time.time() - t0) * 1000

        rec = {"k": key, "model": model, "temperature": temperature, "max_tokens": max_tokens,
               "text": text, "usage": {"in": in_tok, "out": out_tok}, "usd": usd, "ts": time.time()}
        if stats:
            meta = {**meta, meta["provider"]: stats}
            rec.update({meta["provider"]: stats, "model_digest": meta["model_digest"], "latency_ms": latency_ms})
        with self._lock:
            self._cache[key] = rec
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._log_cost(model, in_tok, out_tok, usd, False)
            self.usd_spent += usd
            self._stats["calls_total"] += 1
            self._stats["calls_uncached"] += 1
            self._account(model, in_tok, out_tok, usd)
            spent = self.usd_spent
        if spent > self.max_cost_usd:
            raise BudgetExceeded(f"spent ${spent:.4f} > max ${self.max_cost_usd}")
        return text, {**meta, "output_tokens": out_tok, "cost_usd": usd, "latency_ms": latency_ms}

    def call_chat(self, model: str, messages: list[dict], max_tokens: int, temperature: float = 0.0,
                  cache_salt: str | None = None) -> tuple[str, dict]:
        """Chat-message entry point (DSPy ClientLM). Supports [user] and [system, user] with string
        content; they map onto call(), so the wire request and the cache key are the same as for a
        harness call with that system/prompt pair. Other shapes are refused rather than flattened."""
        roles = [m.get("role") for m in messages]
        if roles not in (["user"], ["system", "user"]) or not all(isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"call_chat supports [user] or [system, user] text messages only; got roles {roles}")
        system = messages[0]["content"] if roles[0] == "system" else ""
        return self.call(model=model, system=system, prompt=messages[-1]["content"], max_tokens=max_tokens,
                         temperature=temperature, cache_salt=cache_salt)

    def call_many(self, requests: list[dict]) -> list[tuple[str, dict]]:
        """requests = call() kwargs dicts. Duplicate requests run once. Returns input order."""
        keys = [self._key(r["model"], r["system"], r["prompt"], r.get("temperature", 0.0), r["max_tokens"],
                          r.get("cache_salt")) for r in requests]
        unique = {k: r for k, r in zip(keys, requests)}
        # ponytail: concurrent threads can overshoot max_cost by up to `concurrency` calls
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            done = dict(zip(unique, ex.map(lambda r: self.call(**r), unique.values())))
        return [done[k] for k in keys]

    def summary(self) -> dict:
        """by_model counts calls that are not cache hits (mock: all calls; dry_run: planned calls;
        execute: real calls). usd_est_total excludes unpriced models (listed in unpriced_models)."""
        with self._lock:
            by_model = {m: dict(v) for m, v in self._by_model.items()}
            total = sum(v["usd_est"] for v in by_model.values() if v["usd_est"] is not None)
            return {**self._stats, "by_model": by_model, "usd_est_total": total,
                    "usd_spent": self.usd_spent, "unpriced_models": sorted(self._unpriced),
                    "mode": self.mode}
