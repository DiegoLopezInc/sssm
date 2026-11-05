"""Minimal chat client for local OpenAI-compatible / Ollama servers (stdlib only)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class CallStats:
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@dataclass
class Usage:
    calls: list[CallStats] = field(default_factory=list)

    def add(self, stats: CallStats) -> None:
        self.calls.append(stats)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def peak_context(self) -> int:
        return max((c.prompt_tokens + c.completion_tokens for c in self.calls), default=0)

    @property
    def seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    def summary(self) -> dict:
        return {
            "calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "peak_context": self.peak_context,
            "llm_seconds": round(self.seconds, 2),
        }


class ChatClient:
    """Talks to Ollama's native API (default) or any OpenAI-compatible server.

    The Ollama backend lets us pin ``num_ctx`` per request, which controls the KV-cache
    size and therefore RAM use - the main knob that differs between full-history and
    SSSM operation on a memory-limited machine.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        backend: str = "ollama",
        num_ctx: int = 8192,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == 2:
                    raise
                print(f"server error {e.code}: {e.read()[:200]!r}; retrying", file=sys.stderr, flush=True)
                time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    def chat(self, messages: list[dict], max_tokens: int = 512, usage: Usage | None = None) -> str:
        t0 = time.perf_counter()
        if self.backend == "ollama":
            data = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_ctx": self.num_ctx,
                        "num_predict": max_tokens,
                        "temperature": self.temperature,
                        "seed": self.seed,
                    },
                },
            )
            text = data["message"]["content"]
            stats = CallStats(data.get("prompt_eval_count", 0), data.get("eval_count", 0), 0.0)
        else:
            data = self._post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                    "seed": self.seed,
                },
            )
            text = data["choices"][0]["message"]["content"]
            u = data.get("usage", {})
            stats = CallStats(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), 0.0)
        stats.seconds = time.perf_counter() - t0
        if usage is not None:
            usage.add(stats)
        return text
