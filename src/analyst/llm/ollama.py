"""Ollama provider — a local model, free to run, no API key.

Two details that matter and are easy to get wrong:

  num_ctx   Ollama's default context is 4096 tokens. A system prompt plus a
            dataset profile is already ~1,500, so the default silently truncates
            once a conversation gets going. Set it explicitly, every request.

  format    Passing a JSON schema constrains decoding, so structured extraction
            returns valid JSON rather than prose wrapped in a code fence that
            has to be salvaged with a regex.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from analyst.llm.base import Completion, Message, Provider, ProviderError, ProviderInfo

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_NUM_CTX = 8192


class OllamaProvider(Provider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.info = ProviderInfo(name="ollama", model=model, local=True)

    # ----------------------------------------------------------------- health

    def health(self) -> tuple[bool, str]:
        try:
            installed = self._installed_models()
        except ProviderError as exc:
            return False, str(exc)

        if self.model not in installed:
            available = ", ".join(sorted(installed)) or "none"
            return False, (
                f"Model '{self.model}' is not installed. Run:\n"
                f"    ollama pull {self.model}\n"
                f"Installed: {available}"
            )
        return True, f"{self.model} ready at {self.host}"

    def _installed_models(self) -> set[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                payload = json.load(response)
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.host} ({exc.reason}). Start it with:\n"
                "    brew services start ollama"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise ProviderError(f"Cannot reach Ollama at {self.host}: {exc}") from exc

        return {m["name"] for m in payload.get("models", [])}

    # ------------------------------------------------------------------ call

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code == 404:
                raise ProviderError(
                    f"Ollama does not have '{self.model}'. Run: ollama pull {self.model}"
                ) from exc
            raise ProviderError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.host} ({exc.reason}). "
                "Start it with: brew services start ollama"
            ) from exc

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        payload = self._post("/api/chat", body)
        return Completion(
            text=payload.get("message", {}).get("content", "").strip(),
            model=self.model,
            input_tokens=payload.get("prompt_eval_count", 0),
            output_tokens=payload.get("eval_count", 0),
            duration_s=payload.get("total_duration", 0) / 1e9,
        )

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if schema is not None:
            body["format"] = schema

        started = time.monotonic()
        payload = self._post("/api/generate", body)
        text = payload.get("response", "").strip()
        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Expected JSON matching the schema, got:\n{text[:400]}"
                ) from exc

        return Completion(
            text=text,
            model=self.model,
            data=data,
            input_tokens=payload.get("prompt_eval_count", 0),
            output_tokens=payload.get("eval_count", 0),
            duration_s=time.monotonic() - started,
            cost_usd=0.0,
        )
