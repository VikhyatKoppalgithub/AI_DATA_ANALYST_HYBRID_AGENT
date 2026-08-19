"""Google Gemini provider — the hosted path, used by the deployed demo.

The project is built to run locally and free, and that is still the default. But
no free host will run a local LLM: Hugging Face made Docker Spaces PRO-only, and
Streamlit Community Cloud has neither Ollama nor the RAM for it. So the public
demo needs a hosted model, and Gemini's free tier is the one that costs nothing
(15 req/min, 1,500/day at the time of writing).

This file is also the claim the README makes, made checkable. The architecture
argues that keeping `Provider` abstract makes local and hosted models
interchangeable. This is that swap: same three methods, no change anywhere in
the pipeline, and the deployed demo runs the identical analysis code.

Written against the REST API with urllib rather than the google-genai SDK, for
the same reason ollama.py is: one fewer dependency to install on a host, and the
request shape stays visible.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from analyst.llm.base import Completion, Message, Provider, ProviderError, ProviderInfo

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
# Overridable: Google renames and retires these, and a demo that dies on a model
# rename should be fixable with an env var rather than a commit.
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Free-tier flash models bill nothing, so the ledger stays honest at zero rather
# than inventing a price. If a paid model is configured this under-reports, which
# is why the demo pins a flash model.
COST_PER_MTOK_IN = 0.0
COST_PER_MTOK_OUT = 0.0

# JSON Schema keys Gemini's responseSchema does not accept. The pipeline's
# schemas are plain JSON Schema; passing an unknown key is a 400, so they are
# stripped rather than hand-maintaining a second copy of each schema.
_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema", "$id", "$ref", "definitions", "additionalProperties",
    "patternProperties", "default", "examples", "title",
}


def _clean_schema(node: Any) -> Any:
    """Recursively drop schema keys Gemini rejects."""
    if isinstance(node, dict):
        return {
            k: _clean_schema(v)
            for k, v in node.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


class GeminiProvider(Provider):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "No Gemini API key. Set GEMINI_API_KEY in the environment, or add "
                "it to .streamlit/secrets.toml as:\n    GEMINI_API_KEY = \"...\"\n"
                "Get one free at https://aistudio.google.com/apikey"
            )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.info = ProviderInfo(
            name="gemini",
            model=model,
            local=False,
            cost_per_mtok_in=COST_PER_MTOK_IN,
            cost_per_mtok_out=COST_PER_MTOK_OUT,
        )

    # ----------------------------------------------------------------- health

    def health(self) -> tuple[bool, str]:
        """One real generation, because a reachable API with a wrong model name
        looks identical to a working one until the first question fails."""
        try:
            self._post(
                self.model,
                {
                    "contents": [{"role": "user", "parts": [{"text": "ok"}]}],
                    "generationConfig": {"maxOutputTokens": 1},
                },
            )
        except ProviderError as exc:
            return False, str(exc)
        return True, f"{self.model} ready (Gemini free tier, hosted)"

    # ------------------------------------------------------------------ call

    def _post(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        # The key travels as a header, not a query parameter, so it cannot leak
        # into logs or proxy access records the way ?key=... does.
        request = urllib.request.Request(
            f"{API_ROOT}/{urllib.parse.quote(model)}:generateContent",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (401, 403):
                raise ProviderError(
                    f"Gemini rejected the API key ({exc.code}). Check GEMINI_API_KEY "
                    "and that the Generative Language API is enabled for it."
                ) from exc
            if exc.code == 404:
                raise ProviderError(
                    f"Gemini has no model {model!r}. Set GEMINI_MODEL to a current "
                    "one — see https://ai.google.dev/gemini-api/docs/models"
                ) from exc
            if exc.code == 429:
                raise ProviderError(
                    "Gemini free-tier rate limit hit (15 requests/minute, "
                    "1,500/day). Wait a minute and ask again."
                ) from exc
            raise ProviderError(f"Gemini returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Cannot reach Gemini ({exc.reason}).") from exc

    @staticmethod
    def _text(payload: dict[str, Any]) -> str:
        """First candidate's concatenated text parts, or "" if it was blocked.

        A safety block returns candidates with no parts rather than an error, so
        indexing straight into parts[0] would raise KeyError on a response that
        is structurally fine — the model simply declined.
        """
        for candidate in payload.get("candidates", []):
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if text:
                return text.strip()
        return ""

    @staticmethod
    def _usage(payload: dict[str, Any]) -> tuple[int, int]:
        usage = payload.get("usageMetadata", {})
        return (
            int(usage.get("promptTokenCount", 0)),
            int(usage.get("candidatesTokenCount", 0)),
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
        config: dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None:
            # Constrained decoding, so structured extraction returns valid JSON
            # instead of prose in a fence that has to be salvaged with a regex.
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = _clean_schema(schema)

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": config,
        }

        started = time.monotonic()
        payload = self._post(self.model, body)
        text = self._text(payload)
        input_tokens, output_tokens = self._usage(payload)

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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_s=time.monotonic() - started,
            cost_usd=0.0,
        )

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        # Gemini calls the assistant role "model"; everything else maps straight
        # across. The codegen loop depends on this alternation being preserved.
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
        ]
        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        started = time.monotonic()
        payload = self._post(self.model, body)
        input_tokens, output_tokens = self._usage(payload)

        return Completion(
            text=self._text(payload),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_s=time.monotonic() - started,
            cost_usd=0.0,
        )
