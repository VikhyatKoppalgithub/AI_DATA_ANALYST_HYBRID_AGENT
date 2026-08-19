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
# rename should be fixable with an env var rather than a commit. `gemini-2.0-flash`
# was the first default here and was already retired by deploy day, which is why
# a 404 now lists what the key can actually use instead of pointing at the docs.
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Free-tier flash models bill nothing, so the ledger stays honest at zero rather
# than inventing a price. If a paid model is configured this under-reports, which
# is why the demo pins a flash model.
COST_PER_MTOK_IN = 0.0
COST_PER_MTOK_OUT = 0.0

# Gemini 2.5+ models reason internally before answering, and those tokens are
# charged against maxOutputTokens. The routing call allows 150, which the model
# spent thinking — returning JSON cut off mid-string. That surfaced as "Expected
# JSON matching the schema", a parse error whose real cause was a token cap, and
# it cost an afternoon to read correctly. Thinking buys nothing on this pipeline:
# the model picks a column name, or writes three sentences over figures it was
# handed. It never reasons about arithmetic, by design.
THINKING_OFF: dict[str, Any] = {"thinkingConfig": {"thinkingBudget": 0}}

# Truncation under a schema is silent — it produces invalid JSON rather than an
# error — so schema-constrained calls get headroom regardless of what the caller
# asked for. The pipeline's plans are small; the cost of the floor is nothing.
MIN_SCHEMA_TOKENS = 512

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
        """Verify the model exists by listing, not by generating.

        The first version here spent a real generateContent call. Streamlit
        re-runs the whole script on every widget interaction, so that burned a
        request per keystroke and exhausted the free tier's 15/minute before a
        question could be asked. Listing costs no generation quota and catches
        the same failure more precisely — a wrong model name is a name absent
        from the list, rather than an opaque 404.
        """
        try:
            available = self._list_models()
        except ProviderError as exc:
            return False, str(exc)

        if not available:
            # Listing worked but told us nothing; do not block on it.
            return True, f"{self.model} (hosted; could not verify the model list)"
        if self.model not in available:
            return False, (
                f"Gemini has no model {self.model!r} for this key."
                f"{self._suggest_from(available)}"
            )
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
                    f"Gemini has no model {model!r}.{self._suggest_models()}"
                ) from exc
            if exc.code == 429:
                raise ProviderError(
                    "Gemini free-tier rate limit hit (15 requests/minute, "
                    "1,500/day). Wait a minute and ask again."
                ) from exc
            raise ProviderError(f"Gemini returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Cannot reach Gemini ({exc.reason}).") from exc

    _DOCS_HINT = (
        " Set GEMINI_MODEL to a current one — "
        "see https://ai.google.dev/gemini-api/docs/models"
    )

    def _list_models(self) -> list[str]:
        """Models this key can call generateContent on."""
        request = urllib.request.Request(
            API_ROOT, headers={"x-goog-api-key": self.api_key}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ProviderError(
                    f"Gemini rejected the API key ({exc.code}). Check GEMINI_API_KEY "
                    "and that the Generative Language API is enabled for it."
                ) from exc
            raise ProviderError(f"Gemini returned HTTP {exc.code} listing models.") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Cannot reach Gemini ({exc.reason}).") from exc

        return sorted(
            m.get("name", "").removeprefix("models/")
            for m in payload.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        )

    @staticmethod
    def _suggest_from(available: list[str]) -> str:
        # Prefer stable flash models: this pipeline wants cheap and fast, and a
        # preview name is a worse thing to pin a demo to than a stable one.
        preferred = [n for n in available if "flash" in n and "preview" not in n]
        shortlist = (preferred or available)[:6]
        if not shortlist:
            return GeminiProvider._DOCS_HINT
        return " Set GEMINI_MODEL to one this key can use, e.g. " + ", ".join(shortlist)

    def _suggest_models(self) -> str:
        """Name the models this key can actually use.

        Google retires model names on its own schedule, so "no such model" is a
        routine failure rather than a typo — the first default committed here was
        already dead by the time the demo deployed. Listing what is available
        turns a docs search into a copy-paste. Best-effort: if the listing call
        also fails, fall back to the docs link rather than masking the real error.
        """
        try:
            return self._suggest_from(self._list_models())
        except Exception:  # noqa: BLE001 — a hint must never replace the real failure
            return self._DOCS_HINT

    def _generate(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST, withdrawing thinkingConfig if this model refuses to go without it.

        Not every model accepts a zero thinking budget — some reasoning models
        require one — so the field is sent optimistically and retracted when the
        API objects, rather than maintaining a hardcoded list of which models
        support it that would rot the same way the model names do.
        """
        try:
            return self._post(self.model, body)
        except ProviderError as exc:
            if "thinking" not in str(exc).lower():
                raise
            retry = {**body, "generationConfig": dict(body.get("generationConfig", {}))}
            retry["generationConfig"].pop("thinkingConfig", None)
            return self._post(self.model, retry)

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
    def _finish_reason(payload: dict[str, Any]) -> str:
        """Why generation stopped.

        The difference between "the model wrote prose" and "the model was cut
        off" is invisible in the text alone — both arrive as a fragment that
        fails to parse — but it points at completely different fixes. MAX_TOKENS
        means raise the budget; STOP means the schema was not applied.
        """
        for candidate in payload.get("candidates", []):
            reason = candidate.get("finishReason")
            if reason:
                return str(reason)
        return "unknown"

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
            **THINKING_OFF,
        }
        if schema is not None:
            # Constrained decoding, so structured extraction returns valid JSON
            # instead of prose in a fence that has to be salvaged with a regex.
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = _clean_schema(schema)
            config["maxOutputTokens"] = max(max_tokens, MIN_SCHEMA_TOKENS)

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": config,
        }

        started = time.monotonic()
        payload = self._generate(body)
        text = self._text(payload)
        input_tokens, output_tokens = self._usage(payload)

        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                # Name the stop reason and the budget. "got: Here is" is
                # ambiguous between a truncated reply and a prose one, and the
                # two need opposite fixes.
                reason = self._finish_reason(payload)
                hint = {
                    "MAX_TOKENS": (
                        " — the reply was cut off, so raise maxOutputTokens"
                    ),
                    "STOP": (
                        " — the model finished normally without honouring the "
                        "schema, so responseSchema was not applied"
                    ),
                }.get(reason, "")
                raise ProviderError(
                    f"Expected JSON matching the schema. finishReason={reason}, "
                    f"{output_tokens} output tokens of "
                    f"{config['maxOutputTokens']} allowed{hint}.\nGot: {text[:300]!r}"
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
                **THINKING_OFF,
            },
        }

        started = time.monotonic()
        payload = self._generate(body)
        input_tokens, output_tokens = self._usage(payload)

        return Completion(
            text=self._text(payload),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_s=time.monotonic() - started,
            cost_usd=0.0,
        )
