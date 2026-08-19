"""The hosted provider, exercised without touching the network.

The demo deployment runs on this path rather than Ollama, so a break here is
invisible locally and visible to whoever opens the portfolio link. Every test
substitutes urlopen, so the suite still needs no model and no API key.

The interesting cases are the mismatches between JSON Schema and Gemini's
`responseSchema`, and between the pipeline's message roles and Gemini's — both
are silent-wrong-answer territory rather than crashes.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from analyst.interpret import PLAN_SCHEMA
from analyst.llm.base import Message, ProviderError
from analyst.llm.gemini import GeminiProvider, _clean_schema
from analyst.session import ROUTE_SCHEMA


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the context manager urlopen returns."""

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _reply(text: str, *, prompt_tokens: int = 11, output_tokens: int = 7) -> _FakeResponse:
    return _FakeResponse(
        json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": text}]}}],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": output_tokens,
                },
            }
        ).encode()
    )


@pytest.fixture
def capture(monkeypatch):
    """Capture the request body, and reply with whatever the test queues."""
    sent: dict = {}

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        sent["url"] = request.full_url
        sent["headers"] = {k.lower(): v for k, v in request.header_items()}
        sent["body"] = json.loads(request.data.decode())
        return sent.pop("reply")

    monkeypatch.setattr("analyst.llm.gemini.urllib.request.urlopen", fake_urlopen)
    return sent


# ------------------------------------------------------------------ schemas


@pytest.mark.parametrize("schema", [PLAN_SCHEMA, ROUTE_SCHEMA])
def test_pipeline_schemas_survive_cleaning(schema):
    """Cleaning must not damage the schemas the pipeline actually sends."""
    cleaned = _clean_schema(schema)
    assert cleaned["type"] == schema["type"]
    assert set(cleaned["properties"]) == set(schema["properties"])
    assert cleaned["required"] == schema["required"]


def test_unsupported_keys_are_stripped_at_every_depth():
    """Gemini 400s on unknown schema keys, including nested ones."""
    cleaned = _clean_schema(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "title": "drop me", "default": {}},
                }
            },
        }
    )
    flat = json.dumps(cleaned)
    for banned in ("$schema", "additionalProperties", "title", "default"):
        assert banned not in flat
    # ...without taking the legitimate keys with it.
    assert cleaned["properties"]["items"]["type"] == "array"


def test_enum_is_preserved():
    """The router depends on enum constraining the route to two values."""
    cleaned = _clean_schema({"type": "string", "enum": ["change_over_time", "other"]})
    assert cleaned["enum"] == ["change_over_time", "other"]


# ------------------------------------------------------------------ requests


def test_a_schema_request_asks_for_constrained_json(capture):
    capture["reply"] = _reply('{"route": "other", "reason": "ranking"}')
    provider = GeminiProvider("k")

    completion = provider.complete(
        system="sys", prompt="q", schema=ROUTE_SCHEMA, max_tokens=150
    )

    config = capture["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["type"] == "object"
    assert completion.data == {"route": "other", "reason": "ranking"}


def test_a_plain_request_does_not_ask_for_json(capture):
    capture["reply"] = _reply("prose")
    GeminiProvider("k").complete(system="sys", prompt="q")

    assert "responseSchema" not in capture["body"]["generationConfig"]
    assert "responseMimeType" not in capture["body"]["generationConfig"]


def test_the_api_key_travels_as_a_header_not_a_query_parameter(capture):
    """?key=... leaks into logs and proxy records; a header does not."""
    capture["reply"] = _reply("ok")
    GeminiProvider("secret-key").complete(system="s", prompt="p")

    assert "secret-key" not in capture["url"]
    assert capture["headers"]["x-goog-api-key"] == "secret-key"


def test_chat_maps_the_assistant_role_to_model(capture):
    """Gemini names the assistant "model"; the codegen loop alternates roles and
    breaks silently if that mapping is wrong."""
    capture["reply"] = _reply("done")
    GeminiProvider("k").chat(
        system="sys",
        messages=[
            Message(role="user", content="question"),
            Message(role="assistant", content="```python\nprint(1)\n```"),
            Message(role="user", content="STDOUT:\n1"),
        ],
    )

    assert [c["role"] for c in capture["body"]["contents"]] == ["user", "model", "user"]
    assert capture["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_usage_is_reported_from_the_response(capture):
    capture["reply"] = _reply("hi", prompt_tokens=120, output_tokens=34)
    completion = GeminiProvider("k").complete(system="s", prompt="p")

    assert (completion.input_tokens, completion.output_tokens) == (120, 34)
    assert completion.cost_usd == 0.0  # free tier — the ledger must not invent a price


# ------------------------------------------------------------------ failures


def test_a_blocked_response_yields_empty_text_rather_than_raising(capture):
    """A safety block returns candidates with no parts. Indexing into parts[0]
    would raise on a response that is structurally fine."""
    capture["reply"] = _FakeResponse(
        json.dumps({"candidates": [{"content": {}, "finishReason": "SAFETY"}]}).encode()
    )
    assert GeminiProvider("k").complete(system="s", prompt="p").text == ""


def test_malformed_json_under_a_schema_is_an_actionable_error(capture):
    capture["reply"] = _reply("not json at all")
    with pytest.raises(ProviderError, match="Expected JSON"):
        GeminiProvider("k").complete(system="s", prompt="p", schema=ROUTE_SCHEMA)


@pytest.mark.parametrize(
    "code,expected",
    [
        (403, "rejected the API key"),
        (404, "no model"),
        (429, "rate limit"),
    ],
)
def test_http_errors_carry_the_fix(monkeypatch, code, expected):
    """Errors name the remedy, matching the Ollama provider's contract."""

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(
            request.full_url, code, "err", {}, io.BytesIO(b"{}")
        )

    monkeypatch.setattr("analyst.llm.gemini.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError, match=expected):
        GeminiProvider("k").complete(system="s", prompt="p")


def test_a_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="No Gemini API key"):
        GeminiProvider("")


def test_health_reports_a_bad_model_instead_of_claiming_ready(monkeypatch):
    """A reachable API with a wrong model name looks healthy until the first
    real question fails, so health() makes an actual generation call."""

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(
            request.full_url, 404, "not found", {}, io.BytesIO(b"{}")
        )

    monkeypatch.setattr("analyst.llm.gemini.urllib.request.urlopen", fake_urlopen)
    ok, status = GeminiProvider("k", "gemini-does-not-exist").health()
    assert not ok
    assert "no model" in status
