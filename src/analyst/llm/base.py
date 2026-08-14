"""Provider-agnostic LLM interface.

The model is used for language, never for arithmetic — question interpretation
and narrative, with every number computed by analyst.analysis. That keeps the
surface small enough that a local 14B and a frontier API model are genuinely
interchangeable, which is why this abstraction is worth having.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Completion:
    text: str
    model: str
    data: dict[str, Any] | None = None  # populated when a schema was requested
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.duration_s if self.duration_s else 0.0


@dataclass
class ProviderInfo:
    name: str
    model: str
    local: bool
    cost_per_mtok_in: float = 0.0
    cost_per_mtok_out: float = 0.0

    def describe(self) -> str:
        where = "local" if self.local else "hosted"
        price = (
            "free"
            if self.local
            else f"${self.cost_per_mtok_in:.2f}/${self.cost_per_mtok_out:.2f} per Mtok"
        )
        return f"{self.name}:{self.model} ({where}, {price})"


class ProviderError(RuntimeError):
    """Raised with actionable remediation rather than a raw transport error."""


class Provider(ABC):
    """Minimal surface: one call, optionally schema-constrained."""

    info: ProviderInfo

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Return a completion. When `schema` is given, `.data` holds parsed JSON."""

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion:
        """Multi-turn completion. Needed by loops that iterate on their own output."""

    @abstractmethod
    def health(self) -> tuple[bool, str]:
        """(reachable, human-readable status) — used by the UI before accepting work."""


@dataclass
class UsageLedger:
    """Running total, so cost is visible rather than a surprise at month end."""

    completions: list[Completion] = field(default_factory=list)

    def record(self, completion: Completion) -> Completion:
        self.completions.append(completion)
        return completion

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.completions)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.completions)

    def summary(self) -> str:
        if not self.completions:
            return "no model calls"
        cost = f"${self.total_cost_usd:.4f}" if self.total_cost_usd else "free"
        seconds = sum(c.duration_s for c in self.completions)
        return (
            f"{len(self.completions)} model call(s), "
            f"{self.total_output_tokens:,} output tokens, {seconds:.1f}s, {cost}"
        )
