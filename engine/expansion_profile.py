"""Durable per-parent expansion profiles for sibling-adaptive search prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def complexity_for_sibling(sibling_ordinal: int) -> str:
    """AIRA-style local complexity: children 1-2 simple, 3-4 normal, 5+ complex."""

    ordinal = max(1, int(sibling_ordinal or 1))
    if ordinal <= 2:
        return "simple"
    if ordinal <= 4:
        return "normal"
    return "complex"


@dataclass(frozen=True)
class ExpansionProfile:
    sibling_ordinal: int
    complexity: str
    operator: str = "auto"
    task_family: str = "prediction"

    @classmethod
    def create(
        cls,
        sibling_ordinal: int,
        *,
        operator: str = "auto",
        task_family: str = "prediction",
    ) -> "ExpansionProfile":
        ordinal = max(1, int(sibling_ordinal or 1))
        family = "decision" if str(task_family).lower() == "decision" else "prediction"
        return cls(
            sibling_ordinal=ordinal,
            complexity=complexity_for_sibling(ordinal),
            operator=str(operator or "auto").lower(),
            task_family=family,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ExpansionProfile | None":
        if not isinstance(payload, Mapping):
            return None
        try:
            ordinal = int(payload.get("sibling_ordinal") or 0)
        except (TypeError, ValueError):
            return None
        if ordinal <= 0:
            return None
        return cls.create(
            ordinal,
            operator=str(payload.get("operator") or "auto"),
            task_family=str(payload.get("task_family") or "prediction"),
        )

    def with_operator(self, operator: str) -> "ExpansionProfile":
        return ExpansionProfile(
            sibling_ordinal=self.sibling_ordinal,
            complexity=self.complexity,
            operator=str(operator or "auto").lower(),
            task_family=self.task_family,
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)
