"""Deterministic retrieval for reusable optimization-method experience cards."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


logger = logging.getLogger("MLEvolve")
DEFAULT_LIBRARY_PATH = Path(__file__).with_name("optimization_experiences.yaml")


def _text_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


@dataclass(frozen=True)
class TriggerGroup:
    label: str
    weight: float
    terms: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationExperienceCard:
    experience_id: str
    title: str
    provenance: str
    summary: str
    problem_signature: tuple[str, ...]
    trigger_groups: tuple[TriggerGroup, ...]
    core_transform: tuple[str, ...]
    solve_trajectory: tuple[str, ...]
    runtime_evidence: tuple[str, ...]
    failure_lessons: tuple[str, ...]
    not_applicable_when: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OptimizationExperienceCard | None":
        experience_id = str(payload.get("experience_id") or "").strip()
        title = str(payload.get("title") or "").strip()
        if not experience_id or not title:
            return None
        trigger_groups = []
        for raw_group in payload.get("trigger_groups") or []:
            if not isinstance(raw_group, Mapping):
                continue
            terms = _text_list(raw_group.get("terms"))
            if not terms:
                continue
            try:
                weight = float(raw_group.get("weight") or 1.0)
            except (TypeError, ValueError):
                weight = 1.0
            trigger_groups.append(
                TriggerGroup(
                    label=str(raw_group.get("label") or "signal"),
                    weight=max(0.0, weight),
                    terms=terms,
                )
            )
        return cls(
            experience_id=experience_id,
            title=title,
            provenance=str(payload.get("provenance") or "").strip(),
            summary=str(payload.get("summary") or "").strip(),
            problem_signature=_text_list(payload.get("problem_signature")),
            trigger_groups=tuple(trigger_groups),
            core_transform=_text_list(payload.get("core_transform")),
            solve_trajectory=_text_list(payload.get("solve_trajectory")),
            runtime_evidence=_text_list(payload.get("runtime_evidence")),
            failure_lessons=_text_list(payload.get("failure_lessons")),
            not_applicable_when=_text_list(payload.get("not_applicable_when")),
        )


@dataclass(frozen=True)
class OptimizationExperienceMatch:
    card: OptimizationExperienceCard
    score: float
    matched_signals: tuple[str, ...]


def load_optimization_experience_cards(
    path: str | Path | None = None,
) -> tuple[OptimizationExperienceCard, ...]:
    library_path = Path(path).expanduser() if path else DEFAULT_LIBRARY_PATH
    if not library_path.is_absolute():
        library_path = (Path(__file__).resolve().parents[2] / library_path).resolve()
    payload = yaml.safe_load(library_path.read_text(encoding="utf-8-sig")) or {}
    cards = []
    for raw_card in payload.get("cards") or []:
        if isinstance(raw_card, Mapping):
            card = OptimizationExperienceCard.from_mapping(raw_card)
            if card is not None:
                cards.append(card)
    return tuple(cards)


def retrieve_optimization_experiences(
    query_text: str,
    *,
    cards: Iterable[OptimizationExperienceCard] | None = None,
    max_cards: int = 2,
    min_score: float = 3.0,
) -> list[OptimizationExperienceMatch]:
    normalized = _normalize_text(query_text)
    if not normalized:
        return []
    matches = []
    for card in cards or load_optimization_experience_cards():
        score = 0.0
        matched_signals = []
        for group in card.trigger_groups:
            if any(_normalize_text(term) in normalized for term in group.terms):
                score += group.weight
                matched_signals.append(group.label)
        if score >= min_score:
            matches.append(
                OptimizationExperienceMatch(
                    card=card,
                    score=score,
                    matched_signals=tuple(matched_signals),
                )
            )
    matches.sort(key=lambda match: (-match.score, match.card.experience_id))
    return matches[: max(0, int(max_cards))]


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_optimization_experience_context(
    cards: Iterable[OptimizationExperienceCard],
    matches: Iterable[OptimizationExperienceMatch],
    *,
    max_chars: int = 6000,
) -> str:
    card_list = list(cards)
    match_list = list(matches)
    parts = [
        "## Optimization Method Experience Index",
        "Use these as conditional hypotheses, not mandatory algorithms. The current task contract and evaluator remain authoritative.",
    ]
    for card in card_list:
        signature = "; ".join(card.problem_signature[:2])
        parts.append(f"- `{card.experience_id}`: {card.title}. Signals: {signature}")

    if match_list:
        parts.append("\n## Retrieved Optimization Experience")
    for match in match_list:
        card = match.card
        parts.extend(
            [
                f"### {card.title}",
                f"- experience_id: `{card.experience_id}`",
                f"- provenance: {card.provenance}" if card.provenance else "",
                f"- retrieval_evidence: {', '.join(match.matched_signals)}",
                f"- summary: {card.summary}",
                "**Applicability checks**",
                _bullets(card.problem_signature),
                "**Core transform**",
                _bullets(card.core_transform),
                "**Solve trajectory**",
                _bullets(card.solve_trajectory),
                "**Runtime evidence to preserve**",
                _bullets(card.runtime_evidence),
                "**Failure lessons**",
                _bullets(card.failure_lessons),
                "**Do not apply when**",
                _bullets(card.not_applicable_when),
            ]
        )
    text = "\n".join(part for part in parts if part)
    max_chars = max(500, int(max_chars))
    if len(text) > max_chars:
        text = text[: max_chars - 80].rstrip() + "\n[Experience context truncated by configured limit.]"
    return text


def build_optimization_experience_for_agent(
    agent: Any,
    *,
    task_mode: str,
    extra_context: str = "",
) -> str:
    if task_mode != "optimization":
        return ""
    acfg = getattr(agent, "acfg", None)
    if not bool(getattr(acfg, "use_optimization_experience_library", True)):
        return ""
    library_path = str(getattr(acfg, "optimization_experience_library_path", "") or "").strip()
    try:
        cards = load_optimization_experience_cards(library_path or None)
    except Exception as exc:
        logger.warning("Failed to load optimization experience library: %s", exc)
        return ""
    query_text = "\n".join(
        [
            str(getattr(agent, "task_desc", "") or ""),
            str(getattr(agent, "data_preview", "") or ""),
            str(getattr(agent, "coldstart_description", "") or ""),
            str(extra_context or ""),
        ]
    )
    matches = retrieve_optimization_experiences(
        query_text,
        cards=cards,
        max_cards=int(getattr(acfg, "optimization_experience_max_cards", 2)),
        min_score=float(getattr(acfg, "optimization_experience_min_score", 3.0)),
    )
    logger.info(
        "Optimization experience retrieval: cards=%d, selected=%s",
        len(cards),
        [match.card.experience_id for match in matches],
    )
    return render_optimization_experience_context(
        cards,
        matches,
        max_chars=int(getattr(acfg, "optimization_experience_max_chars", 6000)),
    )
