"""Explicit, runtime-object-free persistence snapshots for search journals."""

from __future__ import annotations

import dataclasses
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.metric import MetricValue


_VALID_STAGES = {
    "root",
    "improve",
    "debug",
    "draft",
    "fusion_draft",
    "evolution",
    "fusion",
}


def _safe_json_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Convert nested persisted values without copying or pickling runtime objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)

    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return f"<NON_SERIALIZABLE:{type(value).__module__}.{type(value).__qualname__}>"

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            return {
                str(key): _safe_json_value(item, _seen=seen)
                for key, item in value.items()
            }
        finally:
            seen.discard(value_id)
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(value_id)
        try:
            return [_safe_json_value(item, _seen=seen) for item in value]
        finally:
            seen.discard(value_id)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(value_id)
        try:
            return {
                field.name: _safe_json_value(getattr(value, field.name), _seen=seen)
                for field in dataclasses.fields(value)
            }
        finally:
            seen.discard(value_id)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            converted = None
        if converted is not None and converted is not value:
            return _safe_json_value(converted, _seen=seen)

    return f"<NON_SERIALIZABLE:{type(value).__module__}.{type(value).__qualname__}>"


@dataclass(frozen=True)
class MetricSnapshot:
    value: float | None = None
    maximize: bool | None = None

    @classmethod
    def from_metric(cls, metric: MetricValue | None) -> "MetricSnapshot | None":
        if metric is None:
            return None
        value = metric.value
        return cls(
            value=float(value) if value is not None else None,
            maximize=metric.maximize,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "MetricSnapshot | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            return cls()
        raw_value = payload.get("value")
        try:
            value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            value = None
        maximize = payload.get("maximize")
        return cls(value=value, maximize=maximize if isinstance(maximize, bool) else None)

    def to_metric(self) -> MetricValue:
        return MetricValue(self.value, maximize=self.maximize)

    def to_payload(self) -> dict[str, Any]:
        return {"value": self.value, "maximize": self.maximize}


@dataclass(frozen=True)
class NodeSnapshot:
    """Serializable whitelist of SearchNode state.

    Runtime-only and dynamically attached attributes are deliberately absent.
    Graph references are represented by ids and rebuilt during loading.
    """

    code: str
    stage: str
    plan: str | None = None
    prompt_input: str | None = None
    step: int | None = None
    id: str = ""
    ctime: float = 0.0
    parent_id: str | None = None
    local_best_node_id: str | None = None
    term_out: Any = None
    exec_time: float | None = None
    exc_type: str | None = None
    exc_info: Any = None
    exc_stack: Any = None
    analysis: str | None = None
    parser_analysis: str | None = None
    decision_signals: Any = None
    llm_insight: str | None = None
    metric: MetricSnapshot | None = None
    is_buggy: bool | None = None
    is_valid: bool | None = None
    runtime_ok: bool = False
    search_eligible: bool = False
    score_recomputed: bool = False
    contract_valid: bool = False
    artifact_ready: bool = False
    delivery_ready: bool = False
    delivery_certified: bool = False
    certification_source: str = ""
    certification_notes: Any = None
    method_mode: str = "unknown"
    method_family: str = "unknown"
    sibling_ordinal: int | None = None
    expansion_complexity: str = "unknown"
    expansion_operator: str = "unknown"
    solution_interface: str = ""
    preflight_report: Any = None
    review_verdict: str = ""
    review_reason_codes: Any = None
    review_confidence: float | None = None
    visits: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    uct: float = 0.0
    is_debug_success: bool = False
    continue_improve: bool = False
    improve_failure_depth: int = 0
    lock: bool = False
    expected_child_count: int = 0
    next_sibling_ordinal: int = 0
    finish_time: str | None = None
    created_time: str | None = None
    alpha: int = 1
    beta: int = 1
    branch_id: int | None = None
    from_topk: bool = False
    code_summary: str | None = None
    work_dir: str | None = None

    @classmethod
    def from_node(
        cls,
        node: Any,
        *,
        omit_execution_details: bool = False,
    ) -> "NodeSnapshot":
        """Read only declared persistence fields; never traverse ``node.__dict__``."""
        parent = getattr(node, "parent", None)
        local_best = getattr(node, "local_best_node", None)
        return cls(
            code=str(getattr(node, "code", "") or ""),
            stage=str(getattr(node, "stage", "draft") or "draft"),
            plan=getattr(node, "plan", None),
            prompt_input=getattr(node, "prompt_input", None),
            step=getattr(node, "step", None),
            id=str(getattr(node, "id", "") or ""),
            ctime=float(getattr(node, "ctime", 0.0) or 0.0),
            parent_id=str(parent.id) if parent is not None else None,
            local_best_node_id=str(local_best.id) if local_best is not None else None,
            term_out=(
                "<OMITTED>"
                if omit_execution_details
                else _safe_json_value(getattr(node, "_term_out", None))
            ),
            exec_time=getattr(node, "exec_time", None),
            exc_type=getattr(node, "exc_type", None),
            exc_info=_safe_json_value(getattr(node, "exc_info", None)),
            exc_stack=(
                "<OMITTED>"
                if omit_execution_details
                else _safe_json_value(getattr(node, "exc_stack", None))
            ),
            analysis=getattr(node, "analysis", None),
            parser_analysis=getattr(node, "parser_analysis", None),
            decision_signals=_safe_json_value(getattr(node, "decision_signals", None)),
            llm_insight=getattr(node, "llm_insight", None),
            metric=MetricSnapshot.from_metric(getattr(node, "metric", None)),
            is_buggy=getattr(node, "is_buggy", None),
            is_valid=getattr(node, "is_valid", None),
            runtime_ok=bool(getattr(node, "runtime_ok", False)),
            search_eligible=bool(getattr(node, "search_eligible", False)),
            score_recomputed=bool(getattr(node, "score_recomputed", False)),
            contract_valid=bool(getattr(node, "contract_valid", False)),
            artifact_ready=bool(getattr(node, "artifact_ready", False)),
            delivery_ready=bool(getattr(node, "delivery_ready", False)),
            delivery_certified=bool(getattr(node, "delivery_certified", False)),
            certification_source=str(getattr(node, "certification_source", "") or ""),
            certification_notes=_safe_json_value(getattr(node, "certification_notes", [])),
            method_mode=str(getattr(node, "method_mode", "unknown") or "unknown"),
            method_family=str(getattr(node, "method_family", "unknown") or "unknown"),
            sibling_ordinal=getattr(node, "sibling_ordinal", None),
            expansion_complexity=str(getattr(node, "expansion_complexity", "unknown") or "unknown"),
            expansion_operator=str(getattr(node, "expansion_operator", "unknown") or "unknown"),
            solution_interface=str(getattr(node, "solution_interface", "") or ""),
            preflight_report=_safe_json_value(getattr(node, "preflight_report", None)),
            review_verdict=str(getattr(node, "review_verdict", "") or ""),
            review_reason_codes=_safe_json_value(getattr(node, "review_reason_codes", [])),
            review_confidence=getattr(node, "review_confidence", None),
            visits=int(getattr(node, "visits", 0) or 0),
            total_reward=float(getattr(node, "total_reward", 0.0) or 0.0),
            is_terminal=bool(getattr(node, "is_terminal", False)),
            uct=float(getattr(node, "_uct", 0.0) or 0.0),
            is_debug_success=bool(getattr(node, "is_debug_success", False)),
            continue_improve=bool(getattr(node, "continue_improve", False)),
            improve_failure_depth=int(getattr(node, "improve_failure_depth", 0) or 0),
            lock=bool(getattr(node, "lock", False)),
            expected_child_count=int(getattr(node, "expected_child_count", 0) or 0),
            next_sibling_ordinal=int(getattr(node, "next_sibling_ordinal", 0) or 0),
            finish_time=getattr(node, "finish_time", None),
            created_time=getattr(node, "created_time", None),
            alpha=int(getattr(node, "alpha", 1) or 1),
            beta=int(getattr(node, "beta", 1) or 1),
            branch_id=getattr(node, "branch_id", None),
            from_topk=bool(getattr(node, "from_topk", False)),
            code_summary=getattr(node, "code_summary", None),
            work_dir=(
                str(getattr(node, "work_dir"))
                if getattr(node, "work_dir", None) is not None
                else None
            ),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        parent_id: str | None = None,
        local_best_node_id: str | None = None,
    ) -> "NodeSnapshot":
        stage = str(payload.get("stage") or "draft")
        if stage not in _VALID_STAGES:
            stage = "draft"
        ctime = payload.get("ctime", 0.0)
        try:
            ctime_value = float(ctime or 0.0)
        except (TypeError, ValueError):
            ctime_value = 0.0
        return cls(
            code=str(payload.get("code") or ""),
            stage=stage,
            plan=payload.get("plan"),
            prompt_input=payload.get("prompt_input"),
            step=payload.get("step"),
            id=str(payload.get("id") or ""),
            ctime=ctime_value,
            parent_id=parent_id,
            local_best_node_id=local_best_node_id,
            term_out=_safe_json_value(payload.get("_term_out")),
            exec_time=payload.get("exec_time"),
            exc_type=payload.get("exc_type"),
            exc_info=_safe_json_value(payload.get("exc_info")),
            exc_stack=_safe_json_value(payload.get("exc_stack")),
            analysis=payload.get("analysis"),
            parser_analysis=payload.get("parser_analysis"),
            decision_signals=_safe_json_value(payload.get("decision_signals")),
            llm_insight=payload.get("llm_insight"),
            metric=MetricSnapshot.from_payload(payload.get("metric")),
            is_buggy=payload.get("is_buggy"),
            is_valid=payload.get("is_valid"),
            runtime_ok=bool(
                payload.get("runtime_ok", payload.get("is_buggy") is False)
            ),
            search_eligible=bool(
                payload.get(
                    "search_eligible",
                    payload.get("is_buggy") is False
                    and isinstance(payload.get("metric"), Mapping)
                    and payload.get("metric", {}).get("value") is not None,
                )
            ),
            score_recomputed=bool(payload.get("score_recomputed", False)),
            contract_valid=bool(payload.get("contract_valid", False)),
            artifact_ready=bool(payload.get("artifact_ready", False)),
            delivery_ready=bool(payload.get("delivery_ready", False)),
            delivery_certified=bool(payload.get("delivery_certified", False)),
            certification_source=str(payload.get("certification_source") or ""),
            certification_notes=_safe_json_value(payload.get("certification_notes") or []),
            method_mode=str(payload.get("method_mode") or "unknown"),
            method_family=str(payload.get("method_family") or "unknown"),
            sibling_ordinal=payload.get("sibling_ordinal"),
            expansion_complexity=str(payload.get("expansion_complexity") or "unknown"),
            expansion_operator=str(payload.get("expansion_operator") or "unknown"),
            solution_interface=str(payload.get("solution_interface") or ""),
            preflight_report=_safe_json_value(payload.get("preflight_report")),
            review_verdict=str(payload.get("review_verdict") or ""),
            review_reason_codes=_safe_json_value(payload.get("review_reason_codes") or []),
            review_confidence=payload.get("review_confidence"),
            visits=int(payload.get("visits") or 0),
            total_reward=float(payload.get("total_reward") or 0.0),
            is_terminal=bool(payload.get("is_terminal", False)),
            uct=float(payload.get("_uct") or 0.0),
            is_debug_success=bool(payload.get("is_debug_success", False)),
            continue_improve=bool(payload.get("continue_improve", False)),
            improve_failure_depth=int(payload.get("improve_failure_depth") or 0),
            lock=bool(payload.get("lock", False)),
            expected_child_count=int(payload.get("expected_child_count") or 0),
            next_sibling_ordinal=int(payload.get("next_sibling_ordinal") or 0),
            finish_time=payload.get("finish_time"),
            created_time=payload.get("created_time"),
            alpha=int(payload.get("alpha") or 1),
            beta=int(payload.get("beta") or 1),
            branch_id=payload.get("branch_id"),
            from_topk=bool(payload.get("from_topk", False)),
            code_summary=payload.get("code_summary"),
            work_dir=payload.get("work_dir"),
        )

    def to_node(self) -> Any:
        from engine.search_node import SearchNode

        stage = self.stage if self.stage in _VALID_STAGES else "draft"
        node = SearchNode(
            code=self.code,
            stage=stage,
            plan=self.plan,
            prompt_input=self.prompt_input,
            step=self.step,
            id=self.id,
            ctime=self.ctime,
            _term_out=self.term_out,
            exec_time=self.exec_time,
            exc_type=self.exc_type,
            exc_info=self.exc_info,
            exc_stack=self.exc_stack,
            analysis=self.analysis,
            parser_analysis=self.parser_analysis,
            decision_signals=self.decision_signals,
            llm_insight=self.llm_insight,
            metric=self.metric.to_metric() if self.metric is not None else None,
            is_buggy=self.is_buggy,
            is_valid=self.is_valid,
            runtime_ok=self.runtime_ok,
            search_eligible=self.search_eligible,
            score_recomputed=self.score_recomputed,
            contract_valid=self.contract_valid,
            artifact_ready=self.artifact_ready,
            delivery_ready=self.delivery_ready,
            delivery_certified=self.delivery_certified,
            certification_source=self.certification_source,
            certification_notes=list(self.certification_notes or []),
            method_mode=self.method_mode,
            method_family=self.method_family,
            sibling_ordinal=self.sibling_ordinal,
            expansion_complexity=self.expansion_complexity,
            expansion_operator=self.expansion_operator,
            solution_interface=self.solution_interface,
            preflight_report=self.preflight_report,
            review_verdict=self.review_verdict,
            review_reason_codes=list(self.review_reason_codes or []),
            review_confidence=self.review_confidence,
            visits=self.visits,
            total_reward=self.total_reward,
            is_terminal=self.is_terminal,
            _uct=self.uct,
            is_debug_success=self.is_debug_success,
            continue_improve=self.continue_improve,
            improve_failure_depth=self.improve_failure_depth,
            lock=self.lock,
            expected_child_count=self.expected_child_count,
            next_sibling_ordinal=self.next_sibling_ordinal,
            finish_time=self.finish_time,
            created_time=self.created_time,
            alpha=self.alpha,
            beta=self.beta,
            branch_id=self.branch_id,
            from_topk=self.from_topk,
            code_summary=self.code_summary,
            work_dir=self.work_dir,
        )
        node.child_count_lock = threading.Lock()
        return node

    def to_payload(self) -> dict[str, Any]:
        # Preserve the version-2 node shape for the service API and old tooling.
        return {
            "code": self.code,
            "plan": self.plan,
            "prompt_input": self.prompt_input,
            "step": self.step,
            "id": self.id,
            "ctime": self.ctime,
            "parent": None,
            "children": [],
            "_term_out": self.term_out,
            "exec_time": self.exec_time,
            "exc_type": self.exc_type,
            "exc_info": self.exc_info,
            "exc_stack": self.exc_stack,
            "analysis": self.analysis,
            "parser_analysis": self.parser_analysis,
            "decision_signals": self.decision_signals,
            "llm_insight": self.llm_insight,
            "metric": self.metric.to_payload() if self.metric is not None else None,
            "is_buggy": self.is_buggy,
            "is_valid": self.is_valid,
            "runtime_ok": self.runtime_ok,
            "search_eligible": self.search_eligible,
            "score_recomputed": self.score_recomputed,
            "contract_valid": self.contract_valid,
            "artifact_ready": self.artifact_ready,
            "delivery_ready": self.delivery_ready,
            "delivery_certified": self.delivery_certified,
            "certification_source": self.certification_source,
            "certification_notes": self.certification_notes,
            "method_mode": self.method_mode,
            "method_family": self.method_family,
            "sibling_ordinal": self.sibling_ordinal,
            "expansion_complexity": self.expansion_complexity,
            "expansion_operator": self.expansion_operator,
            "solution_interface": self.solution_interface,
            "preflight_report": self.preflight_report,
            "review_verdict": self.review_verdict,
            "review_reason_codes": self.review_reason_codes,
            "review_confidence": self.review_confidence,
            "stage": self.stage,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "is_terminal": self.is_terminal,
            "_uct": self.uct,
            "local_best_node": None,
            "is_debug_success": self.is_debug_success,
            "continue_improve": self.continue_improve,
            "improve_failure_depth": self.improve_failure_depth,
            "lock": self.lock,
            "child_count_lock": None,
            "expected_child_count": self.expected_child_count,
            "next_sibling_ordinal": self.next_sibling_ordinal,
            "finish_time": self.finish_time,
            "created_time": self.created_time,
            "alpha": self.alpha,
            "beta": self.beta,
            "branch_id": self.branch_id,
            "from_topk": self.from_topk,
            "code_summary": self.code_summary,
            "work_dir": self.work_dir,
        }


@dataclass(frozen=True)
class JournalSnapshot:
    nodes: tuple[NodeSnapshot, ...] = ()

    @classmethod
    def from_journal(
        cls,
        journal: Any,
        *,
        node_ids: Iterable[str] | None = None,
        omit_execution_details: bool = False,
    ) -> "JournalSnapshot":
        included = set(node_ids) if node_ids is not None else None
        return cls(
            nodes=tuple(
                NodeSnapshot.from_node(
                    node,
                    omit_execution_details=omit_execution_details,
                )
                for node in journal.nodes
                if included is None or node.id in included
            )
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "JournalSnapshot":
        parent_map = payload.get("node2parent") or {}
        local_best_map = payload.get("node2best_local_node") or {}
        node_payloads = payload.get("nodes") or []
        return cls(
            nodes=tuple(
                NodeSnapshot.from_payload(
                    node_payload,
                    parent_id=parent_map.get(str(node_payload.get("id") or "")),
                    local_best_node_id=local_best_map.get(str(node_payload.get("id") or "")),
                )
                for node_payload in node_payloads
                if isinstance(node_payload, Mapping)
            )
        )

    def to_journal(self) -> Any:
        from engine.search_node import Journal

        nodes = [snapshot.to_node() for snapshot in self.nodes]
        id_to_node = {node.id: node for node in nodes}
        for node in nodes:
            node.parent = None
            node.children = set()
            node.local_best_node = None
            node.child_count_lock = threading.Lock()
        for snapshot, node in zip(self.nodes, nodes):
            if snapshot.parent_id in id_to_node:
                parent = id_to_node[snapshot.parent_id]
                node.parent = parent
                parent.children.add(node)
            if snapshot.local_best_node_id in id_to_node:
                node.local_best_node = id_to_node[snapshot.local_best_node_id]
        return Journal(nodes=nodes)

    def to_payload(self) -> dict[str, Any]:
        included_ids = {node.id for node in self.nodes}
        return {
            "nodes": [node.to_payload() for node in self.nodes],
            "node2parent": {
                node.id: node.parent_id
                for node in self.nodes
                if node.parent_id in included_ids
            },
            "node2best_local_node": {
                node.id: node.local_best_node_id
                for node in self.nodes
                if node.local_best_node_id in included_ids
            },
            "__version": "3",
        }
