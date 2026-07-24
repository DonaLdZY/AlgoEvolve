"""Durable runtime state for resuming in-flight MLEvolve search actions."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.journal_snapshot import NodeSnapshot
from engine.expansion_profile import ExpansionProfile


SEARCH_STATE_SCHEMA_VERSION = "mlevolve.search_state.v1"
ACTIVE_ACTION_STATUSES = {"scheduled", "generating", "generated", "executing"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


@dataclass(frozen=True)
class ActiveSearchAction:
    action_id: str
    parent_node_id: str
    status: str = "scheduled"
    parent_from_topk: bool = False
    expansion_profile: ExpansionProfile | None = None
    generated_node: NodeSnapshot | None = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActiveSearchAction | None":
        action_id = str(payload.get("action_id") or "").strip()
        parent_node_id = str(payload.get("parent_node_id") or "").strip()
        status = str(payload.get("status") or "scheduled").strip()
        if not action_id or not parent_node_id or status not in ACTIVE_ACTION_STATUSES:
            return None

        generated_node = None
        node_payload = payload.get("generated_node")
        if isinstance(node_payload, Mapping):
            raw_snapshot = node_payload.get("snapshot")
            if isinstance(raw_snapshot, Mapping):
                generated_node = NodeSnapshot.from_payload(
                    raw_snapshot,
                    parent_id=str(node_payload.get("parent_id") or parent_node_id),
                    local_best_node_id=(
                        str(node_payload.get("local_best_node_id"))
                        if node_payload.get("local_best_node_id")
                        else None
                    ),
                )

        return cls(
            action_id=action_id,
            parent_node_id=parent_node_id,
            status=status,
            parent_from_topk=bool(payload.get("parent_from_topk", False)),
            expansion_profile=ExpansionProfile.from_payload(payload.get("expansion_profile")),
            generated_node=generated_node,
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )

    def to_payload(self) -> dict[str, Any]:
        generated_node = None
        if self.generated_node is not None:
            generated_node = {
                "snapshot": self.generated_node.to_payload(),
                "parent_id": self.generated_node.parent_id,
                "local_best_node_id": self.generated_node.local_best_node_id,
            }
        return {
            "action_id": self.action_id,
            "parent_node_id": self.parent_node_id,
            "status": self.status,
            "parent_from_topk": self.parent_from_topk,
            "expansion_profile": (
                self.expansion_profile.to_payload() if self.expansion_profile is not None else None
            ),
            "generated_node": generated_node,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SearchRuntimeState:
    cumulative_search_elapsed_seconds: float = 0.0
    python_random_state: Any = None
    active_actions: tuple[ActiveSearchAction, ...] = ()
    last_checkpoint_at: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SearchRuntimeState":
        if str(payload.get("schema_version") or "") != SEARCH_STATE_SCHEMA_VERSION:
            return cls()
        try:
            elapsed = max(0.0, float(payload.get("cumulative_search_elapsed_seconds") or 0.0))
        except (TypeError, ValueError):
            elapsed = 0.0
        actions = []
        for item in payload.get("active_actions") or []:
            if isinstance(item, Mapping):
                action = ActiveSearchAction.from_payload(item)
                if action is not None:
                    actions.append(action)
        return cls(
            cumulative_search_elapsed_seconds=elapsed,
            python_random_state=payload.get("python_random_state"),
            active_actions=tuple(actions),
            last_checkpoint_at=str(payload.get("last_checkpoint_at") or ""),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SEARCH_STATE_SCHEMA_VERSION,
            "cumulative_search_elapsed_seconds": self.cumulative_search_elapsed_seconds,
            "python_random_state": self.python_random_state,
            "active_actions": [action.to_payload() for action in self.active_actions],
            "last_checkpoint_at": self.last_checkpoint_at,
        }


def load_search_runtime_state(path: Path) -> SearchRuntimeState:
    if not path.exists():
        return SearchRuntimeState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return SearchRuntimeState()
    return SearchRuntimeState.from_payload(payload)


class SearchRuntimeStateStore:
    """Thread-safe state store. Configuration is intentionally not persisted here."""

    def __init__(
        self,
        path: Path,
        *,
        initial_state: SearchRuntimeState | None = None,
        enabled: bool = True,
        checkpoint_seconds: float = 5.0,
        write_max_attempts: int = 5,
        write_retry_delay_seconds: float = 0.05,
    ) -> None:
        state = initial_state or SearchRuntimeState()
        self.path = path
        self.enabled = enabled
        self.checkpoint_seconds = max(0.1, float(checkpoint_seconds))
        self.write_max_attempts = max(1, int(write_max_attempts))
        self.write_retry_delay_seconds = max(0.0, float(write_retry_delay_seconds))
        self._base_elapsed = max(0.0, state.cumulative_search_elapsed_seconds)
        self._initial_random_state = state.python_random_state
        self._session_started = time.monotonic()
        self._actions = {action.action_id: action for action in state.active_actions}
        self._lock = threading.RLock()
        self._last_checkpoint_monotonic = 0.0
        self._accept_updates = True
        self._stop_checkpointing = threading.Event()
        self._checkpoint_thread: threading.Thread | None = None

    @property
    def cumulative_elapsed_at_start(self) -> float:
        return self._base_elapsed

    def elapsed_seconds(self) -> float:
        return self._base_elapsed + max(0.0, time.monotonic() - self._session_started)

    def actions(self) -> list[ActiveSearchAction]:
        with self._lock:
            return list(self._actions.values())

    def get_action(self, action_id: str) -> ActiveSearchAction | None:
        with self._lock:
            return self._actions.get(action_id)

    def register_action(
        self,
        parent_node_id: str,
        *,
        action_id: str | None = None,
        status: str = "scheduled",
        parent_from_topk: bool = False,
        expansion_profile: ExpansionProfile | None = None,
    ) -> ActiveSearchAction:
        now = _utc_now()
        action = ActiveSearchAction(
            action_id=action_id or uuid.uuid4().hex,
            parent_node_id=str(parent_node_id),
            status=status,
            parent_from_topk=parent_from_topk,
            expansion_profile=expansion_profile,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if self._accept_updates:
                self._actions[action.action_id] = action
                self._checkpoint_locked(force=True)
        return action

    def update_action(
        self,
        action_id: str,
        *,
        status: str,
        generated_node: Any | None = None,
    ) -> ActiveSearchAction | None:
        if status not in ACTIVE_ACTION_STATUSES:
            raise ValueError(f"Unsupported active action status: {status}")
        with self._lock:
            if not self._accept_updates:
                return self._actions.get(action_id)
            current = self._actions.get(action_id)
            if current is None:
                return None
            snapshot = current.generated_node
            if generated_node is not None:
                snapshot = NodeSnapshot.from_node(generated_node)
            updated = ActiveSearchAction(
                action_id=current.action_id,
                parent_node_id=current.parent_node_id,
                status=status,
                parent_from_topk=current.parent_from_topk,
                expansion_profile=current.expansion_profile,
                generated_node=snapshot,
                created_at=current.created_at,
                updated_at=_utc_now(),
            )
            self._actions[action_id] = updated
            self._checkpoint_locked(force=True)
            return updated

    def remove_action(self, action_id: str) -> ActiveSearchAction | None:
        with self._lock:
            action = self._actions.pop(action_id, None)
            if self._accept_updates:
                self._checkpoint_locked(force=True)
            return action

    def replace_actions(self, actions: Iterable[ActiveSearchAction]) -> None:
        with self._lock:
            self._actions = {action.action_id: action for action in actions}
            self._checkpoint_locked(force=True)

    def restore_random_state(self) -> bool:
        if not self.enabled:
            return False
        try:
            state = self._initial_random_state
            if state is None:
                return False
            random.setstate(_tuple_tree(state))
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def checkpoint_if_due(self) -> None:
        with self._lock:
            self._checkpoint_locked(force=False)

    def checkpoint(self) -> None:
        with self._lock:
            self._checkpoint_locked(force=True)

    def start_periodic_checkpointing(self) -> None:
        if not self.enabled or self._checkpoint_thread is not None:
            return

        def _checkpoint_loop() -> None:
            while not self._stop_checkpointing.wait(self.checkpoint_seconds):
                try:
                    self.checkpoint_if_due()
                except Exception:
                    continue

        self._checkpoint_thread = threading.Thread(
            target=_checkpoint_loop,
            name="mlevolve-search-state-checkpoint",
            daemon=True,
        )
        self._checkpoint_thread.start()

    def close(
        self,
        *,
        clear_actions: bool,
        replacement_actions: Iterable[ActiveSearchAction] | None = None,
    ) -> None:
        self._stop_checkpointing.set()
        with self._lock:
            if replacement_actions is not None:
                self._actions = {
                    action.action_id: action for action in replacement_actions
                }
            elif clear_actions:
                self._actions.clear()
            self._checkpoint_locked(force=True)
            self._accept_updates = False
        if self._checkpoint_thread is not None:
            self._checkpoint_thread.join(timeout=max(1.0, self.checkpoint_seconds + 0.5))

    def _checkpoint_locked(self, *, force: bool) -> None:
        if not self.enabled:
            return
        now_monotonic = time.monotonic()
        if not force and now_monotonic - self._last_checkpoint_monotonic < self.checkpoint_seconds:
            return
        state = SearchRuntimeState(
            cumulative_search_elapsed_seconds=self.elapsed_seconds(),
            python_random_state=random.getstate(),
            active_actions=tuple(self._actions.values()),
            last_checkpoint_at=_utc_now(),
        )
        self._write_payload(state.to_payload())
        self._last_checkpoint_monotonic = now_monotonic

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(data, encoding="utf-8")
        last_error: Exception | None = None
        for attempt in range(self.write_max_attempts):
            try:
                tmp_path.replace(self.path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(self.write_retry_delay_seconds * (attempt + 1))
        try:
            self.path.write_text(data, encoding="utf-8")
            tmp_path.unlink(missing_ok=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            if last_error is not None:
                raise last_error
            raise


def prune_completed_actions(
    actions: Iterable[ActiveSearchAction],
    journal_node_ids: set[str],
) -> list[ActiveSearchAction]:
    return [
        action
        for action in actions
        if action.generated_node is None or action.generated_node.id not in journal_node_ids
    ]


def retain_generated_actions(
    actions: Iterable[ActiveSearchAction],
) -> list[ActiveSearchAction]:
    """Keep only interrupted work whose generated code can be resumed exactly."""
    return [action for action in actions if action.generated_node is not None]


def retain_one_action_per_parent(
    actions: Iterable[ActiveSearchAction],
) -> list[ActiveSearchAction]:
    """Keep one resumable reservation per parent, preferring generated code."""
    chosen: dict[str, tuple[int, ActiveSearchAction]] = {}
    for index, action in enumerate(actions):
        current = chosen.get(action.parent_node_id)
        if current is None:
            chosen[action.parent_node_id] = (index, action)
            continue
        _, current_action = current
        if current_action.generated_node is None and action.generated_node is not None:
            chosen[action.parent_node_id] = (index, action)
    return [
        action
        for _, action in sorted(chosen.values(), key=lambda item: item[0])
    ]


def reconcile_runtime_locks(nodes: Iterable[Any], actions: Iterable[ActiveSearchAction]) -> set[str]:
    """Make transient node locks exactly match parents with active search actions."""
    active_parent_ids = {action.parent_node_id for action in actions}
    for node in nodes:
        node.lock = node.id in active_parent_ids
    return active_parent_ids


def repair_journal_for_resume(journal: Any, actions: Iterable[ActiveSearchAction]) -> Any:
    """Rebuild graph links and reserve child slots for genuinely active actions."""
    if len(journal) == 0:
        return journal
    action_list = list(actions)
    id_to_node = {node.id: node for node in journal.nodes}
    active_counts: dict[str, int] = {}
    for action in action_list:
        if action.parent_node_id in id_to_node:
            active_counts[action.parent_node_id] = active_counts.get(action.parent_node_id, 0) + 1

    reconcile_runtime_locks(journal.nodes, action_list)
    for node in journal.nodes:
        node.children = set()
        node.child_count_lock = threading.Lock()
    for node in journal.nodes:
        parent = getattr(node, "parent", None)
        if parent is not None and parent.id in id_to_node:
            node.parent = id_to_node[parent.id]
            node.parent.children.add(node)
    for node in journal.nodes:
        node.expected_child_count = len(node.children) + active_counts.get(node.id, 0)
        materialized_max = max(
            [int(getattr(child, "sibling_ordinal", 0) or 0) for child in node.children],
            default=0,
        )
        active_max = max(
            [
                action.expansion_profile.sibling_ordinal
                for action in action_list
                if action.parent_node_id == node.id and action.expansion_profile is not None
            ],
            default=0,
        )
        node.next_sibling_ordinal = max(
            int(getattr(node, "next_sibling_ordinal", 0) or 0),
            materialized_max,
            active_max,
        )
    return journal


def restore_generated_node(action: ActiveSearchAction, journal: Any) -> Any | None:
    if action.generated_node is None:
        return None
    id_to_node = {node.id: node for node in journal.nodes}
    parent = id_to_node.get(action.parent_node_id)
    if parent is None:
        return None
    node = action.generated_node.to_node()
    node.parent = parent
    parent.children.add(node)
    local_best_id = action.generated_node.local_best_node_id
    if local_best_id == node.id:
        node.local_best_node = node
    elif local_best_id in id_to_node:
        node.local_best_node = id_to_node[local_best_id]
    node.pending_execution = True
    node.runtime_action_id = action.action_id
    return node
