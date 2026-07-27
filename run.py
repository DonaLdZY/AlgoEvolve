import atexit
import json
import logging
import os
import signal
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional


def _print_cli_help() -> None:
    print(
        """usage: python run.py [key=value ...]

Run the AlgoEvolve search engine with OmegaConf dotted-key overrides.

Required configuration:
  data_dir=PATH                  Source data or AutoRealize output directory
  desc_file=PATH                 Task description markdown
    or goal=TEXT                 Inline task goal when desc_file is omitted

Common overrides:
  exp_id=ID                      External experiment identifier
  exp_name=NAME                  Human-readable run name
  log_dir=PATH                   Log output root
  workspace_dir=PATH             Execution workspace root
  agent.steps=50                 Maximum search nodes/steps
  agent.time_limit=10800         Search wall-clock limit in seconds
  agent.initial_drafts=3         Number of initial drafts
  agent.search.parallel_search_num=4
  runtime.resume_run=true        Resume from existing log/workspace paths

Configuration precedence:
  config/config.yaml < ALGOEVOLVE_CONFIG_PATH file < key=value overrides
  (legacy MLEVOLVE_CONFIG_PATH is also accepted)

Examples:
  python run.py data_dir=./input desc_file=./description.md exp_name=demo
  python run.py runtime.resume_run=true log_dir=./existing/logs workspace_dir=./existing/workspace data_dir=./input desc_file=./description.md

The complete documented configuration is config/config.yaml.
"""
    )


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _print_cli_help()
    raise SystemExit(0)


import torch
from omegaconf import OmegaConf
from rich.status import Status

from config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from engine.agent_search import AgentSearch as Agent
from engine.evaluation import ExpansionReservation
from engine.coldstart import build_guidance_description
from engine.executor import Interpreter
from engine.expansion_profile import ExpansionProfile
from engine.node_selection import refresh_persisted_uct_values
from engine.search_node import Journal
from engine.search_budget import resolve_search_budget
from engine.search_runtime_state import (
    SearchRuntimeState,
    SearchRuntimeStateStore,
    load_search_runtime_state,
    prune_completed_actions,
    reconcile_runtime_locks,
    repair_journal_for_resume,
    retain_generated_actions,
    retain_one_action_per_parent,
    restore_generated_node,
)
from engine.solution_manager import persist_resumable_checkpoint
from agents.prompt_policy import infer_task_family
from utils.seed import set_global_seed
from utils.logging_config import setup_logging
from utils.visualization import journal_to_string_tree
from utils.serialize import load_json


PENDING_NODES_FILE = "pending_nodes.json"
RUN_STATUS_FILE = "run_status.json"
PENDING_DRAFT_STATUSES = {"generating", "pending_execution", "executing", "reviewing", "cancelled", "failed"}


def _node_attr(node, name: str, default=None):
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _interrupted_budget_targets(cfg) -> tuple[int, int]:
    """Read effective targets from the interrupted continuation session."""

    runtime_cfg = getattr(cfg, "runtime", None)
    filename = str(
        getattr(runtime_cfg, "run_status_filename", RUN_STATUS_FILE) or RUN_STATUS_FILE
    )
    path = cfg.log_dir / filename
    if not path.exists():
        return 0, 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 0, 0
    if str(payload.get("status") or "") not in {
        "interrupted_resumable",
        "interrupted_incomplete",
    }:
        return 0, 0
    return (
        max(0, int(payload.get("total_steps") or 0)),
        max(0, int(payload.get("time_limit_secs") or 0)),
    )


def _pending_node_row(node, status: str) -> dict:
    parent = _node_attr(node, "parent", None)
    metric = _node_attr(node, "metric", None)
    metric_value = getattr(metric, "value", None) if metric is not None else None
    metric_maximize = getattr(metric, "maximize", None) if metric is not None else None
    return {
        "id": str(_node_attr(node, "id", "")),
        "parent_id": _node_attr(node, "parent_id", None) or getattr(parent, "id", None),
        "stage": _node_attr(node, "stage", "draft"),
        "plan": _node_attr(node, "plan", None),
        "code": _node_attr(node, "code", None),
        "result": "",
        "insight": _node_attr(node, "llm_insight", None) or _node_attr(node, "analysis", None),
        "llm_insight": _node_attr(node, "llm_insight", None),
        "parser_analysis": _node_attr(node, "parser_analysis", None) or _node_attr(node, "analysis", None),
        "decision_signals": _node_attr(node, "decision_signals", None),
        "metric": metric_value,
        "maximize": metric_maximize if isinstance(metric_maximize, bool) else None,
        "is_buggy": _node_attr(node, "is_buggy", None),
        "is_valid": _node_attr(node, "is_valid", None),
        "visits": _node_attr(node, "visits", 0),
        "total_reward": _node_attr(node, "total_reward", 0.0),
        "uct": _node_attr(node, "_uct", None),
        "finish_time": _node_attr(node, "finish_time", None),
        "exec_time": _node_attr(node, "exec_time", None),
        "branch_id": _node_attr(node, "branch_id", None),
        "from_topk": _node_attr(node, "from_topk", None),
        "created_time": _node_attr(node, "created_time", None),
        "status": status,
        "pending_execution": status in {"generating", "pending_execution", "executing", "reviewing"},
        "label": {
            "generating": "Draft code is being generated",
            "pending_execution": "Draft generated, pending execution",
            "executing": "Draft execution is running",
            "reviewing": "Draft execution finished; result review is running",
            "cancelled": "Draft execution was cancelled before journal append",
            "failed": "Draft generation failed before execution",
        }.get(status, status),
    }


def _active_action_placeholder(action, parent) -> dict:
    profile = action.expansion_profile
    if getattr(parent, "stage", "") == "root":
        stage = "draft"
    elif bool(getattr(parent, "is_buggy", False)):
        stage = "debug"
    else:
        stage = "improve"
    return {
        "id": f"action-{action.action_id}",
        "runtime_action_id": action.action_id,
        "parent": parent,
        "parent_id": action.parent_node_id,
        "stage": stage,
        "plan": f"Worker is expanding parent {action.parent_node_id}.",
        "code": "",
        "metric": None,
        "is_buggy": None,
        "is_valid": None,
        "visits": 0,
        "total_reward": 0.0,
        "_uct": None,
        "branch_id": getattr(parent, "branch_id", None),
        "from_topk": bool(action.parent_from_topk),
        "created_time": action.created_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sibling_ordinal": getattr(profile, "sibling_ordinal", None),
        "expansion_complexity": getattr(profile, "complexity", "moderate"),
        "expansion_operator": getattr(profile, "operator", "auto"),
    }


def _write_run_status(
    cfg,
    *,
    status: str,
    termination_reason: str,
    completed_steps: int,
    total_steps: int,
    time_limit_secs: int,
) -> None:
    payload = {
        "schema_version": "algoevolve.run_status.v1",
        "status": status,
        "termination_reason": termination_reason,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "time_limit_secs": time_limit_secs,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    runtime_cfg = getattr(cfg, "runtime", None)
    filename = str(getattr(runtime_cfg, "run_status_filename", RUN_STATUS_FILE) or RUN_STATUS_FILE)
    path = cfg.log_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _exit_process_immediately(exit_code: int) -> None:
    """End the whole worker process without waiting for non-cancellable LLM threads."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(exit_code)


def _write_pending_nodes_state(cfg, nodes, status_by_id: dict[str, str], phase: str) -> None:
    runtime_cfg = getattr(cfg, "runtime", None)
    draft_cfg = getattr(getattr(cfg, "agent", None), "draft", None)
    if not bool(getattr(runtime_cfg, "write_pending_nodes", True)) or not bool(
        getattr(draft_cfg, "show_pending_draft_nodes", True)
    ):
        return
    filename = str(getattr(runtime_cfg, "pending_nodes_filename", PENDING_NODES_FILE) or PENDING_NODES_FILE)
    path = cfg.log_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for node in nodes:
        node_id = str(getattr(node, "id", ""))
        status = status_by_id.get(node_id)
        if status:
            rows.append(_pending_node_row(node, status))
    payload = {
        "schema_version": "algoevolve.pending_nodes.v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": phase,
        "nodes": rows,
    }
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    last_error: Exception | None = None
    max_attempts = max(1, int(getattr(runtime_cfg, "state_write_max_attempts", 5)))
    retry_delay = max(0.0, float(getattr(runtime_cfg, "state_write_retry_delay_seconds", 0.05)))
    for attempt in range(max_attempts):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(retry_delay * (attempt + 1))
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        if last_error is not None:
            raise last_error
        raise


def run():
    cfg = load_cfg()
    runtime_cfg = getattr(cfg, "runtime", None)
    env_resume = (
        os.environ.get("ALGOEVOLVE_RESUME_RUN", "").strip()
        or os.environ.get("MLEVOLVE_RESUME_RUN", "").strip()
    ).lower()
    resume_run = bool(getattr(runtime_cfg, "resume_run", False))
    if env_resume:
        resume_run = env_resume in {"1", "true", "yes", "on"}
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg, task_desc=task_desc)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    interruption_requested = threading.Event()
    normal_exit = {"done": False}

    def request_interruption(signum, _frame):
        interruption_requested.set()
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            try:
                signal.signal(signal_value, request_interruption)
            except (OSError, RuntimeError, ValueError):
                pass

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if (
            global_step == 0
            and not resume_run
            and not interruption_requested.is_set()
            and bool(getattr(runtime_cfg, "cleanup_empty_workspace_on_exit", True))
        ):
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    search_state_path = cfg.log_dir / str(
        getattr(runtime_cfg, "search_state_filename", "search_state.json") or "search_state.json"
    )
    initial_search_state = SearchRuntimeState()
    if resume_run and search_state_path.exists():
        try:
            initial_search_state = load_search_runtime_state(search_state_path)
        except Exception as exc:
            logger.warning("Failed to load search runtime state; journal resume will continue without it: %s", exc)

    search_state = SearchRuntimeStateStore(
        search_state_path,
        initial_state=initial_search_state,
        enabled=bool(getattr(runtime_cfg, "save_search_state", True)),
        checkpoint_seconds=float(getattr(runtime_cfg, "search_state_checkpoint_seconds", 5.0)),
        write_max_attempts=int(getattr(runtime_cfg, "state_write_max_attempts", 5)),
        write_retry_delay_seconds=float(
            getattr(runtime_cfg, "state_write_retry_delay_seconds", 0.05)
        ),
    )
    if resume_run and bool(getattr(runtime_cfg, "restore_random_state", True)):
        if search_state.restore_random_state():
            logger.info("Restored Python random state from %s", search_state_path)

    restored_actions = (
        search_state.actions()
        if resume_run and bool(getattr(runtime_cfg, "restore_inflight_actions", True))
        else []
    )

    resume_path = cfg.log_dir / "journal.json"
    if resume_path.exists():
        try:
            journal = load_json(resume_path, Journal)
            logger.info(f"Resuming from existing journal: {resume_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing journal, starting fresh: {e}")
            journal = Journal()
    else:
        journal = Journal()

    journal_node_ids = {node.id for node in journal.nodes}
    restored_actions = prune_completed_actions(restored_actions, journal_node_ids)
    valid_parent_ids = journal_node_ids
    restored_actions = [
        action for action in restored_actions if action.parent_node_id in valid_parent_ids
    ]
    restored_actions = retain_one_action_per_parent(restored_actions)
    discarded_unmaterialized_actions = [
        action for action in restored_actions if action.generated_node is None
    ]
    restored_actions = [
        action for action in restored_actions if action.generated_node is not None
    ]
    if discarded_unmaterialized_actions:
        logger.info(
            "Discarding %s interrupted actions without generated code; "
            "their parents will be selected again from current UCT state.",
            len(discarded_unmaterialized_actions),
        )
    search_state.replace_actions(restored_actions)
    search_state.start_periodic_checkpointing()
    journal = repair_journal_for_resume(journal, restored_actions)
    requested_steps = max(0, int(cfg.agent.steps))
    requested_time_limit = max(0, int(getattr(cfg.agent, "time_limit", 0) or 0))
    resume_budget_mode = str(
        getattr(runtime_cfg, "resume_budget_mode", "total") or "total"
    ).strip().lower()
    restored_completed = max(0, len(journal) - 1)
    restored_elapsed = max(0.0, search_state.elapsed_seconds())
    preserved_target_steps, preserved_time_limit = _interrupted_budget_targets(cfg)
    cfg.agent.steps, cfg.agent.time_limit = resolve_search_budget(
        requested_steps=requested_steps,
        requested_time_limit=requested_time_limit,
        restored_completed=restored_completed,
        restored_elapsed=restored_elapsed,
        resume_run=resume_run,
        resume_budget_mode=resume_budget_mode,
        preserved_target_steps=preserved_target_steps,
        preserved_time_limit=preserved_time_limit,
    )
    if resume_run and resume_budget_mode == "additional":
        logger.info(
            "Appending resume budget: completed=%s + steps=%s -> target=%s; "
            "elapsed=%.1fs + time=%ss -> target=%ss",
            restored_completed,
            requested_steps,
            cfg.agent.steps,
            restored_elapsed,
            requested_time_limit,
            cfg.agent.time_limit,
        )
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    agent.restore_search_elapsed(search_state.elapsed_seconds())
    agent.runtime_checkpoint_callback = lambda action_id, action_status, node: search_state.update_action(
        action_id,
        status=action_status,
        generated_node=node,
    )

    checkpoint_lock = threading.Lock()
    checkpoint_committed = {"done": False}

    def commit_interrupted_checkpoint() -> bool:
        if checkpoint_committed["done"]:
            return True
        if normal_exit["done"]:
            return False
        with checkpoint_lock:
            if checkpoint_committed["done"]:
                return True
            if normal_exit["done"]:
                return False
            interruption_requested.set()
            search_state_closed = False
            try:
                # Freeze action updates before serializing the final interrupt
                # checkpoint so search_state.json and the manifest agree.
                agent.accept_search_results = False
                agent.runtime_checkpoint_callback = None
                all_actions = search_state.actions()
                resumable_actions = retain_one_action_per_parent(
                    retain_generated_actions(all_actions)
                )
                discarded_count = len(all_actions) - len(resumable_actions)
                search_state.close(
                    clear_actions=False,
                    replacement_actions=resumable_actions,
                )
                search_state_closed = True
                active_actions = [
                    action.to_payload() for action in resumable_actions
                ]
                with agent.journal_lock:
                    repair_journal_for_resume(journal, resumable_actions)
                    save_run(cfg, journal)
                _write_run_status(
                    cfg,
                    status="interrupted_resumable",
                    termination_reason="external_interrupt",
                    completed_steps=max(0, len(journal) - 1),
                    total_steps=int(cfg.agent.steps),
                    time_limit_secs=int(getattr(cfg.agent, "time_limit", 0) or 0),
                )
                # Manifests are written last and therefore serve as the service's
                # durable readiness marker for terminating the process tree.
                persist_resumable_checkpoint(
                    agent,
                    status="interrupted_resumable",
                    reason="external_interrupt",
                    active_actions=active_actions,
                    manifest_filename=str(
                        getattr(
                            runtime_cfg,
                            "checkpoint_manifest_filename",
                            "checkpoint_manifest.json",
                        )
                    ),
                    materialize_artifacts=False,
                )
                checkpoint_committed["done"] = True
                logger.info(
                    "Interrupted checkpoint committed: resumable_generated_actions=%s, "
                    "discarded_unmaterialized_actions=%s",
                    len(resumable_actions),
                    discarded_count,
                )
                return True
            except Exception:
                logger.exception("Failed to commit interrupted resumable checkpoint")
                return False
            finally:
                if not search_state_closed:
                    try:
                        search_state.close(clear_actions=False)
                    except Exception:
                        logger.exception("Failed to finalize interrupted search state")

    atexit.register(commit_interrupted_checkpoint)
    logger.info(
        "Search resume state: elapsed=%.1fs, active_actions=%d. "
        "Current YAML/CLI settings remain authoritative.",
        search_state.elapsed_seconds(),
        len(restored_actions),
    )

    restored_generated_nodes: dict[str, object] = {}
    invalid_restored_action_ids: set[str] = set()
    for action in restored_actions:
        if action.generated_node is None:
            invalid_restored_action_ids.add(action.action_id)
            continue
        node = restore_generated_node(action, journal)
        if node is None:
            invalid_restored_action_ids.add(action.action_id)
            continue
        restored_generated_nodes[action.action_id] = node
        if getattr(node, "stage", None) == "fusion_draft":
            agent.fusion_draft_count += 1
        branch_id = getattr(node, "branch_id", None)
        if branch_id is not None:
            branch_nodes = agent.branch_all_nodes.setdefault(branch_id, [])
            if all(existing.id != node.id for existing in branch_nodes):
                branch_nodes.append(node)
            agent.branch_successful_nodes.setdefault(branch_id, [])
            try:
                agent.next_branch_id = max(agent.next_branch_id, int(branch_id) + 1)
            except (TypeError, ValueError):
                pass
    if invalid_restored_action_ids:
        search_state.replace_actions(
            action
            for action in search_state.actions()
            if action.action_id not in invalid_restored_action_ids
        )
        repair_journal_for_resume(journal, search_state.actions())

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    completed = max(0, len(journal) - 1)
    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def _reconcile_parent_runtime_state(parent_node_id: str) -> None:
        parent = next((node for node in journal.nodes if node.id == parent_node_id), None)
        if parent is None:
            return
        journal_ids = {node.id for node in journal.nodes}
        remaining_actions = [
            action for action in search_state.actions() if action.parent_node_id == parent_node_id
        ]
        active_generated_ids = {
            action.generated_node.id
            for action in remaining_actions
            if action.generated_node is not None
        }
        parent.children = {
            child
            for child in getattr(parent, "children", set())
            if child.id in journal_ids or child.id in active_generated_ids
        }
        unmaterialized_count = sum(
            1 for action in remaining_actions if action.generated_node is None
        )
        parent.expected_child_count = len(parent.children) + unmaterialized_count
        parent.lock = bool(remaining_actions)

        active_ids = {
            action.generated_node.id
            for action in search_state.actions()
            if action.generated_node is not None
        }
        for branch_id, branch_nodes in list(agent.branch_all_nodes.items()):
            agent.branch_all_nodes[branch_id] = [
                node for node in branch_nodes if node.id in journal_ids or node.id in active_ids
            ]

    def finish_search_action(action_id: str) -> None:
        action = search_state.remove_action(action_id)
        if action is not None:
            _reconcile_parent_runtime_state(action.parent_node_id)

    def register_search_action(parent, *, parent_from_topk: bool = False):
        profile = ExpansionProfile.create(
            parent.reserve_sibling_ordinal(),
            task_family=infer_task_family(agent),
        )
        return search_state.register_action(
            parent.id,
            parent_from_topk=parent_from_topk,
            expansion_profile=profile,
        )

    def step_task(
        action_id: str,
        node,
        *,
        reuse_reserved_slot: bool = False,
        execute_immediately: bool = True,
        return_result_node: bool = False,
        draft_single_call: bool = False,
        fast_draft_mode: bool = False,
        expansion_reservation=None,
    ):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info("[step_task] Processing virtual root node.")
        if reuse_reserved_slot:
            with node.child_count_lock:
                node.expected_child_count = max(len(node.children), node.expected_child_count - 1)
        search_state.update_action(action_id, status="generating")
        action = search_state.get_action(action_id)
        expansion_profile = action.expansion_profile if action is not None else None
        if expansion_profile is None:
            expansion_profile = ExpansionProfile.create(
                node.reserve_sibling_ordinal(),
                task_family=infer_task_family(agent),
            )
        return agent.step(
            exec_callback=exec_callback,
            node=node,
            execute_immediately=execute_immediately,
            runtime_action_id=action_id,
            parent_preselected=True,
            return_result_node=return_result_node,
            expansion_profile=expansion_profile,
            draft_single_call=draft_single_call,
            fast_draft_mode=fast_draft_mode,
            expansion_reservation=expansion_reservation,
        )

    def schedule_new_step(executor):
        # Selection, parent reservation, and virtual visits are one atomic tree
        # operation. A second worker therefore observes the updated UCT state
        # and skips the parent already owned by the first worker.
        with agent.tree_state_lock:
            reconcile_runtime_locks(journal.nodes, search_state.actions())
            parent = agent.select_parent(None)
            if parent is None:
                return None
            journal_ids = {node.id for node in journal.nodes}
            active_parent_ids = {
                action.parent_node_id for action in search_state.actions()
            }
            if (
                parent.id not in journal_ids
                or bool(getattr(parent, "pending_execution", False))
                or parent.id in active_parent_ids
            ):
                return None
            root_draft = agent.is_root(parent) and not parent.reached_child_limit(
                scfg=agent.scfg
            )
            action = register_search_action(
                parent,
                parent_from_topk=bool(getattr(parent, "_topk_triggered", False)),
            )
            _reconcile_parent_runtime_state(parent.id)
            reservation = ExpansionReservation(parent, agent.tree_state_lock)
            profile = action.expansion_profile
            ordinal = int(getattr(profile, "sibling_ordinal", 0) or 0)
            draft_single_call = bool(
                root_draft
                and (
                    (ordinal == 1 and getattr(draft_cfg, "fast_first_draft", True))
                    or (
                        ordinal > 1
                        and not getattr(draft_cfg, "use_stepwise_after_first", True)
                    )
                )
            )
            fast_draft_mode = bool(root_draft and ordinal == 1 and draft_single_call)
            placeholder = _active_action_placeholder(action, parent)
            placeholder_id = str(placeholder["id"])
            pending_draft_nodes.append(placeholder)
            pending_status_by_id[placeholder_id] = "generating"
            pending_action_id_by_node_id[placeholder_id] = action.action_id
            refresh_pending_nodes_state("worker_selected")
        future = executor.submit(
            step_task,
            action.action_id,
            parent,
            reuse_reserved_slot=True,
            return_result_node=True,
            draft_single_call=draft_single_call,
            fast_draft_mode=fast_draft_mode,
            expansion_reservation=reservation,
        )
        return future, action, reservation, placeholder_id

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    draft_cfg = getattr(cfg.agent, "draft", None)
    time_limit_secs = int(getattr(cfg.agent, "time_limit", 0) or 0)
    remaining_time_secs = max(
        0.0,
        time_limit_secs - search_state.elapsed_seconds(),
    ) if time_limit_secs > 0 else None
    run_deadline: Optional[float] = (
        time.time() + remaining_time_secs if remaining_time_secs is not None else None
    )
    timed_out = False

    logger.info(f"ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"Initial forced root draft count: {initial_draft_count}")
    logger.info(
        "Worker-local draft generation: fast_first=%s, stepwise_after_first=%s",
        bool(getattr(draft_cfg, "fast_first_draft", True)),
        bool(getattr(draft_cfg, "use_stepwise_after_first", True)),
    )
    if run_deadline is not None:
        logger.info(
            "Hard timeout enabled: total=%ss, restored_elapsed=%.1fs, remaining=%.1fs",
            time_limit_secs,
            search_state.elapsed_seconds(),
            remaining_time_secs,
        )

    def is_timed_out() -> bool:
        return run_deadline is not None and time.time() >= run_deadline

    lock = threading.Lock()
    logger.info(f"Resume progress: completed={completed}/{total_steps} from journal nodes={len(journal)}")

    pending_draft_nodes = []
    pending_status_by_id: dict[str, str] = {}
    pending_action_id_by_node_id: dict[str, str] = {}

    for action_id, node in restored_generated_nodes.items():
        pending_draft_nodes.append(node)
        pending_status_by_id[node.id] = "pending_execution"
        pending_action_id_by_node_id[node.id] = action_id
    _write_pending_nodes_state(cfg, pending_draft_nodes, pending_status_by_id, "initialized")

    def refresh_pending_nodes_state(phase: str) -> None:
        try:
            _write_pending_nodes_state(cfg, pending_draft_nodes, pending_status_by_id, phase)
        except Exception as exc:
            logger.warning(f"Failed to write {PENDING_NODES_FILE}: {exc}")

    logger.info(
        "Search pipeline: parallel closed-loop workers with virtual-visit UCT reservations"
    )
    if pending_draft_nodes or completed < total_steps:
        drafts_to_execute = [
            node for node in pending_draft_nodes
            if pending_status_by_id.get(str(_node_attr(node, "id", ""))) == "pending_execution"
        ]
        logger.info("Parallel MCTS: closed-loop worker execution")
        logger.info(f"  - Restored generated nodes: {len(drafts_to_execute)}")
        logger.info(f"  - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node, action_id: str, reservation):
            try:
                executed_node = agent.execute_deferred_node(
                    node,
                    exec_callback,
                    runtime_action_id=action_id,
                    expansion_reservation=reservation,
                )
                logger.info(f"Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"Exception during draft node {node.id} execution: {e}")
                return None

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        fast_shutdown = False
        try:
            futures = set()
            future_metadata: dict = {}
            submitted_actions = 0
            action_submission_budget = max(0, total_steps - completed)
            for node in drafts_to_execute:
                if submitted_actions >= action_submission_budget:
                    break
                if is_timed_out():
                    timed_out = True
                    logger.warning("Time limit reached before submitting pending draft executions.")
                    break
                pending_status_by_id[node.id] = "executing"
                refresh_pending_nodes_state("phase2_execution")
                action_id = pending_action_id_by_node_id[node.id]
                reservation = ExpansionReservation(node.parent, agent.tree_state_lock)
                fut = executor.submit(
                    execute_draft_node,
                    node,
                    action_id,
                    reservation,
                )
                futures.add(fut)
                future_metadata[fut] = {
                    "action_id": action_id,
                    "pending_node_id": node.id,
                    "reservation": reservation,
                }
                submitted_actions += 1
                logger.info(f"Resumed generated node in a closed-loop worker: {node.id}")

            def fill_worker_slots() -> int:
                added = 0
                while (
                    len(futures) < max_workers
                    and completed + len(futures) < total_steps
                    and not timed_out
                ):
                    scheduled = schedule_new_step(executor)
                    if scheduled is None:
                        break
                    future, action, reservation, placeholder_id = scheduled
                    futures.add(future)
                    future_metadata[future] = {
                        "action_id": action.action_id,
                        "pending_node_id": placeholder_id,
                        "reservation": reservation,
                    }
                    added += 1
                    logger.info(
                        "Assigned worker to parent %s after virtual-visit UCT update",
                        action.parent_node_id,
                    )
                return added

            fill_worker_slots()

            while futures:
                if is_timed_out():
                    timed_out = True
                    logger.warning("Time limit reached in Phase 2 main loop.")
                    break

                done, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                    timeout=max(0.05, float(getattr(runtime_cfg, "scheduler_poll_seconds", 1.0))),
                )

                if not done:
                    search_state.checkpoint_if_due()
                    continue

                for fut in done:
                    futures.remove(fut)
                    metadata = future_metadata.pop(fut, {})
                    action_id = metadata.get("action_id")
                    pending_node_id = metadata.get("pending_node_id")
                    reservation = metadata.get("reservation")
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(
                                f"Task completed: node_id={cur_node.id}, step={cur_node.step}, "
                                f"is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}"
                            )
                        else:
                            logger.warning("Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"Exception during task execution: {e}")
                        cur_node = None
                    if cur_node is None and isinstance(reservation, ExpansionReservation):
                        reservation.settle_failure(-1.0)

                    with lock:
                        completed = len(journal) - 1
                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                        if pending_node_id and cur_node:
                            pending_draft_nodes[:] = [
                                node
                                for node in pending_draft_nodes
                                if str(_node_attr(node, "id", "")) != pending_node_id
                            ]
                            pending_status_by_id.pop(pending_node_id, None)
                            pending_action_id_by_node_id.pop(pending_node_id, None)
                            refresh_pending_nodes_state("phase2_execution")
                        elif pending_node_id:
                            pending_status_by_id[pending_node_id] = "failed"
                            refresh_pending_nodes_state("phase2_execution")
                        if action_id:
                            finish_search_action(action_id)
                        with agent.journal_lock:
                            save_run(cfg, journal)

                    if is_timed_out():
                        timed_out = True
                        logger.warning("Time limit reached before scheduling next task.")
                    else:
                        fill_worker_slots()
                    logger.info(f"Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")

            if timed_out:
                logger.warning(
                    f"Time limit reached (configured={time_limit_secs}s). "
                    "Search budget is exhausted; saving current best artifacts and ending AutoML normally."
                )
                with agent.journal_lock:
                    agent.accept_search_results = False
                    agent.runtime_checkpoint_callback = None
                    for metadata in future_metadata.values():
                        reservation = metadata.get("reservation")
                        if isinstance(reservation, ExpansionReservation):
                            reservation.cancel()
                interpreter.terminate_all_subprocesses()
                fast_shutdown = True
                with lock:
                    for node_id in list(pending_status_by_id):
                        pending_status_by_id[node_id] = "cancelled"
                    refresh_pending_nodes_state("timed_out")
                    with agent.journal_lock:
                        repair_journal_for_resume(journal, [])
                        save_run(cfg, journal)
            elif completed < total_steps and not futures:
                logger.warning(
                    f"Phase 2 exited with no active futures before reaching target steps: {completed}/{total_steps}"
                )
        except KeyboardInterrupt:
            interrupted = True
            interruption_requested.set()
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            for node_id in list(pending_status_by_id):
                pending_status_by_id[node_id] = "cancelled"
            with agent.journal_lock:
                agent.accept_search_results = False
                agent.runtime_checkpoint_callback = None
                for metadata in future_metadata.values():
                    reservation = metadata.get("reservation")
                    if isinstance(reservation, ExpansionReservation):
                        reservation.cancel()
            refresh_pending_nodes_state("interrupted")
            interpreter.terminate_all_subprocesses()
            checkpoint_ok = commit_interrupted_checkpoint()
            if checkpoint_ok and bool(
                getattr(runtime_cfg, "exit_immediately_after_interrupt_checkpoint", True)
            ):
                logger.warning(
                    "Interrupted checkpoint is durable; terminating the AlgoEvolve process immediately."
                )
                _exit_process_immediately(130)
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
            raise
        finally:
            if fast_shutdown:
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=False)
            elif not interrupted:
                executor.shutdown(wait=True)
    else:
        logger.info("Search target was already complete; no worker expansion was scheduled")

    if timed_out and bool(getattr(runtime_cfg, "force_process_exit_on_timeout", True)):
        logger.warning(f"AlgoEvolve search budget exhausted: {time_limit_secs}s")

    pending_draft_nodes.clear()
    pending_status_by_id.clear()
    pending_action_id_by_node_id.clear()
    refresh_pending_nodes_state("complete")
    with agent.journal_lock:
        if timed_out:
            # Timed-out workers are intentionally abandoned. Remove their
            # transient locks and unjournaled child links before committing a
            # completed checkpoint that may later be resumed.
            repair_journal_for_resume(journal, [])
        with agent.tree_state_lock:
            refresh_persisted_uct_values(agent)
            save_run(cfg, journal)

    termination_reason = "time_limit_exhausted" if timed_out else "steps_completed"
    _write_run_status(
        cfg,
        status="completed",
        termination_reason=termination_reason,
        completed_steps=max(0, len(journal) - 1),
        total_steps=total_steps,
        time_limit_secs=time_limit_secs,
    )
    # Clear cancelled/completed runtime actions before the completion manifest
    # is written. The manifest is the final durable marker and must never point
    # at an older search_state.json that still contains in-flight work.
    search_state.close(clear_actions=True)
    persist_resumable_checkpoint(
        agent,
        status="completed",
        reason=termination_reason,
        active_actions=[],
        manifest_filename=str(
            getattr(
                runtime_cfg,
                "checkpoint_manifest_filename",
                "checkpoint_manifest.json",
            )
        ),
    )
    normal_exit["done"] = True
    interpreter.cleanup_session(-1)
    if timed_out and bool(getattr(runtime_cfg, "force_process_exit_on_timeout", True)):
        # ThreadPoolExecutor cannot cancel already-running LLM calls. Once the
        # search budget is exhausted and artifacts are saved, exit the process
        # immediately so the outer service can continue to AutoReport instead
        # of waiting for stale worker threads until the grace timeout.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        logging.shutdown()
        os._exit(0)


if __name__ == "__main__":
    run()
