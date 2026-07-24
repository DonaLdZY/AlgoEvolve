"""AgentSearch: tree search coordinator; delegates to node_selection, evaluation, execution, solution_manager."""

import logging
import random
import time
from typing import Callable, List, Dict, Optional

from engine.executor import ExecutionResult
from engine.search_node import SearchNode, Journal
from engine.expansion_profile import ExpansionProfile
import utils.data_preview as data_preview
from config import Config
from utils.metric import WorstMetricValue
import threading
import json

from agents import (
    draft_agent, improve_agent, debug_agent,
    evolution_agent, fusion_agent, aggregation_agent,
    code_review_agent,
    result_parse_agent,
)
from engine import node_selection, evaluation, execution, solution_manager
from engine.conditions import is_branch_stagnant, should_trigger_node_fusion
from utils.data_preview import clean_task_desc
from utils.autorealize_context import (
    build_autorealize_context_md,
    has_autorealize_context,
    load_autorealize_description_md,
    submission_required_from_context,
)
from agents.prompt_policy import (
    apply_expansion_profile,
    ensure_expansion_profile,
    infer_task_family,
)
from engine.solution_protocol import interface_for, preflight_code

logger = logging.getLogger("MLEvolve")


ExecCallbackType = Callable[[str, bool], ExecutionResult]
RuntimeCheckpointType = Callable[[str, str, SearchNode | None], None]

class AgentSearch:
    def __init__(
            self,
            task_desc: str,
            cfg: Config,
            journal: Journal,
    ):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        input_dir = cfg.workspace_dir / "input"
        self.has_autorealize_package = has_autorealize_context(input_dir)
        self.autorealize_context = build_autorealize_context_md(input_dir, write_context_file=True)
        if self.has_autorealize_package and not str(self.autorealize_context or "").strip():
            raise RuntimeError(
                "AutoRealize package detected, but no prompt-ready automl_context.md was found. "
                "AutoRealize must generate the complete MLEvolve context before AutoML starts."
            )
        self.has_autorealize_context = bool(str(self.autorealize_context or "").strip())
        if self.has_autorealize_context:
            self.task_desc = load_autorealize_description_md(input_dir, fallback=task_desc)
            logger.info(
                "[AgentSearch] AutoRealize context detected; using description.md as task_desc "
                "and automl_context.md as the replacement data context."
            )
        else:
            self.task_desc = clean_task_desc(task_desc, cfg)
        context_submission_required = submission_required_from_context(input_dir)
        if context_submission_required is False and getattr(self.acfg, "generate_submission", True):
            self.acfg.generate_submission = False
            logger.info("[AgentSearch] AutoRealize context disables mandatory submission.csv generation.")
        elif context_submission_required is True and not getattr(self.acfg, "generate_submission", True):
            logger.info("[AgentSearch] AutoRealize context indicates a submission contract, but config keeps generation disabled.")
        self.journal = journal
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: SearchNode | None = None
        self.all_root = True
        self.current_node_list = []
        self.best_metric: float = None
        self.best_node: SearchNode = None
        self.provisional_best_node: SearchNode = None
        self.search_start_time = None
        self.runtime_checkpoint_callback: RuntimeCheckpointType | None = None
        self.fast_draft_mode = False
        self.accept_search_results = True
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.tree_state_lock = threading.RLock()
        self.start_time = time.time()
        self.use_stepwise_generation = True

        self.next_branch_id = 1
        self.branch_all_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_successful_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_node_count: Dict[int, int] = {}
        self.use_coldstart = cfg.coldstart.use_coldstart
        self.coldstart_description = cfg.coldstart.description

        # Top-N candidates
        self.top_k = self.scfg.top_candidates_size
        self.top_candidates: List[SearchNode] = []

        # Performance stagnation detection
        self.best_metric_history = []
        self.stagnation_threshold = self.scfg.stagnation_window
        self.post_process_triggered = False
        self.post_process_attempts = 0
        self.max_post_process_attempts = 4
        self.improve_attempts_count = 0
        self.last_successful_improve_step = 0

        self.fusion_draft_count = 0
        self.max_fusion_drafts = cfg.agent.max_fusion_drafts

        self.metric_maximize: bool | None = None
        self.metric_maximize_reasoning: str | None = None
        result_parse_agent.determine_metric_direction(self)
        self.virtual_root = self._init_or_restore_journal_state()

        # Global memory
        self.global_memory = None
        if self.acfg.use_global_memory:
            try:
                from agents.memory.global_memory import GlobalMemoryLayer
                memory_dir = str(self.cfg.workspace_dir / "global_memory")
                self.global_memory = GlobalMemoryLayer(
                    memory_dir=memory_dir,
                    embedding_backend=getattr(self.acfg, "memory_embedding_backend", "local"),
                    embedding_api_key=getattr(self.acfg, "memory_embedding_api_key", ""),
                    embedding_base_url=getattr(self.acfg, "memory_embedding_base_url", ""),
                    embedding_model=getattr(self.acfg, "memory_embedding_model", ""),
                    embedding_model_path=self.acfg.memory_embedding_model_path,
                    embedding_device=self.acfg.memory_embedding_device,
                    similarity_threshold=self.acfg.memory_similarity_threshold,
                )
                logger.info(f"[AgentSearch] Global memory enabled and initialized at {memory_dir}")
            except Exception as e:
                import traceback
                logger.warning(f"[AgentSearch] Failed to initialize global memory: {e}")
                logger.debug(f"[AgentSearch] Global memory initialization traceback: {traceback.format_exc()}")
                self.global_memory = None
        else:
            logger.info("[AgentSearch] Global memory is disabled by config")

    def _init_or_restore_journal_state(self) -> SearchNode:
        """Initialize a new root or rebuild in-memory search state from a loaded journal."""
        existing_roots = [
            n for n in self.journal.nodes
            if getattr(n, "stage", None) == "root" and getattr(n, "parent", None) is None
        ]
        if not existing_roots:
            root = SearchNode(parent=None, plan="(root)", code="", metric=WorstMetricValue(), stage="root")
            self.journal.append(root)
            logger.info("[AgentSearch] Initialized a fresh search journal.")
            return root

        root = existing_roots[0]
        restored_nodes = [n for n in self.journal.nodes if n is not root]
        valid_nodes: list[SearchNode] = []

        for node in restored_nodes:
            if getattr(node, "stage", None) == "fusion_draft":
                self.fusion_draft_count += 1
            branch_id = getattr(node, "branch_id", None)
            if branch_id is not None:
                self.branch_all_nodes.setdefault(branch_id, []).append(node)
                self.branch_node_count[branch_id] = self.branch_node_count.get(branch_id, 0) + 1
                try:
                    self.next_branch_id = max(self.next_branch_id, int(branch_id) + 1)
                except Exception:
                    pass

            metric = getattr(node, "metric", None)
            has_metric = metric is not None and getattr(metric, "value", None) is not None
            if getattr(node, "search_eligible", False) and has_metric:
                self.current_node_list.append(node)
                if branch_id is not None:
                    self.branch_successful_nodes.setdefault(branch_id, []).append(node)
            if getattr(node, "search_eligible", False) and has_metric:
                valid_nodes.append(node)

        maximize = True if self.metric_maximize is None else self.metric_maximize
        valid_nodes.sort(
            key=lambda n: n.metric.value if (n.metric and n.metric.value is not None) else (float("-inf") if maximize else float("inf")),
            reverse=maximize,
        )
        self.top_candidates = valid_nodes[: self.top_k]
        if valid_nodes:
            self.best_node = valid_nodes[0]
            self.best_metric = self.best_node.metric.value
        self.provisional_best_node = self.journal.get_best_search_node()

        self.current_step = max(0, len(self.journal) - 1)
        logger.info(
            "[AgentSearch] Restored search journal: "
            f"nodes={len(self.journal)}, completed={self.current_step}, "
            f"branches={len(self.branch_all_nodes)}, best={self.best_metric}"
        )
        return root

    def _serialize_prompt(self, prompt_complete) -> str | None:
        """Serialize prompt (str or dict) to string for saving in node."""
        if prompt_complete is None:
            return None
        if isinstance(prompt_complete, str):
            return prompt_complete
        elif isinstance(prompt_complete, dict):
            return json.dumps(prompt_complete, ensure_ascii=False, indent=2)
        else:
            return str(prompt_complete)

    def update_data_preview(self):
        if getattr(self, "has_autorealize_context", False):
            self.data_preview = getattr(self, "autorealize_context", "") or ""
            logger.info(
                "[AgentSearch] Using AutoRealize automl_context as the sole data-preview context; "
                "standalone MLEvolve preview generation is disabled for provider cache efficiency."
            )
            return
        generate_submission = getattr(self.acfg, "generate_submission", True)
        base_preview = data_preview.generate(
            self.cfg.workspace_dir,
            submission_required=generate_submission,
        )
        if not generate_submission:
            self.data_preview = self._with_autorealize_context(base_preview)
            return
        submission_format_warning = """

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        """
        self.data_preview = self._with_autorealize_context(base_preview + submission_format_warning)

    def _with_autorealize_context(self, preview: str) -> str:
        context = getattr(self, "autorealize_context", "") or ""
        if not context:
            return preview
        return preview

    def is_root(self, node: SearchNode):
        return bool(node and node.id == self.virtual_root.id)

    def restore_search_elapsed(self, elapsed_seconds: float) -> None:
        self.search_start_time = time.time() - max(0.0, float(elapsed_seconds))

    def select_parent(self, node: SearchNode | None = None) -> SearchNode | None:
        if not node or node.stage == "root":
            return node_selection.select_with_soft_switch(self)
        return node

    def select_existing_parent(self) -> SearchNode | None:
        """Select work below an evaluated root draft during initial coverage."""

        return node_selection.select_existing_branch(self)

    def _checkpoint_runtime_action(
        self,
        action_id: str | None,
        status: str,
        node: SearchNode | None,
    ) -> None:
        if not action_id or self.runtime_checkpoint_callback is None:
            return
        try:
            self.runtime_checkpoint_callback(action_id, status, node)
        except Exception as exc:
            logger.warning("Failed to checkpoint search action %s at %s: %s", action_id, status, exc)

    def _run_single_step(
        self,
        parent_node: SearchNode,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
        runtime_action_id: str | None = None,
        expansion_profile: ExpansionProfile | None = None,
        draft_single_call: bool = False,
        fast_draft_mode: bool = False,
        expansion_reservation=None,
    ):
        """Run one search step: select action (draft/debug/improve), execute, parse, validate."""
        result_node = None
        _root = False

        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    if parent_node.reached_child_limit(scfg=self.scfg):
                        logger.info("🎯 Regular draft limit reached, triggering multi-branch aggregation (conditions already checked in select())")
                        profile = ensure_expansion_profile(
                            self, parent_node, expansion_profile, "fusion_draft"
                        )
                        result_node = aggregation_agent.run(
                            self,
                            mode="node",
                            parent_node=parent_node,
                            expansion_profile=profile,
                        )
                        if result_node:
                            result_node.lock = True
                            logger.info(f"[_run_single_step] Aggregation branch node {result_node.id} is locked.")
                        else:
                            logger.info("Aggregation failed or limit reached, skipping. Will continue normal search.")
                            result_node = None
                    else:
                        profile = ensure_expansion_profile(
                            self, parent_node, expansion_profile, "draft"
                        )
                        result_node = draft_agent.run(
                            self,
                            init_solution_path=init_solution_path,
                            expansion_profile=profile,
                            fast_draft_mode=fast_draft_mode,
                            use_stepwise_generation=not draft_single_call,
                        )
                        result_node.lock = True
                        logger.info(f"[_run_single_step] Draft node {result_node.id} is locked.")
                elif parent_node.is_buggy or parent_node.is_valid is False:
                    profile = ensure_expansion_profile(
                        self, parent_node, expansion_profile, "debug"
                    )
                    result_node = debug_agent.run(self, parent_node, expansion_profile=profile)

                elif parent_node.is_buggy is False:
                    can_use_fusion = should_trigger_node_fusion(self, parent_node)
                    is_from_topk = getattr(parent_node, '_topk_triggered', False)
                    stagnation_threshold = self.scfg.topk_stagnation_threshold if is_from_topk else self.scfg.branch_stagnation_threshold
                    if is_from_topk:
                        logger.info(f"🎯 Exploitation mode: using relaxed stagnation threshold ({stagnation_threshold} attempts)")

                    if is_branch_stagnant(self, parent_node.branch_id, threshold=stagnation_threshold):
                        if can_use_fusion:
                            if random.random() < self.acfg.fusion_vs_evolution_prob:
                                logger.info(f"🎯 Triggering evidence-gated fusion for stagnant node {parent_node.id}")
                                profile = ensure_expansion_profile(
                                    self, parent_node, expansion_profile, "fusion"
                                )
                                result_node = fusion_agent.run(
                                    self, parent_node, expansion_profile=profile
                                )
                            else:
                                logger.info(f"🎯 Triggering intra-branch evolution for stagnant node {parent_node.id}")
                                profile = ensure_expansion_profile(
                                    self, parent_node, expansion_profile, "evolution"
                                )
                                result_node = evolution_agent.run(
                                    self, parent_node, expansion_profile=profile
                                )
                        else:
                            logger.info(f"🔄 Using evolution because Fusion evidence/time gates are not met for node {parent_node.id}")
                            profile = ensure_expansion_profile(
                                self, parent_node, expansion_profile, "evolution"
                            )
                            result_node = evolution_agent.run(
                                self, parent_node, expansion_profile=profile
                            )
                    else:
                        logger.info(f"🔄 Using normal improve for node {parent_node.id}")
                        profile = ensure_expansion_profile(
                            self, parent_node, expansion_profile, "improve"
                        )
                        result_node = improve_agent.run(
                            self, parent_node, expansion_profile=profile
                        )

                else:
                    logger.warning(f"[_run_single_step] node {parent_node.id} is_buggy is None.")

                if result_node:
                    apply_expansion_profile(
                        self,
                        result_node,
                        profile,
                        str(getattr(result_node, "stage", profile.operator)),
                    )
                    interface = interface_for(
                        task_family=infer_task_family(self),
                        method_family=result_node.method_family,
                    )
                    initial_preflight = preflight_code(result_node.code, interface)
                    if init_solution_path:
                        logger.info(f"Node {result_node.id} from init_solution, skipping code review")
                    elif fast_draft_mode and initial_preflight.ok and bool(
                        getattr(self.acfg.draft, "fast_first_draft_skip_pre_review", True)
                    ):
                        result_node.fast_draft = True
                        logger.info(
                            "Fast draft node %s skips pre-execution LLM review; "
                            "runtime and result validation remain mandatory.",
                            result_node.id,
                        )
                    else:
                        reviewed_code = code_review_agent.run(self, result_node)
                        if reviewed_code.strip() != result_node.code.strip():
                            logger.info(f"Node {result_node.id} code has been reviewed and modified")
                            result_node.code = reviewed_code
                        else:
                            logger.info(f"Node {result_node.id} passed code review without changes")

                    final_preflight = preflight_code(result_node.code, interface)
                    if not final_preflight.ok:
                        result_node.preflight_report = final_preflight.__dict__
                        logger.warning(
                            "Node %s still fails deterministic preflight; requesting one focused repair",
                            result_node.id,
                        )
                        repaired_code = code_review_agent.run(self, result_node)
                        if repaired_code.strip() != result_node.code.strip():
                            result_node.code = repaired_code
                        final_preflight = preflight_code(result_node.code, interface)
                    max_regenerations = max(
                        0,
                        int(
                            getattr(
                                getattr(self.acfg, "retries", None),
                                "preflight_regeneration_max_attempts",
                                2,
                            )
                            or 0
                        ),
                    )
                    for regeneration_attempt in range(1, max_regenerations + 1):
                        if final_preflight.ok:
                            break
                        result_node.preflight_report = final_preflight.__dict__
                        logger.warning(
                            "Node %s still fails deterministic preflight after patch review; "
                            "regenerating the same node with rejection evidence (%s/%s)",
                            result_node.id,
                            regeneration_attempt,
                            max_regenerations,
                        )
                        try:
                            regenerated_plan, regenerated_code = (
                                code_review_agent.regenerate_after_preflight_failure(
                                    self,
                                    result_node,
                                    interface,
                                    final_preflight,
                                    attempt=regeneration_attempt,
                                )
                            )
                        except Exception as exc:
                            logger.exception(
                                "Preflight regeneration %s/%s failed for node %s: %s",
                                regeneration_attempt,
                                max_regenerations,
                                result_node.id,
                                exc,
                            )
                            continue
                        result_node.plan = regenerated_plan or result_node.plan
                        result_node.code = regenerated_code
                        final_preflight = preflight_code(result_node.code, interface)
                        logger.info(
                            "Preflight regeneration %s/%s for node %s: ok=%s issues=%s",
                            regeneration_attempt,
                            max_regenerations,
                            result_node.id,
                            final_preflight.ok,
                            final_preflight,
                        )
                    result_node.solution_interface = interface.kind
                    result_node.preflight_report = final_preflight.__dict__
                    if not final_preflight.ok:
                        raise RuntimeError(
                            "Generated code failed mandatory pre-execution interface/safety preflight after review: "
                            f"{final_preflight}"
                        )

                    self._checkpoint_runtime_action(runtime_action_id, "generated", result_node)

                    if not execute_immediately:
                        logger.info(f"Node {result_node.id} code generated and reviewed, execution deferred")
                        result_node.pending_execution = True
                        result_node._expansion_reservation = expansion_reservation
                        return _root, result_node
                    self._checkpoint_runtime_action(runtime_action_id, "executing", result_node)
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                    result_node = result_parse_agent.run(self,
                        node=result_node,
                        exec_result=exe_res
                    )
                    execution.validate_executed_node(self, result_node)
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

                    with self.journal_lock:
                        if not getattr(self, "accept_search_results", True):
                            logger.info(
                                "Discarding late node %s after interruption checkpoint began.",
                                result_node.id,
                            )
                            return True, None
                        if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                            raise ValueError(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                        else:
                            self.journal.append(result_node)
                            if parent_node.is_buggy and result_node.is_buggy is False:
                                parent_node.is_debug_success = True
                            result_node._expansion_reservation = expansion_reservation
                            _root = evaluation.check_improvement(
                                self, result_node, parent_node
                            )

            except Exception as e:
                logger.warning(f"Step failed for parent {parent_node.id}, rolling back expected child count and propagating zero reward.")
                evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
                raise e

        else:
            evaluation.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node

    def step(
        self,
        node: SearchNode | None,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
        runtime_action_id: str | None = None,
        parent_preselected: bool = False,
        return_result_node: bool = False,
        expansion_profile: ExpansionProfile | None = None,
        draft_single_call: bool = False,
        fast_draft_mode: bool = False,
        expansion_reservation=None,
    ) -> SearchNode:
        if self.data_preview is None:
            self.update_data_preview()
        if self.search_start_time is None:
            self.search_start_time = time.time()

        if not parent_preselected:
            node = self.select_parent(node)
        elif node is None:
            raise ValueError("A preselected search parent is required.")

        _root, result_node = self._run_single_step(
            node,
            exec_callback=exec_callback,
            execute_immediately=execute_immediately,
            init_solution_path=init_solution_path,
            runtime_action_id=runtime_action_id,
            expansion_profile=expansion_profile,
            draft_single_call=draft_single_call,
            fast_draft_mode=fast_draft_mode,
            expansion_reservation=expansion_reservation,
        )

        if result_node:
            metric_value = result_node.metric.value if result_node.metric else None
            best_metric = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
            logger.info(f"[step] {node.id} → {result_node.id}: metric={metric_value}, best={best_metric}")

        if (
            getattr(self, "accept_search_results", True)
            and result_node
            and result_node.metric
            and result_node.metric.value is not None
        ):
            solution_manager.update_best_solution(self, result_node)

        self.current_step = len(self.journal)

        # Cumulative stats
        total_nodes = len(self.journal)
        n_branches = len(self.branch_all_nodes)
        best_val = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
        logger.info(f"[stats] step={self.current_step}, nodes={total_nodes}, branches={n_branches}, best={best_val}")

        if return_result_node and result_node is not None:
            return result_node
        if _root or result_node is None:
            return self.virtual_root
        else:
            return result_node

    def execute_deferred_node(
        self,
        node: SearchNode,
        exec_callback: ExecCallbackType,
        runtime_action_id: str | None = None,
        expansion_reservation=None,
    ) -> SearchNode:
        """Backward-compatible closed loop for a deferred node."""

        exec_result = self.execute_deferred_code(
            node,
            exec_callback,
            runtime_action_id=runtime_action_id,
        )
        return self.finalize_deferred_node(
            node,
            exec_result,
            expansion_reservation=expansion_reservation,
        )

    def execute_deferred_code(
        self,
        node: SearchNode,
        exec_callback: ExecCallbackType,
        runtime_action_id: str | None = None,
    ):
        """Run generated code without occupying the result-review worker."""

        if not hasattr(node, 'pending_execution') or not node.pending_execution:
            logger.warning(f"Node {node.id} is not marked for deferred execution")
            raise ValueError(f"Node {node.id} is not pending execution")

        logger.info(f"Executing deferred node {node.id}")
        self._checkpoint_runtime_action(runtime_action_id, "executing", node)
        return exec_callback(node.code, node.id, True)

    def finalize_deferred_node(
        self,
        node: SearchNode,
        exec_result,
        expansion_reservation=None,
    ) -> SearchNode:
        """Review an execution result, validate it, and commit the node."""

        parent_node = node.parent
        try:
            node = result_parse_agent.run(
                self,
                node=node,
                exec_result=exec_result,
            )

            execution.validate_executed_node(self, node)

            logger.info(f"Node {node.id} execution completed: metric={node.metric.value}, is_buggy={node.is_buggy}")

            node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            with self.journal_lock:
                if not getattr(self, "accept_search_results", True):
                    logger.info(
                        "Discarding late deferred node %s after interruption checkpoint began.",
                        node.id,
                    )
                    return node
                if self.best_node and node.metric.maximize and self.best_node.metric.maximize != node.metric.maximize:
                    logger.warning("New node's metric is inconsistent with metrics in the journal")
                    raise ValueError("New node's metric is inconsistent with metrics in the journal")
                else:
                    self.journal.append(node)
                    if parent_node and parent_node.is_buggy and node.is_buggy is False:
                        parent_node.is_debug_success = True
                    node._expansion_reservation = (
                        expansion_reservation
                        or getattr(node, "_expansion_reservation", None)
                    )
                    _root = evaluation.check_improvement(self, node, parent_node)
                    logger.info(f"Node {node.id} added to journal")

            node.pending_execution = False
            if getattr(self, "accept_search_results", True):
                solution_manager.update_best_solution(self, node)

            return node

        except Exception as e:
            logger.exception(f"Exception during deferred node result review: {e}")
            evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
            parent_node.sub_expected_child_count()
            raise e
