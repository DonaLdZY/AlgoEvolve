"""configuration and setup utils"""

import os
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Hashable, cast
import datetime
import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

# Lazy import to avoid circular dependencies while configuration is imported.
def _get_journal_classes():
    from engine.journal_snapshot import JournalSnapshot
    from engine.search_node import Journal, get_longest_path, get_path_to_node

    return Journal, JournalSnapshot, get_longest_path, get_path_to_node

from utils import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("MLEvolve")


"""These dataclasses provide typing for the default config/config.yaml."""


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    minimum_output_tokens: int = 32768
    max_tokens: int | None = 32768
    request_timeout_seconds: float = 1200.0
    network_retry_max_attempts: int = 5
    network_retry_base_sleep_seconds: float = 5.0
    network_retry_max_sleep_seconds: float = 30.0
    generation_max_retries: int = 5
    generation_retry_delay_seconds: float = 3.0
    continuation_max_rounds: int = 2
    continuation_overlap_scan_chars: int = 4096
    # Used only for local headroom accounting; zero means unknown/provider-managed.
    context_window_tokens: int = 0


@dataclass
class DraftConfig:
    """Initial draft generation and visibility controls."""

    fast_first_draft: bool = True
    fast_first_draft_skip_pre_review: bool = True
    # Deprecated: first-result sequencing replaces synchronous repair blocking.
    fast_first_draft_max_repairs: int = 2
    # Deprecated: full AutoRealize context is always used.
    fast_first_draft_compact_context: bool = False
    use_stepwise_after_first: bool = True
    # Deprecated compatibility field. agent.initial_drafts is authoritative.
    optimization_initial_drafts_cap: int = 0
    show_pending_draft_nodes: bool = True
    # Deprecated: execution is now scheduled by a priority queue.
    submission_stagger_seconds: float = 10.0
    # Deprecated: full AutoRealize context is always used.
    stepwise_stage_context: bool = False
    stepwise_accumulate_context: bool = True
    stepwise_context_max_tokens: int = 90000
    stepwise_compaction_keep_recent_steps: int = 2
    stepwise_compaction_max_tokens: int = 32768
    stepwise_context_headroom_ratio: float = 0.15
    stepwise_snapshot_dirname: str = "context_snapshots"


@dataclass
class RetryConfig:
    """Agent-level structured-output and review retry policy."""

    code_review_max_attempts: int = 2
    preflight_regeneration_max_attempts: int = 2
    code_review_delay_seconds: float = 5.0
    code_review_model_role: str = "feedback"
    code_review_escalate_to_code: bool = True
    code_generation_extract_max_attempts: int = 2
    metric_direction_max_attempts: int = 3
    metric_direction_delay_seconds: float = 1.0
    result_parse_max_attempts: int = 3
    refine_plan_max_attempts: int = 3
    result_adjudicator_on_anomaly: bool = True

@dataclass
class DecayConfig:
    exploration_constant: float
    lower_bound: float
    alpha: float
    phase_ratios: list
    

@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    topk_max_improves: int
    max_improve_failure: int
    parallel_search_num: int
    branch_stagnation_threshold: int
    topk_stagnation_threshold: int
    top_candidates_size: int
    stagnation_window: int
    num_gpus: int
    explore_switch_start: float
    explore_switch_end: float
    min_exploration_weight: float
    topk_early_k: int
    topk_early_max_per_branch: int
    topk_late_k: int
    topk_late_max_per_branch: int
    force_backprop_late_threshold: float
    force_backprop_late_prob: float
    force_backprop_mid_threshold: float
    force_backprop_mid_modulo: int
    recent_best_window: int
    fusion_min_time_hours: float
    fusion_max_time_hours: float
    fusion_min_successful_nodes: int
    fusion_min_branches: int
    root_new_draft_probability: float = 0.25
    fusion_min_remaining_seconds: int = 300

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    initial_drafts: int
    seed: int
    data_preview: bool
    generate_submission: bool
    code: StageConfig
    feedback: StageConfig
    check_data_leakage: bool
    fusion_vs_evolution_prob: float
    branch_fusion_trigger_prob: float
    max_fusion_drafts: int
    use_global_memory: bool
    memory_similarity_threshold: float
    memory_embedding_backend: str
    memory_embedding_api_key: str
    memory_embedding_base_url: str
    memory_embedding_model: str
    memory_embedding_device: str
    memory_embedding_model_path: str
    search: SearchConfig
    decay: DecayConfig
    use_diff_mode: bool = True
    draft: DraftConfig = field(default_factory=DraftConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    use_optimization_experience_library: bool = True
    optimization_experience_library_path: str = ""
    optimization_experience_max_cards: int = 2
    optimization_experience_min_score: float = 3.0
    optimization_experience_max_chars: int = 6000
    output_language: str = "english"
@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    auto_install_missing_dependencies: bool = True
    dependency_install_policy: str = "ai_declared"
    dependency_install_target_path: str = ""
    dependency_install_timeout_seconds: int = 600
    dependency_install_lock_timeout_seconds: int = 900
    dependency_install_max_packages_per_execution: int = 3
    dependency_install_output_tail_chars: int = 8000
    dependency_install_allowlist: list[str] = field(default_factory=list)
    dependency_import_map: dict[str, str] = field(default_factory=dict)
    dependency_package_specs: dict[str, str] = field(default_factory=dict)
    dependency_install_log_filename: str = "dependency_installations.jsonl"
    dependency_install_summary_filename: str = "dependency_installations_summary.json"
    dependency_install_central_log_path: str = ""
    dependency_install_central_summary_path: str = ""


@dataclass
class ColdstartConfig:
    use_coldstart: bool
    task_json_path: str
    model_json_path: str
    description: str


@dataclass
class InitSolutionConfig:
    use: bool = False


@dataclass
class RuntimeConfig:
    """Process lifecycle, resume behavior, and state-file controls."""

    resume_run: bool = False
    # ``total`` keeps configured limits as cumulative run targets. ``additional``
    # treats them as fresh budget appended to the restored checkpoint.
    resume_budget_mode: str = "total"
    run_timestamp: str = ""
    cleanup_empty_workspace_on_exit: bool = True
    force_process_exit_on_timeout: bool = True
    exit_immediately_after_interrupt_checkpoint: bool = True
    write_pending_nodes: bool = True
    pending_nodes_filename: str = "pending_nodes.json"
    run_status_filename: str = "run_status.json"
    state_write_max_attempts: int = 5
    state_write_retry_delay_seconds: float = 0.05
    scheduler_poll_seconds: float = 1.0
    save_search_state: bool = True
    search_state_filename: str = "search_state.json"
    search_state_checkpoint_seconds: float = 5.0
    checkpoint_manifest_filename: str = "checkpoint_manifest.json"
    restore_inflight_actions: bool = True
    restore_random_state: bool = True
    service_startup_buffer_seconds: int = 1800
    graceful_shutdown_buffer_seconds: int = 600
    termination_wait_seconds: int = 20
    interruption_checkpoint_wait_seconds: int = 30
    snapshot_journal_max_bytes: int = 157286400
    snapshot_event_limit: int = 400
    snapshot_text_tail_chars: int = 200000
    job_status_tail_chars: int = 60000
    service_log_tail_chars: int = 200000
    service_last_error_chars: int = 300
    save_journal: bool = True
    save_filtered_journal: bool = True
    save_resolved_config: bool = True
    save_best_solution: bool = True


@dataclass
class LoggingConfig:
    """Brief/detailed log output controls."""

    write_brief_log: bool = True
    write_verbose_log: bool = True
    write_console_log: bool = True
    brief_log_filename: str = "MLEvolve.log"
    verbose_log_filename: str = "MLEvolve.verbose.log"
    brief_log_max_bytes: int = 67108864
    verbose_log_max_bytes: int = 268435456
    log_backup_count: int = 2
    suppress_httpx_logs: bool = True


@dataclass
class ResourceConfig:
    """Per-task CPU, host-memory, and accelerator visibility limits."""

    cpu_cores: int = 4
    memory_limit_gb: float = 8.0
    accelerator_mode: str = "all"
    accelerator_device_ids: list[str] = field(default_factory=list)
    monitor_interval_seconds: float = 0.5


@dataclass
class Config(Hashable):
    data_dir: Path | None
    dataset_dir: Path | None
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str | None
    exp_id: str

    torch_hub_dir: str
    pretrain_model_dir: str

    exec: ExecConfig
    agent: AgentConfig
    start_cpu_id: str
    cpu_number: str

    coldstart: ColdstartConfig

    use_grading_server: bool = False
    init_solution: InitSolutionConfig = field(default_factory=InitSolutionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)


def _normalize_model_base_url(model_name: str, base_url: str) -> str:
    """Normalize provider URLs when a model family requires a specific endpoint."""
    model_name = (model_name or "").strip().lower()
    base_url = (base_url or "").strip()
    if model_name.startswith("deepseek") and base_url in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/",
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/v1/",
    }:
        return "https://api.deepseek.com/beta"
    return base_url


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _resolve_config_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get("MLEVOLVE_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH.resolve()


def _load_cfg(
    path: Path | str | None = None,
    use_cli_args: bool = True,
) -> Config:
    cfg = OmegaConf.load(_resolve_config_path(path))
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path | str | None = None) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    # Generation and execution now share the same actual worker count. Silently
    # discard this retired key so saved configs from older runs remain resumable.
    search_cfg = OmegaConf.select(cfg, "agent.search")
    if search_cfg is not None and "generation_parallel_num" in search_cfg:
        del search_cfg["generation_parallel_num"]
    if search_cfg is not None and "pending_execution_headroom" in search_cfg:
        del search_cfg["pending_execution_headroom"]
    draft_cfg = OmegaConf.select(cfg, "agent.draft")
    if draft_cfg is not None and "decouple_generation_execution" in draft_cfg:
        del draft_cfg["decouple_generation_execution"]

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    runtime_cfg = getattr(cfg, "runtime", RuntimeConfig())
    env_resume = os.environ.get("MLEVOLVE_RESUME_RUN", "").strip().lower()
    resume_run = bool(getattr(runtime_cfg, "resume_run", False))
    if env_resume:
        resume_run = env_resume in {"1", "true", "yes", "on"}
    # generate experiment name and prefix with consecutive index
    if resume_run:
        cfg.log_dir = top_log_dir
        cfg.workspace_dir = top_workspace_dir
        cfg.exp_name = cfg.exp_name or top_log_dir.name or coolname.generate_slug(3)
    else:
        timestamp = (
            str(getattr(runtime_cfg, "run_timestamp", "") or "").strip()
            or os.environ.get("MLEVOLVE_RUN_TIMESTAMP", "").strip()
            or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        cfg.exp_name = f"{timestamp}_{cfg.exp_name or coolname.generate_slug(3)}"

        # If log_dir and workspace_dir point to the same path, treat it as a unified
        # "runs" root and place logs/workspace under the per-run directory
        if top_log_dir == top_workspace_dir:
            runs_root = top_log_dir
            runs_root.mkdir(parents=True, exist_ok=True)
            per_run_root = (runs_root / cfg.exp_name).resolve()
            cfg.log_dir = (per_run_root / "logs").resolve()
            cfg.workspace_dir = (per_run_root / "workspace").resolve()
        else:
            top_log_dir.mkdir(parents=True, exist_ok=True)
            top_workspace_dir.mkdir(parents=True, exist_ok=True)
            cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
            cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    cfg.resources.cpu_cores = max(1, int(cfg.resources.cpu_cores))
    cfg.resources.memory_limit_gb = max(0.0, float(cfg.resources.memory_limit_gb))
    cfg.resources.accelerator_mode = str(cfg.resources.accelerator_mode or "all").strip().lower()
    if cfg.resources.accelerator_mode not in {"all", "selected", "none"}:
        raise ValueError("resources.accelerator_mode must be one of: all, selected, none")
    cfg.resources.accelerator_device_ids = [
        str(item).strip().lower()
        for item in (cfg.resources.accelerator_device_ids or [])
        if str(item).strip()
    ]
    cfg.resources.monitor_interval_seconds = max(0.1, float(cfg.resources.monitor_interval_seconds))
    cfg.cpu_number = str(cfg.resources.cpu_cores)

    cfg.runtime.resume_budget_mode = str(
        cfg.runtime.resume_budget_mode or "total"
    ).strip().lower()
    if cfg.runtime.resume_budget_mode not in {"total", "additional"}:
        raise ValueError("runtime.resume_budget_mode must be one of: total, additional")

    # Normalize model endpoints after schema merge so runtime clients see the right URL.
    cfg.agent.code.base_url = _normalize_model_base_url(cfg.agent.code.model, cfg.agent.code.base_url)
    cfg.agent.feedback.base_url = _normalize_model_base_url(cfg.agent.feedback.model, cfg.agent.feedback.base_url)
    from agents.prompt_policy import normalize_output_language

    cfg.agent.output_language = normalize_output_language(cfg.agent.output_language)
    cfg.agent.code.context_window_tokens = max(0, int(cfg.agent.code.context_window_tokens or 0))
    cfg.agent.feedback.context_window_tokens = max(0, int(cfg.agent.feedback.context_window_tokens or 0))
    cfg.agent.draft.stepwise_context_headroom_ratio = min(
        0.5,
        max(0.05, float(cfg.agent.draft.stepwise_context_headroom_ratio)),
    )
    cfg.agent.code.api_key = (
        str(cfg.agent.code.api_key or "").strip()
        or os.environ.get("MLEVOLVE_CODE_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    )
    cfg.agent.feedback.api_key = (
        str(cfg.agent.feedback.api_key or "").strip()
        or os.environ.get("MLEVOLVE_FEEDBACK_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    )
    cfg.agent.memory_embedding_api_key = (
        str(cfg.agent.memory_embedding_api_key or "").strip()
        or os.environ.get("MLEVOLVE_EMBEDDING_API_KEY", "").strip()
        or os.environ.get("EMBEDDING_API_KEY", "").strip()
    )

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(_redacted_cfg(cfg)), "yaml", theme="paraiso-dark"))


def _redacted_cfg(cfg: Config):
    """Return a serializable config copy without credentials."""
    redacted = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for path in (
        "agent.code.api_key",
        "agent.feedback.api_key",
        "agent.memory_embedding_api_key",
    ):
        if OmegaConf.select(redacted, path) is not None:
            OmegaConf.update(redacted, path, "", merge=False)
    return redacted


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    env_resume = os.environ.get("MLEVOLVE_RESUME_RUN", "").strip().lower()
    resume_run = bool(getattr(getattr(cfg, "runtime", None), "resume_run", False))
    if env_resume:
        resume_run = env_resume in {"1", "true", "yes", "on"}
    input_dir = cfg.workspace_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    if not (resume_run and any(input_dir.iterdir())):
        copytree(cfg.data_dir, input_dir, use_symlinks=not cfg.copy_data)
        if cfg.preprocess_data:
            preproc_data(input_dir)
    elif cfg.preprocess_data:
        logger.info("Resume mode: reusing existing preprocessed workspace input.")


def save_run(cfg: Config, journal):
    Journal, JournalSnapshot, get_longest_path, get_path_to_node = _get_journal_classes()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = getattr(cfg, "runtime", None)
    if bool(getattr(runtime_cfg, "save_journal", True)):
        serialize.dump_json(
            JournalSnapshot.from_journal(journal),
            cfg.log_dir / "journal.json",
        )
    if bool(getattr(runtime_cfg, "save_filtered_journal", True)):
        best_for_path = journal.get_best_node(only_good=True)
        if best_for_path is not None:
            filtered_path = get_path_to_node(journal, best_for_path.id)
        else:
            filtered_path = get_longest_path(journal)
        serialize.dump_json(
            JournalSnapshot.from_journal(
                journal,
                node_ids=filtered_path,
                omit_execution_details=True,
            ),
            cfg.log_dir / "filtered_journal.json",
        )
    # save config
    if bool(getattr(runtime_cfg, "save_resolved_config", True)):
        OmegaConf.save(config=_redacted_cfg(cfg), f=cfg.log_dir / "config.yaml")
    
    # save the best found solution
    best_node = journal.get_best_node(only_good=True)
    if best_node is not None and bool(getattr(runtime_cfg, "save_best_solution", True)):
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
