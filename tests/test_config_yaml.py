from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from config import Config, _load_cfg, _redacted_cfg, prep_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_commented_default_yaml_matches_runtime_schema() -> None:
    runtime = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    merged = OmegaConf.merge(OmegaConf.structured(Config), runtime)

    assert merged.data_dir is None
    assert merged.exp_name is None
    assert merged.agent.steps == 50
    assert merged.agent.time_limit == 10800
    assert merged.agent.output_language == "english"
    assert merged.agent.search.parallel_search_num == 4
    assert merged.runtime.resume_budget_mode == "total"
    assert merged.agent.search.num_drafts == 8
    assert merged.agent.search.num_improves == 5
    assert merged.agent.draft.fast_first_draft is True
    assert merged.agent.draft.fast_first_draft_skip_pre_review is True
    assert merged.agent.draft.fast_first_draft_max_repairs == 2
    assert merged.agent.draft.fast_first_draft_compact_context is False
    assert merged.exec.auto_install_missing_dependencies is True
    assert merged.exec.timeout == 1800
    assert merged.exec.dependency_install_policy == "ai_declared"
    assert merged.exec.dependency_install_target_path == ""
    assert merged.exec.dependency_install_lock_timeout_seconds == 900
    assert "ortools" in merged.exec.dependency_install_allowlist
    assert merged.exec.dependency_import_map.ortools == "ortools"
    assert merged.exec.dependency_import_map.sklearn == "scikit-learn"
    assert (
        merged.exec.dependency_package_specs["scikit-learn"]
        == "scikit-learn>=1.4.0,<2"
    )
    assert merged.exec.dependency_package_specs.ortools == "ortools>=9.9.0,<10"
    assert merged.agent.draft.use_stepwise_after_first is True
    assert merged.agent.retries.preflight_regeneration_max_attempts == 2
    assert merged.agent.draft.optimization_initial_drafts_cap == 0
    assert merged.agent.initial_drafts == 3
    assert merged.agent.draft.stepwise_stage_context is False
    assert merged.agent.draft.stepwise_accumulate_context is True
    assert merged.agent.draft.stepwise_context_max_tokens == 90000
    assert merged.agent.draft.stepwise_compaction_max_tokens == 32768
    assert merged.agent.draft.stepwise_compaction_keep_recent_steps == 2
    assert merged.agent.draft.stepwise_context_headroom_ratio == 0.15
    assert merged.agent.code.context_window_tokens == 131072
    assert merged.agent.code.minimum_output_tokens == 32768
    assert merged.agent.code.max_tokens == 32768
    assert merged.agent.feedback.minimum_output_tokens == 32768
    assert merged.agent.feedback.max_tokens == 32768
    assert merged.agent.code.request_timeout_seconds == 1200.0
    assert merged.agent.code.continuation_max_rounds == 2
    assert merged.agent.retries.result_parse_max_attempts == 3
    assert merged.agent.retries.result_adjudicator_on_anomaly is True
    assert merged.agent.retries.code_review_model_role == "feedback"
    assert merged.agent.retries.code_review_escalate_to_code is True
    assert merged.agent.retries.code_generation_extract_max_attempts == 2
    assert merged.agent.use_optimization_experience_library is True
    assert merged.agent.optimization_experience_min_score == 3.0
    assert merged.agent.optimization_experience_max_chars == 6000
    assert merged.resources.cpu_cores == 4
    assert merged.resources.memory_limit_gb == 8.0
    assert merged.resources.accelerator_mode == "all"
    assert merged.runtime.job_status_tail_chars == 60000
    assert merged.runtime.service_last_error_chars == 300
    assert merged.runtime.checkpoint_manifest_filename == "checkpoint_manifest.json"
    assert merged.runtime.interruption_checkpoint_wait_seconds == 30
    assert merged.runtime.exit_immediately_after_interrupt_checkpoint is True
    assert merged.runtime.service_startup_buffer_seconds == 1800
    assert merged.logging.brief_log_max_bytes == 67108864
    assert merged.logging.verbose_log_max_bytes == 268435456
    assert merged.logging.log_backup_count == 2


def test_environment_api_key_fallback(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    description = tmp_path / "description.md"
    description.write_text("demo", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-test-key")

    raw = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    raw.data_dir = str(data_dir)
    raw.desc_file = str(description)
    raw.log_dir = str(tmp_path / "logs")
    raw.workspace_dir = str(tmp_path / "workspaces")
    raw.runtime.run_timestamp = "20260710_000000"
    raw.exp_name = "config-test"
    raw.agent.search.generation_parallel_num = 99
    cfg = prep_cfg(raw)

    assert cfg.agent.code.api_key == "env-test-key"
    assert cfg.agent.feedback.api_key == "env-test-key"
    assert "generation_parallel_num" not in cfg.agent.search
    assert cfg.log_dir.name == "20260710_000000_config-test"
    redacted = _redacted_cfg(cfg)
    assert redacted.agent.code.api_key == ""
    assert redacted.agent.feedback.api_key == ""


def test_task_config_path_and_key_override_environment(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    description = tmp_path / "description.md"
    description.write_text("demo", encoding="utf-8")
    task_cfg = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    task_cfg.data_dir = str(data_dir)
    task_cfg.desc_file = str(description)
    task_cfg.log_dir = str(tmp_path / "logs")
    task_cfg.workspace_dir = str(tmp_path / "workspaces")
    task_cfg.exp_name = "priority-test"
    task_cfg.agent.code.api_key = "config-key"
    config_path = tmp_path / "task.yaml"
    OmegaConf.save(task_cfg, config_path)

    monkeypatch.setenv("MLEVOLVE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    loaded = prep_cfg(_load_cfg(use_cli_args=False))

    assert loaded.agent.code.api_key == "config-key"
