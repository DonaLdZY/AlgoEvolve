"""Build guidance description for agent from task/model JSON."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"
logger = logging.getLogger("AlgoEvolve")

CATEGORIES = ["General Image", "Detection", "Segmentation", "NLP", "Audio", "Optimization", "Others"]

CLASSIFY_SYSTEM_PROMPT = """You are a machine learning task classifier.
Given a machine learning competition/task name and description, classify it by calling the classify function.

Rules:
- "General Image": image classification, image regression, or other general image tasks that are NOT detection or segmentation
- "Detection": object detection, bounding box prediction, localization, 3D object detection -- any task predicting bounding boxes or object locations
- "Segmentation": image segmentation, pixel-level labeling, mask prediction tasks
- "NLP": text classification, NER, QA, text generation, sentiment analysis, code understanding, or any text/language-based task
- "Audio": audio classification, speech recognition, music tagging, sound event detection, or any audio-based task
- "Optimization": scheduling, routing, assignment, resource allocation, portfolio/knapsack, combinatorial optimization, sequential decision-making, control, simulator/environment tasks, reinforcement learning, MDP, policy learning, reward-driven optimization
- "Others": tabular prediction, time series forecasting, graph learning, video, molecular, signal processing, recommendation, or anything not fitting above and not primarily an optimization/decision task

You MUST call the classify function with your answer."""


def _authoritative_category(cfg: Any, task_desc: str) -> str | None:
    """Read an explicit AutoRealize paradigm before asking a classifier LLM."""

    texts = [str(task_desc or "")]
    try:
        from utils.autorealize_context import build_autorealize_context_md

        raw_data_dir = getattr(cfg, "data_dir", "")
        if raw_data_dir:
            data_dir = Path(raw_data_dir)
            texts.append(
                build_autorealize_context_md(data_dir, write_context_file=False)
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not inspect AutoRealize paradigm for cold start: %s", exc)

    text = "\n".join(texts)
    match = re.search(
        r"(?im)^\s*[-*]?\s*(?:problem[ _]paradigm|Problem paradigm)\s*:\s*`?"
        r"(static_optimization|reinforcement_learning)\b",
        text,
    )
    return "Optimization" if match else None


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_models_for_task(
    task_name: str, tasks: Dict, models: Dict
) -> List[Dict[str, str]]:
    """Match model list for task from knowledge by task name."""
    if task_name not in tasks:
        return []
    category = tasks[task_name]  # flat string: "General Image", "NLP", etc.
    return collect_models_for_category(category, models)


def collect_models_for_category(category: str, models: Dict) -> List[Dict[str, str]]:
    """Collect model guidance entries for a task category."""
    if category not in models:
        return []
    matched = []
    for m_name, m_info in models[category].items():
        matched.append({
            "model_name": m_name,
            "description": m_info.get("Description", ""),
            "code_template": m_info.get("Code_template", ""),
            "copy_exact": m_info.get("Copy_exact", True),
        })
    return matched


def _build_guidance_text_from_models(model_list: List[Dict[str, str]]) -> str:
    """Build prompt text from collected model guidance entries."""
    if not model_list:
        return "None model"
    lines = []
    for i, m in enumerate(model_list):
        lines.append(f"\nModel{i+1}: {m['model_name']}\n")
        lines.append(f"Description:{m['description']}\n")
        if m.get("copy_exact", True):
            lines.append(
                "Code template (MUST copy exactly - do NOT change model variant names or file paths):\n"
                "```python\n" + m["code_template"] + "\n```"
            )
        else:
            lines.append(
                "Reference pattern (adapt to this task; do NOT copy blindly):\n"
                "```python\n" + m["code_template"] + "\n```"
            )
    return "\n".join(lines)


def _build_guidance_text(task_name: str, tasks: Dict, models: Dict) -> str:
    """Build guidance text from task name and knowledge."""
    return _build_guidance_text_from_models(collect_models_for_task(task_name, tasks, models))


def _classify_task_with_llm(cfg: Any, task_desc: str) -> str | None:
    """Classify an unmapped task at runtime using the configured feedback model."""
    if not task_desc or not task_desc.strip():
        return None

    try:
        from llm import FunctionSpec, query
    except Exception as exc:
        logger.warning("Cold-start runtime task classification unavailable: %s", exc)
        return None

    func_spec = FunctionSpec(
        name="classify",
        description="Submit the classification result for a machine learning task",
        json_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": CATEGORIES,
                    "description": "The task category",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reason for choosing this category based on the task description.",
                },
            },
            "required": ["category", "reasoning"],
        },
    )

    task_name = str(getattr(cfg, "exp_id", "") or getattr(cfg, "exp_name", "") or "task")
    user_prompt = (
        f"Task name: {task_name}\n\n"
        "Description:\n"
        f"{task_desc[:8000]}"
    )
    model = getattr(getattr(cfg.agent, "feedback", None), "model", "") or getattr(cfg.agent.code, "model", "")
    temperature = getattr(getattr(cfg.agent, "feedback", None), "temp", 0.0)

    for attempt in range(1, 4):
        try:
            response = query(
                system_message=CLASSIFY_SYSTEM_PROMPT,
                user_message=user_prompt,
                func_spec=func_spec,
                model=model,
                temperature=temperature,
                stage_name="feedback",
                cfg=cfg,
            )
            if not isinstance(response, dict):
                logger.warning("Cold-start classification returned non-dict response: %r", response)
                continue
            category = str(response.get("category") or "").strip()
            if category in CATEGORIES:
                logger.info(
                    "Cold-start runtime classification: exp_id=%s category=%s reasoning=%s",
                    task_name,
                    category,
                    response.get("reasoning", ""),
                )
                return category
            logger.warning("Cold-start classification returned invalid category: %r", category)
        except Exception as exc:
            logger.warning("Cold-start classification attempt %s/3 failed: %s", attempt, exc)
    return None


def get_init_solution_paths(exp_id: str) -> List[str]:
    """Load init solution paths for exp_id from engine/coldstart/init_solution_paths.json."""
    if not INIT_SOLUTION_JSON.exists():
        return []
    try:
        data = _load_json(str(INIT_SOLUTION_JSON))
        paths = data.get(exp_id)
        if isinstance(paths, list):
            return [str(p) for p in paths if p]
        return []
    except Exception:
        return []


def build_guidance_description(cfg: Any, task_desc: str = "") -> str:

    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    authoritative_category = _authoritative_category(cfg, task_desc)
    if authoritative_category:
        category = authoritative_category
        logger.info(
            "Cold-start category loaded directly from AutoRealize context: %s",
            category,
        )
        text = _build_guidance_text_from_models(collect_models_for_category(category, models))
    elif cfg.exp_id in tasks:
        category = tasks[cfg.exp_id]
        logger.info("Cold-start matched exp_id=%s to category=%s", cfg.exp_id, category)
        if category == "Others" and task_desc:
            runtime_category = _classify_task_with_llm(cfg, task_desc)
            if runtime_category == "Optimization":
                logger.info(
                    "Cold-start remapped exp_id=%s from Others to Optimization based on task description",
                    cfg.exp_id,
                )
                category = runtime_category
        text = _build_guidance_text_from_models(collect_models_for_category(category, models))
        if text == "None model":
            logger.info("Cold-start category=%s has no model guidance; using None model", category)
    else:
        logger.info("Cold-start exp_id=%s not found in task map; classifying task with LLM", cfg.exp_id)
        category = _classify_task_with_llm(cfg, task_desc)
        if category:
            model_list = collect_models_for_category(category, models)
            text = _build_guidance_text_from_models(model_list)
            if text == "None model":
                logger.info("Cold-start category=%s has no model guidance; using None model", category)
        else:
            text = "None model"
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))
    return text
