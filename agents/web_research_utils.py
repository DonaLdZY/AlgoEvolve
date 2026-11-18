"""Auto-extract web search query from task description.

v0.11.0 enhancement: automatically generate meaningful search queries when
web_research is enabled but no explicit query is provided.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("AlgoEvolve")


def extract_search_query_from_task(
    task_desc: str,
    coldstart_desc: str = "",
    task_mode: str = "prediction",
) -> Optional[str]:
    """Extract a meaningful web search query from task description.

    Args:
        task_desc: Main task description
        coldstart_desc: Optional coldstart/user-provided description
        task_mode: "prediction"|"optimization"|"rl"|"unknown"

    Returns:
        Search query string or None if can't extract meaningful query

    Strategy:
        1. Check for explicit domain keywords (NLP, CV, time series, etc.)
        2. Identify task type (classification, regression, forecasting, etc.)
        3. Combine into focused query for ML methods

    Examples:
        "predict house prices" → "machine learning regression methods"
        "classify sentiment" → "text classification machine learning"
        "forecast sales time series" → "time series forecasting methods"
    """

    if not task_desc:
        return None

    text = (task_desc + " " + coldstart_desc).lower()

    # Domain detection
    domain = None
    if any(kw in text for kw in ["time series", "temporal", "sequential", "forecast"]):
        domain = "time series"
    elif any(kw in text for kw in ["nlp", "text", "sentiment", "language", "document"]):
        domain = "natural language processing"
    elif any(kw in text for kw in ["image", "vision", "cv", "photo", "picture"]):
        domain = "computer vision"
    elif any(kw in text for kw in ["graph", "network", "node", "edge"]):
        domain = "graph machine learning"
    elif any(kw in text for kw in ["tabular", "table", "structured data"]):
        domain = "tabular data"

    # Task type detection
    task_type = None
    if task_mode == "prediction":
        if any(kw in text for kw in ["classify", "classification", "category", "label"]):
            task_type = "classification"
        elif any(kw in text for kw in ["regress", "regression", "predict", "continuous"]):
            task_type = "regression"
        elif any(kw in text for kw in ["forecast", "future", "next"]):
            task_type = "forecasting"
        elif any(kw in text for kw in ["recommend", "recommendation", "ranking"]):
            task_type = "recommendation"
    elif task_mode == "optimization":
        task_type = "optimization"
    elif task_mode == "rl":
        task_type = "reinforcement learning"

    # Compose query
    parts = []
    if domain and task_type and domain in task_type:
        # Avoid redundancy: "time series forecasting" already contains "time series"
        parts.append(task_type)
    else:
        if domain:
            parts.append(domain)
        if task_type:
            parts.append(task_type)

    if not parts:
        # Fallback: generic ML query
        parts.append("machine learning")
        if task_mode and task_mode != "unknown":
            parts.append(task_mode)

    parts.append("methods best practices")

    query = " ".join(parts)

    # Clean up
    query = re.sub(r'\s+', ' ', query).strip()

    # Validate length (DeepSeek Search wants 1-128 chars)
    if 1 <= len(query) <= 128:
        logger.info(f"[web-research] Auto-extracted query: {query}")
        return query

    return None


def should_enable_web_research_for_task(
    web_research_config,
    task_desc: str = "",
    coldstart_desc: str = "",
) -> bool:
    """Check if web research should be enabled for this task.

    Returns True if:
    1. Config explicitly enables it, OR
    2. Config is None/empty but task seems suitable for web research

    Returns False if config explicitly disables it.
    """

    if web_research_config is None:
        return False

    enabled = getattr(web_research_config, "enabled", False)

    # Respect explicit disable
    if enabled is False:
        return False

    # Already enabled with explicit query
    query = getattr(web_research_config, "query", "").strip()
    if enabled and query:
        return True

    # Enabled but no query: check if task is suitable
    if enabled:
        # Most prediction/optimization tasks benefit from method search
        text = (task_desc + " " + coldstart_desc).lower()

        # Skip trivial toy tasks
        if any(kw in text for kw in [
            "hello world", "test", "example", "demo", "toy problem",
            "y = 2*x", "y = x + 5",  # Linear toy examples
        ]):
            logger.info("[web-research] Skipping for toy/test task")
            return False

        return True

    return False
