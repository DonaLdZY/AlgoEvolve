"""Stepwise code generation mode.

Provides one three-stage workflow for prediction and one for decision tasks. The
stages share an append-only provider-friendly conversation and finish with a strict
MetaAgent integration pass.

Main entry: stepwise_plan_and_code_query()
"""

from __future__ import annotations

import logging
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

from llm import generate, compile_prompt_to_md
from utils.response import extract_plan_and_code, wrap_code
from agents.prompts import infer_task_mode, plan_and_code_response_format
from agents.prompt_cache import dataset_reference_sentence, task_section
from agents.memory.optimization_experience import build_optimization_experience_for_agent
from llm.usage import estimate_text_tokens
from agents.prompt_policy import (
    configured_output_language,
    output_language_instruction,
)

logger = logging.getLogger("MLEvolve")

STEPWISE_SYSTEM_PROMPT = (
    "You are MLEvolve's stepwise coding agent. Work through one cumulative conversation. "
    "Treat the task/data context and workflow contracts in the first user message as authoritative. "
    "Each later user message requests only the next stage; preserve all compatible decisions and exact "
    "identifiers from earlier turns. Return the requested plan and Python code without redoing other stages."
)
STEPWISE_CONTEXT_ACK = (
    "I have loaded the authoritative task/data context, shared instructions, and every stage contract. "
    "I will preserve them while handling the appended stage requests in order."
)


def _message_text(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{message.get('role', '')}]\n{message.get('content', '')}"
        for message in messages
    )


def _section_before(text: str, markers: tuple[str, ...]) -> str:
    end = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            end = min(end, idx)
    return text[:end].strip()


def _incremental_instruction(prompt: Dict[str, str], *, merge: bool = False) -> str:
    """Extract only the changing final instruction from a legacy step prompt."""

    user = str(prompt.get("user", "") or "")
    instruction_start = user.find("# Instructions")
    instructions = ""
    if instruction_start >= 0:
        instructions = _section_before(
            user[instruction_start:],
            ("\n# Memory", "\n# Previous solution", "\n# Previous steps", "\n# Step results"),
        )

    current_marker = "# Step results" if merge else "# Current step:"
    current_start = user.find(current_marker)
    current = ""
    if current_start >= 0 and not merge:
        current = user[current_start:].strip()
    elif merge:
        current = (
            "# Current step: merge\n"
            "Merge the code produced in the cumulative stage history into one cohesive runnable script. "
            "Use the earlier assistant turns as the source modules; do not redesign the selected method."
        )

    role = str(prompt.get("system", "") or "").strip()
    parts = ["# Appended step instruction", role, instructions, current]
    return "\n\n".join(part for part in parts if part).strip()


@dataclass
class StepwiseConversation:
    """Append-only stepwise history with bounded LLM compaction of old turns."""

    base_messages: List[Dict[str, str]]
    turns: List[Dict[str, str]] = field(default_factory=list)
    compacted_history: str = ""
    snapshots: List[Dict[str, str]] = field(default_factory=list)
    legacy_rebuild_mode: bool = False

    def _history_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.compacted_history:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": (
                            "# Compacted earlier stepwise history\n"
                            f"{self.compacted_history}\n\n"
                            "If an exact omitted detail is required, respond only with "
                            "`REQUEST_CONTEXT_SNAPSHOT: <snapshot_id> | <reason>`; the host will append the exact snapshot once."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "I will preserve this compacted history as prior work and continue from the "
                            "verbatim recent turns that follow."
                        ),
                    },
                ]
            )
        messages.extend(dict(message) for message in self.turns)
        return messages

    def _estimated_tokens(self, next_instruction: str = "") -> int:
        messages = [
            *self.base_messages,
            *self._history_messages(),
            {"role": "user", "content": next_instruction},
        ]
        return estimate_text_tokens(_message_text(messages))

    def _fallback_to_legacy_rebuild(self, reason: str) -> None:
        self.legacy_rebuild_mode = True
        logger.warning(
            "Stepwise cumulative context was cleared; reverting to the original per-step prompt rebuild: %s",
            reason,
        )

    def _compact_if_needed(self, next_instruction: str, agent_instance) -> None:
        if self.legacy_rebuild_mode:
            return
        draft_cfg = getattr(agent_instance.acfg, "draft", None)
        max_tokens = max(0, int(getattr(draft_cfg, "stepwise_context_max_tokens", 90000) or 0))
        context_window = max(
            0,
            int(
                getattr(
                    getattr(agent_instance.acfg, "code", None),
                    "context_window_tokens",
                    0,
                )
                or 0
            ),
        )
        headroom_ratio = min(
            0.5,
            max(0.05, float(getattr(draft_cfg, "stepwise_context_headroom_ratio", 0.15) or 0.15)),
        )
        if context_window > 0:
            window_budget = int(context_window * (1.0 - headroom_ratio))
            max_tokens = min(max_tokens, window_budget) if max_tokens > 0 else window_budget
        keep_steps = max(
            0,
            int(getattr(draft_cfg, "stepwise_compaction_keep_recent_steps", 2) or 0),
        )
        keep_messages = keep_steps * 2
        if max_tokens <= 0 or self._estimated_tokens(next_instruction) <= max_tokens:
            return
        if len(self.turns) <= keep_messages:
            logger.warning(
                "Stepwise prompt is above budget (%s tokens) but no older complete turns can be compacted.",
                self._estimated_tokens(next_instruction),
            )
            return

        split_at = len(self.turns) - keep_messages if keep_messages else len(self.turns)
        older_turns = self.turns[:split_at]
        recent_turns = self.turns[split_at:]
        older_text = _message_text(older_turns)
        snapshot_id = hashlib.sha256(older_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        workspace_dir = getattr(getattr(agent_instance, "cfg", None), "workspace_dir", None)
        snapshot_path: Path | None = None
        if workspace_dir is not None:
            snapshot_dir = Path(workspace_dir) / str(
                getattr(draft_cfg, "stepwise_snapshot_dirname", "context_snapshots")
                or "context_snapshots"
            )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"stepwise-{snapshot_id}.json"
            snapshot_path.write_text(
                json.dumps(
                    {"snapshot_id": snapshot_id, "messages": older_turns},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        snapshot_record = {
            "snapshot_id": snapshot_id,
            "path": str(snapshot_path) if snapshot_path is not None else "",
        }
        if snapshot_record not in self.snapshots:
            self.snapshots.append(snapshot_record)
        if self.compacted_history:
            older_text = (
                "[previous compacted history]\n"
                f"{self.compacted_history}\n\n"
                "[additional turns to compact]\n"
                f"{older_text}"
            )
        compaction_instruction = (
            "Compress the older stepwise coding history below into durable working memory for subsequent "
            "coding stages. Preserve exact file/sheet/column names, interfaces, function signatures, data "
            "structures, evaluator formulas, constraints, selected algorithms, dependencies, assumptions, "
            "unresolved risks, and integration requirements. Preserve code snippets only when exact syntax is "
            "needed downstream. Do not change facts or propose a new method. Return compact Markdown only.\n\n"
            f"Exact source snapshot (retrievable by the host): {snapshot_id}\n\n"
            f"{older_text}"
        )
        compaction_prompt = [
            *self.base_messages,
            {"role": "user", "content": compaction_instruction},
        ]
        output_limit = max(
            512,
            int(getattr(draft_cfg, "stepwise_compaction_max_tokens", 8192) or 8192),
        )
        logger.info(
            "Compacting stepwise history: estimated_tokens=%s older_messages=%s recent_messages=%s",
            self._estimated_tokens(next_instruction),
            len(older_turns),
            len(recent_turns),
        )
        try:
            summary = generate(
                prompt=compaction_prompt,
                temperature=0.0,
                max_tokens=output_limit,
                max_retries=2,
                cfg=agent_instance.cfg,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            self._fallback_to_legacy_rebuild(f"history compaction failed: {exc}")
            return
        if not summary:
            self._fallback_to_legacy_rebuild("history compaction returned empty content")
            return
        snapshot_index = json.dumps(self.snapshots, ensure_ascii=False, sort_keys=True)
        self.compacted_history = (
            f"{summary}\n\nExact-history snapshot index (host-retrievable): {snapshot_index}"
        )
        self.turns = recent_turns

    def messages_for(
        self,
        instruction: str,
        agent_instance,
    ) -> List[Dict[str, str]] | None:
        self._compact_if_needed(instruction, agent_instance)
        if self.legacy_rebuild_mode:
            return None
        return [
            *self.base_messages,
            *self._history_messages(),
            {"role": "user", "content": instruction},
        ]

    def record(self, instruction: str, response: str) -> None:
        self.turns.extend(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": str(response or "")},
            ]
        )

    def requested_snapshot(self, response: str) -> tuple[str, str] | None:
        match = re.match(
            r"^\s*REQUEST_CONTEXT_SNAPSHOT:\s*([a-fA-F0-9]{8,64})(?:\s*\|\s*(.*))?\s*$",
            str(response or ""),
            flags=re.DOTALL,
        )
        if not match:
            return None
        return match.group(1), str(match.group(2) or "exact prior context")[:500]

    def retrieve_snapshot(self, snapshot_id: str) -> str | None:
        record = next(
            (item for item in self.snapshots if item.get("snapshot_id") == snapshot_id),
            None,
        )
        if not record or not record.get("path"):
            return None
        path = Path(record["path"])
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list):
            return None
        return _message_text(
            [message for message in messages if isinstance(message, dict)]
        )


def _generate_submission_enabled(agent_instance) -> bool:
    return getattr(agent_instance.acfg, "generate_submission", True)


def _task_mode(agent_instance, task_desc: str = "", data_preview: str = "") -> str:
    mode = infer_task_mode(
        task_desc=task_desc or getattr(agent_instance, "task_desc", ""),
        coldstart_description=getattr(agent_instance, "coldstart_description", ""),
        autorealize_context=data_preview,
    )
    return "decision" if mode in {"optimization", "rl", "decision"} else "prediction"


def _method_route(agent_instance, task_desc: str = "", data_preview: str = "") -> str:
    """Keep required-method information orthogonal to the Prediction/Decision family."""

    return infer_task_mode(
        task_desc=task_desc or getattr(agent_instance, "task_desc", ""),
        coldstart_description=getattr(agent_instance, "coldstart_description", ""),
        autorealize_context=data_preview,
    )


@dataclass
class StepwiseContext:
    stage: str = "draft"
    task_mode: str = "prediction"
    method_route: str = "prediction"
    memory: str = ""
    previous_code: str = ""
    execution_output: str = ""
    optimization_experience: str = ""
    expansion_control: str = ""


@dataclass
class StepAgent:
    name: str
    introduction: str
    description: str
    guidelines: List[str]

    def generate(
        self,
        task_desc: str,
        data_preview: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 3,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
        conversation: StepwiseConversation | None = None,
    ) -> Tuple[str, str]:
        retry_cfg = getattr(agent_instance.acfg, "retries", None)
        retries = max(
            1,
            int(getattr(retry_cfg, "code_generation_extract_max_attempts", retries)),
        )
        prompt = self._build_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview,
            previous_steps=previous_steps,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
            improvement_mode=improvement_mode,
            previous_module_code=previous_module_code,
            improvement_strategy=improvement_strategy,
        )

        request_prompt: str | dict[str, str] | List[Dict[str, str]] = prompt
        incremental_instruction = ""
        if conversation is not None and isinstance(prompt, dict):
            incremental_instruction = _incremental_instruction(prompt)
            if context.expansion_control:
                incremental_instruction = (
                    f"{incremental_instruction}\n\n{context.expansion_control}"
                )
            cumulative_prompt = conversation.messages_for(
                incremental_instruction,
                agent_instance,
            )
            if cumulative_prompt is not None:
                request_prompt = cumulative_prompt
            else:
                incremental_instruction = ""

        completion_text = None
        snapshot_retrieved = False
        for _ in range(retries):
            completion_text = generate(
                prompt=request_prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            snapshot_request = (
                conversation.requested_snapshot(completion_text)
                if conversation is not None and not snapshot_retrieved
                else None
            )
            if snapshot_request is not None:
                snapshot_id, reason = snapshot_request
                exact_history = conversation.retrieve_snapshot(snapshot_id)
                if exact_history:
                    if incremental_instruction:
                        conversation.record(incremental_instruction, completion_text)
                    incremental_instruction = (
                        f"# Exact retrieved context snapshot: {snapshot_id}\n"
                        f"Reason requested: {reason}\n\n{exact_history}\n\n"
                        f"# Resume current stage\nNow complete only the `{self.name}` stage under the original latest instruction."
                    )
                    cumulative_prompt = conversation.messages_for(
                        incremental_instruction,
                        agent_instance,
                    )
                    if cumulative_prompt is not None:
                        request_prompt = cumulative_prompt
                        snapshot_retrieved = True
                        continue
            nl_text, code = extract_plan_and_code(
                completion_text,
                default_plan=f"Implement only the {self.name} stage.",
            )

            if code:
                if conversation is not None and incremental_instruction:
                    conversation.record(incremental_instruction, completion_text)
                if completion_text.lstrip().startswith("```"):
                    logger.info("Accepted a valid code-first response for %s.", self.name)
                return nl_text, code

            logger.debug(f"Extraction retry for {self.name}...")
        logger.warning(f"Code extraction failed after retries for {self.name}")
        if conversation is not None and incremental_instruction and completion_text:
            conversation.record(incremental_instruction, completion_text)
        return "", completion_text  # type: ignore

    def _build_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        previous_steps: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        improvement_mode: bool = False,
        previous_module_code: str = "",
        improvement_strategy: str = "",
    ) -> str | dict[str, str]:
        base_intro = prompt_base.get("Introduction", "")

        if context.stage == "improve":
            if improvement_mode and previous_module_code:
                step_specific_intro = (
                    f"You are currently working on improving the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the improved code for this specific step, based on the previous module code and the improvement strategy provided below. "
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}"
                )
            else:
                step_specific_intro = (
                    f"You are currently working on the '{self.name}' step of the solution. "
                    f"Your task is to write ONLY the code for this specific step that aligns with the overall improvement strategy. "
                    f"Base your implementation on the previous solution and execution results provided below, ensuring it integrates well with the improved approach."
                )
        else:
            step_specific_intro = (
                f"You are currently focusing on the '{self.name}' step of the solution. "
                f"Your task is to write ONLY the code for this specific step, not the complete solution."
            )
        introduction = base_intro + "\n\n" + step_specific_intro

        prev_summary = ""
        if previous_steps:
            prev_parts = []
            for step in previous_steps:
                prev_parts.append(f"### {step['name']}\n**Plan:** {step['plan']}\n**Code:**\n{wrap_code(step['code'])}")
            prev_summary = "\n\n".join(prev_parts)
        else:
            prev_summary = "This is the first step, no previous steps."

        generate_submission = _generate_submission_enabled(agent_instance)
        guidelines_to_use = self.guidelines.copy()
        current_step_description = self.description

        if self.name in {"inference_and_artifact", "solve_rollout_and_artifact"} and not generate_submission:
            current_step_description = (
                "Implement the training loop, validation, metric tracking, model saving, "
                "and configured non-submission artifacts."
            )
            guidelines_to_use.append(
                "CONFIG: Final submission generation is disabled. Do NOT force creation of `submission.csv`; "
                "focus on training, validation metric computation, and reusable inference code."
            )

        use_exact_coldstart_template = (
            hasattr(agent_instance, 'use_coldstart') and
            agent_instance.use_coldstart and
            hasattr(agent_instance, 'coldstart_description') and
            agent_instance.coldstart_description != "None model" and
            "Reference pattern" not in str(agent_instance.coldstart_description)
        )

        if use_exact_coldstart_template and context.stage == "draft":
            if self.name == "model_and_training":
                pretrain_emphasis = [
                    "**CRITICAL: You MUST prioritize using the recommended pretrained models provided in the Implementation guideline section below.**",
                    "The pretrained models are STRONGLY RECOMMENDED and should be your default first choice.",
                    "Only use custom architectures if the pretrained models are clearly unsuitable for this specific task."
                ]
                guidelines_to_use = pretrain_emphasis + guidelines_to_use
            elif self.name == "data_and_validation":
                pretrain_awareness = [
                    "**IMPORTANT: Be aware that pretrained models may be used in later steps. Consider the input requirements of common pretrained models (e.g., image size, normalization, data format) when preparing the data and engineering features.**",
                    "For image tasks, ensure data is prepared in a format compatible with standard pretrained models (e.g., PIL Image, numpy arrays, proper image sizes).",
                    "For text tasks, ensure text data is properly tokenized and formatted for potential transformer models.",
                ]
                guidelines_to_use = pretrain_awareness + guidelines_to_use

        guidelines_text = "\n".join([f"- {g}" for g in guidelines_to_use])

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = plan_and_code_response_format(
            f"code for the current `{self.name}` stage; do not implement other stages"
        )

        prompt_instructions[f"{self.name} guidelines"] = [guidelines_text]
        if context.optimization_experience and self.name == "decision_method":
            prompt_instructions["Retrieved optimization method experience"] = [
                context.optimization_experience
            ]

        if "Implementation guideline" in prompt_instructions:
            base_impl_guideline = prompt_instructions["Implementation guideline"]
            step_specific_impl = [
                "The code for this step must be self-contained and can be integrated with other steps.",
                "Use clear variable names that are consistent with previous steps.",
                "Do not duplicate code from previous steps - assume those parts already exist.",
                "Make sure to handle edge cases appropriately.",
            ]
            if isinstance(base_impl_guideline, list):
                prompt_instructions["Implementation guideline"] = base_impl_guideline + step_specific_impl
            else:
                prompt_instructions["Implementation guideline"] = [base_impl_guideline] + step_specific_impl

        stage_data_preview = data_preview_str
        logger.info(
            "Stepwise context route %s: %s -> %s chars",
            self.name,
            len(data_preview_str or ""),
            len(stage_data_preview or ""),
        )
        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Data preview": stage_data_preview,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Previous steps": prev_summary,
            "Current step": {
                "Name": self.name,
                "Description": current_step_description,
            },
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve":
            if improvement_mode and previous_module_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(previous_module_code),
                    "Note": f"This is the previous code for the '{self.name}' module. Improve it based on the improvement strategy provided above."
                }
            elif "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        if context.stage == "draft":
            okay_text = "Let me approach this systematically."
            assistant_suffix = ""
        elif context.stage == "improve":
            okay_text = "Let me approach this systematically."
            if improvement_mode and previous_module_code:
                previous_module_code_wrapped = wrap_code(previous_module_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously implemented the '{self.name}' module with the following code:\n{previous_module_code_wrapped}\n"
                    f"The execution of the full solution yielded the following results:\n{execution_output_wrapped}\n"
                    f"Improvement Strategy: {improvement_strategy if improvement_strategy else 'Improve this module based on the execution results.'}\n"
                    f"I need to improve this specific module according to the strategy above, ensuring it integrates well with the other modules."
                )
            elif context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            okay_text = "Let me approach this systematically."
            assistant_suffix = ""

        model_name = agent_instance.acfg.code.model.lower()

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

        previous_solution_section = ""
        if context.stage == "improve" and "Previous solution" in prompt:
            previous_solution_section = f"\n# Previous solution\n{prompt['Previous solution']['Code']}\n"

        user_prompt = (
            f"{task_section(prompt['Task description'], prompt['Data preview'])}\n"
            f"{instructions}"
            f"{memory_section}\n"
            f"{previous_solution_section}"
            f"# Previous steps\n{prompt['Previous steps']}\n\n"
            f"# Current step: {prompt['Current step']['Name']}\n{prompt['Current step']['Description']}\n\n"
        )
        assistant = f"{okay_text}\n{dataset_reference_sentence(prompt['Task description'], prompt['Data preview'])}{assistant_suffix}"
        return {
            "system": introduction,
            "user": user_prompt,
            "assistant": assistant,
        }



@dataclass
class MetaAgent:
    def merge(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        retries: int = 2,
        conversation: StepwiseConversation | None = None,
    ) -> Tuple[str, str]:
        prompt = self._build_merge_prompt(
            task_desc=task_desc,
            data_preview_str=data_preview_str,
            step_results=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=context,
        )

        request_prompt: str | dict[str, str] | List[Dict[str, str]] = prompt
        incremental_instruction = ""
        if conversation is not None and isinstance(prompt, dict):
            incremental_instruction = _incremental_instruction(prompt, merge=True)
            if context.expansion_control:
                incremental_instruction = (
                    f"{incremental_instruction}\n\n{context.expansion_control}"
                )
            cumulative_prompt = conversation.messages_for(
                incremental_instruction,
                agent_instance,
            )
            if cumulative_prompt is not None:
                request_prompt = cumulative_prompt
            else:
                incremental_instruction = ""

        completion_text = None
        for attempt in range(1, retries + 1):
            completion_text = generate(
                prompt=request_prompt,
                temperature=agent_instance.acfg.code.temp,
                cfg=agent_instance.cfg
            )
            snapshot_request = (
                conversation.requested_snapshot(completion_text)
                if conversation is not None and attempt == 1
                else None
            )
            if snapshot_request is not None:
                snapshot_id, reason = snapshot_request
                exact_history = conversation.retrieve_snapshot(snapshot_id)
                if exact_history:
                    if incremental_instruction:
                        conversation.record(incremental_instruction, completion_text)
                    incremental_instruction = (
                        f"# Exact retrieved context snapshot: {snapshot_id}\n"
                        f"Reason requested: {reason}\n\n{exact_history}\n\n"
                        "# Resume merge\nNow faithfully merge the three generated stages under the original latest instruction."
                    )
                    cumulative_prompt = conversation.messages_for(
                        incremental_instruction,
                        agent_instance,
                    )
                    if cumulative_prompt is not None:
                        request_prompt = cumulative_prompt
                        continue
            nl_text, code = extract_plan_and_code(
                completion_text,
                default_plan="Merge the generated stages into one runnable solution.",
            )

            if code:
                if conversation is not None and incremental_instruction:
                    conversation.record(incremental_instruction, completion_text)
                return nl_text or "Merged code from stepwise agents.", code

            logger.debug("Extraction retry for MetaAgent merge after attempt %s/%s...", attempt, retries)
        logger.error("Code extraction failed after %s MetaAgent merge attempts", retries)
        if conversation is not None and incremental_instruction and completion_text:
            conversation.record(incremental_instruction, completion_text)
        raise RuntimeError(
            "MetaAgent failed to produce one extractable faithful merge after the initial call and one retry; "
            "refusing to execute concatenated partial scripts."
        )

    def _build_merge_prompt(
        self,
        task_desc: str,
        data_preview_str: str,
        step_results: List[Dict[str, str]],
        prompt_base: Dict[str, Any],
        agent_instance,
        context: StepwiseContext,
        ) -> str | dict[str, str]:
        introduction = (
            "You are a Kaggle grandmaster attending a competition, an expert in writing clean, efficient, and competition-winning Python code for ML tasks. "
            "You have received code snippets from a team of specialized agents, each focusing on a specific part of the ML pipeline. "
            "Your critical task is to intelligently merge these partial scripts into a single, cohesive, and fully runnable Python script."
        )

        steps_summary = []
        for i, result in enumerate(step_results, 1):
            steps_summary.append(f"""
        ### Step {i}: {result['name']}
        **Plan:** {result['plan']}
        **Code:**
        {wrap_code(result['code'])}
        """)

        prompt_instructions = prompt_base["Instructions"].copy()

        prompt_instructions["Response format"] = plan_and_code_response_format(
            "the complete merged runnable solution"
        )

        decision_mode = context.task_mode == "decision"
        if decision_mode:
            output_guideline = (
                "- Preserve exactly one documented decision interface: non-RL code defines `solve(model_path, data)` (model_path may be None); RL/hybrid code defines `train_policy(data, artifact_dir)` plus `rollout(model_path, data)`. Validate the exact returned decision artifact, print the shared scalar metric, and save configured outputs."
                if _generate_submission_enabled(agent_instance)
                else "- Preserve exactly one documented decision interface: `solve(model_path, data)` or RL/hybrid `train_policy` + `rollout`. Validate and score the exact returned artifact; do not force submission.csv because output generation is disabled."
            )
        else:
            output_guideline = (
                "- Make sure the final code defines `train(data, artifact_dir)` and `predict(model_path, data)`, saves a reusable model/preprocessing artifact, prints the task validation metric, and saves submission.csv"
                if _generate_submission_enabled(agent_instance)
                else "- Make sure the final code defines `train(data, artifact_dir)` and `predict(model_path, data)`, saves a reusable model/preprocessing artifact, and prints the task validation metric; do not force submission.csv"
            )
        if decision_mode:
            execution_flow = (
                "problem data and shared evaluator -> one selected heuristic/optimization/search/RL/hybrid method -> "
                "solve or policy rollout -> independent validator/scorer replay -> configured artifacts"
            )
            interface_guideline = (
                "- Preserve the selected decision interface exactly: non-RL uses `solve(model_path, data)`; "
                "RL/hybrid uses `train_policy(data, artifact_dir)` plus `rollout(model_path, data)`."
            )
        else:
            execution_flow = "data and validation -> model and training -> fresh-load inference and artifacts"
            interface_guideline = (
                "- Expose both `train(data, artifact_dir)` and `predict(model_path, data)`; validation/test/output "
                "inference must use predict or the exact same internal path."
            )

        prompt_instructions["Merge guidelines"] = [
            "- Combine all code sections into a single, runnable Python script",
            "- CRITICAL: You are a MERGER, not a designer. Faithfully integrate the code from all steps. Do NOT introduce new models, algorithms, or approaches that were not in the original steps.",
            "- Ensure variable names are consistent across steps",
            "- Remove duplicate imports and definitions",
            "- Resolve conflicts by preserving the earlier stage's contract and the later stage's faithful implementation of that contract",
            f"- Ensure the execution flow is logical: {execution_flow}",
            output_guideline,
            "- The code should be a single-file Python program that can be executed as-is",
            "- Save model/preprocessing/policy state under ./working, ./models, ./artifacts, or ./checkpoints when the selected method has learned or fitted state. Stateless optimization solvers may use `model_path=None`.",
            interface_guideline,
            "- For decision tasks, keep `validate_solution` and `score_solution`, and compute `Final Validation Score` from that shared evaluator after validation.",
            "- For decision tasks, preserve an explicit `OUTPUT_COLUMNS`/submission schema and write generated result rows with `pd.DataFrame(rows, columns=OUTPUT_COLUMNS)`. Do not replace this with `pd.DataFrame(rows)[OUTPUT_COLUMNS]` or source-dataframe slicing.",
            "- For decision tasks, empty/diagnostic/no-feasible solutions must still run validation/scoring and print a final scalar score; optional diagnostics should use task-defined actionable details. Do not let output-table construction raise `KeyError` first.",
            "- When the selected method is RL/hybrid, the final evaluated solution must come from `rollout()` using the saved policy artifact without retraining or switching to another solver.",
            "- Assume previous steps have NOT been executed; do not skip execution steps and only read files or outputs.",
            "- All parts must work together seamlessly",
        ]

        merge_data_preview = data_preview_str
        logger.info(
            "Stepwise context route merge: %s -> %s chars",
            len(data_preview_str or ""),
            len(merge_data_preview or ""),
        )
        prompt: Dict[str, Any] = {
            "Introduction": introduction,
            "Task description": task_desc,
            "Memory": prompt_base.get("Memory", context.memory if context.memory else ""),
            "Data preview": merge_data_preview,
            "Step results": "".join(steps_summary),
            "Instructions": prompt_instructions,
        }

        if context.stage == "improve":
            if "Previous solution" in prompt_base:
                prompt["Previous solution"] = prompt_base["Previous solution"]
            elif context.previous_code:
                prompt["Previous solution"] = {
                    "Code": wrap_code(context.previous_code),
                }

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        memory_section = ""
        if prompt.get("Memory", "").strip():
            if context.stage == "improve":
                memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"
            else:
                memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

        okay_text = "Let me approach this systematically."

        if context.stage == "improve":
            if context.previous_code:
                previous_code_wrapped = wrap_code(context.previous_code)
                execution_output_wrapped = wrap_code(context.execution_output, lang="") if context.execution_output else "(No execution output available)"
                assistant_suffix = (
                    f"\nRegarding this task, I previously made attempts with the following code:\n{previous_code_wrapped}\n"
                    f"The execution of this code yielded the following results:\n{execution_output_wrapped}\n"
                    f"I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
                )
            else:
                assistant_suffix = ""
        else:
            memory_section = f"# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}"
            okay_text = "Let me approach this systematically."
            assistant_suffix = ""

        user_prompt = (
            f"{task_section(prompt['Task description'], prompt['Data preview'])}\n"
            f"{instructions}"
            f"{memory_section}\n\n"
            f"# Step results\n{prompt['Step results']}\n\n"
        )
        assistant = f"{okay_text}\n{dataset_reference_sentence(prompt['Task description'], prompt['Data preview'])}{assistant_suffix}"
        return {
            "system": introduction,
            "user": user_prompt,
            "assistant": assistant,
        }


def create_default_step_agents(task_mode: str = "prediction") -> List[StepAgent]:
    """Create the only two supported workflows: Prediction or Decision."""

    family = "decision" if task_mode in {"decision", "optimization", "rl"} else "prediction"
    if family == "prediction":
        return [
            StepAgent(
                name="data_and_validation",
                introduction="You own the prediction task's data contract and leakage-safe validation design.",
                description="Load exact source data, build preprocessing, and define the validation/evaluator contract.",
                guidelines=[
                    "Use exact source file, sheet, and column identifiers from the task context; prefer AutoRealize executable read recipes and verify observed shape/columns after loading.",
                    "Split before fitting stateful transforms and preserve a reproducible validation protocol aligned with the official metric.",
                    "Implement reusable data/preprocessing functions; do not train the final model in this stage.",
                ],
            ),
            StepAgent(
                name="model_and_training",
                introduction="You own the prediction model and bounded training procedure.",
                description="Implement the selected model family, objective, training, validation, and artifact serialization.",
                guidelines=[
                    "Use the method selected by the current expansion control and preserve the established validation contract.",
                    "Define `train(data, artifact_dir)` and return or save everything required for later inference.",
                    "Keep runtime within the search budget and report concise, trustworthy validation diagnostics.",
                ],
            ),
            StepAgent(
                name="inference_and_artifact",
                introduction="You own reusable prediction inference and final task artifacts.",
                description="Implement fresh-load inference, evaluator replay, and configured output generation.",
                guidelines=[
                    "Define `predict(model_path, data)` without retraining and make preprocessing identical to training-time validation.",
                    "Load the artifact produced by `train`, generate required outputs, and evaluate the exact returned predictions.",
                    "Print exactly one numeric `Final Validation Score` as the final line.",
                ],
            ),
        ]

    return [
        StepAgent(
            name="problem_and_evaluator",
            introduction="You own the decision problem contract and deterministic evaluator.",
            description="Load exact problem data and implement validation, constraints, and the one authoritative scalar score.",
            guidelines=[
                "Implement `load_problem_data`, `validate_solution`, and `score_solution` from the authoritative task contract; prefer AutoRealize executable read recipes and verify observed shape/columns after loading.",
                "Keep generated output schema separate from raw input schema and make empty/infeasible diagnostics schema-safe.",
                "Freeze evaluator behavior before method optimization; do not implement the final solver or policy here.",
            ],
        ),
        StepAgent(
            name="decision_method",
            introduction="You own one coherent decision method selected for this search child.",
            description="Implement the selected heuristic, local/metaheuristic, mathematical optimization, RL, or hybrid method.",
                guidelines=[
                    "Follow the latest sibling-complexity control; method family is orthogonal to the decision problem type.",
                    "RL is valid for static optimization when instances define state, legal actions, transitions, evaluator-aligned reward, and terminal conditions.",
                    "For a non-RL method define `solve(model_path, data)`; for RL/hybrid define `train_policy(data, artifact_dir)` and `rollout(model_path, data)`.",
                    "Mask or constrain illegal decisions before selection and keep all final scoring in the shared evaluator.",
                    "For mathematical/search methods, emit an `Optimization Structure Assessment` covering decision variables, aggregate state ranges, objective locality, relaxations, bounds, warm starts, and tractability before choosing reformulation details.",
                    "Use exact large-neighborhood search, decomposition, bound tightening, or hybridization only when the structure assessment supports it; never require a commercial solver.",
                    "For RL/hybrid methods, emit an `RL Design Summary` covering state, action, transition, masks, reward, terminal conditions, episode construction, and artifact contract.",
                    "For RL/hybrid methods, include a `Candidate/Action Probe Summary` and `Env Smoke Trace`; an empty legal-action mask must take a deterministic fallback/termination path before logits or softmax so it cannot create NaN probabilities.",
                    "For RL/hybrid methods, emit a `Curriculum Plan`: when suitable, progress from small to full instances, relaxed to original constraints, or short to full horizons, with advancement thresholds based on success/feasibility/evaluator metrics; if unsuitable, state why. Every curriculum level must preserve the final objective, and final acceptance must use the original full instance and official evaluator.",
                ],
        ),
        StepAgent(
            name="solve_rollout_and_artifact",
            introduction="You own execution, fresh-load reproducibility, and final decision artifacts.",
            description="Run solve or policy rollout, independently validate/score its exact output, and save configured artifacts.",
                guidelines=[
                "Use the decision method from the preceding stage without silently replacing it.",
                "A stateless solver may call `solve(None, data)`; RL/hybrid must load the saved policy and call `rollout(model_path, data)` without retraining.",
                    "Replay `validate_solution` and `score_solution` on the exact returned solution and expose task-relevant anomaly evidence.",
                    "For optimization/search methods, print an `Optimization Solver Summary` with known solver status, incumbent/bound/gap, model size, warm start, or neighborhood facts; omit unknown fields.",
                    "Print a `Method Usage Summary` identifying the actual evaluated family. If RL code exists but rollout does not use its policy, report `unused_rl_scaffold` rather than claiming RL success.",
                    "Print exactly one numeric `Final Validation Score` as the final line.",
            ],
        ),
    ]


def _build_stepwise_conversation(
    *,
    task_desc: str,
    data_preview: str,
    prompt_base: Dict[str, Any],
    step_agents: List[StepAgent],
    context: StepwiseContext,
    agent_instance,
) -> StepwiseConversation:
    workflow_parts: List[str] = []
    for index, step_agent in enumerate(step_agents, 1):
        guidelines = "\n".join(f"- {item}" for item in step_agent.guidelines)
        workflow_parts.append(
            f"## Stage {index}: {step_agent.name}\n"
            f"Role: {step_agent.introduction}\n"
            f"Deliverable: {step_agent.description}\n"
            f"Contract:\n{guidelines}"
        )
    workflow_parts.append(
        "## Final stage: merge\n"
        "Faithfully integrate all generated modules into one runnable script without changing the selected "
        "method. Resolve imports, names, execution order, the family-specific reusable interface, evaluator replay, artifacts, "
        "and configured output generation."
    )

    shared_instructions = compile_prompt_to_md(prompt_base.get("Instructions", {}), 2)
    stable_parts = [
        task_section(task_desc, data_preview).strip(),
        "# Complete stepwise workflow contracts\n" + "\n\n".join(workflow_parts),
    ]
    base_intro = str(prompt_base.get("Introduction", "") or "").strip()
    if base_intro:
        stable_parts.append(f"# Shared coding role\n{base_intro}")
    if shared_instructions.strip():
        stable_parts.append(f"# Shared instructions for every stage\n{shared_instructions}")
    if context.optimization_experience:
        stable_parts.append(
            "# Retrieved optimization experience\n"
            f"{context.optimization_experience}"
        )
    if context.memory:
        stable_parts.append(f"# Search memory available to every stage\n{context.memory}")
    if context.previous_code:
        stable_parts.append(
            "# Previous full solution available to every improvement stage\n"
            f"{wrap_code(context.previous_code)}"
        )
    if context.execution_output:
        stable_parts.append(
            "# Previous execution output available to every improvement stage\n"
            f"{wrap_code(context.execution_output, lang='')}"
        )

    return StepwiseConversation(
        base_messages=[
            {
                "role": "system",
                "content": (
                    f"{STEPWISE_SYSTEM_PROMPT} "
                    f"{output_language_instruction(configured_output_language(agent_instance))}"
                ),
            },
            {"role": "user", "content": "\n\n".join(stable_parts)},
            {"role": "assistant", "content": STEPWISE_CONTEXT_ACK},
        ]
    )


def stepwise_plan_and_code_query(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    ) -> Tuple[str, str]:
    task_mode = _task_mode(
        agent_instance,
        prompt_base["Task description"],
        data_preview,
    )
    method_route = _method_route(
        agent_instance,
        prompt_base["Task description"],
        data_preview,
    )
    logger.info("Using stepwise generation route: mode=%s", task_mode)

    stepwise_context = StepwiseContext(
        stage=context.get("stage", "draft"),
        task_mode=task_mode,
        method_route=method_route,
        memory=context.get("memory", ""),
        previous_code=context.get("previous_code", ""),
        execution_output=context.get("execution_output", ""),
        optimization_experience=build_optimization_experience_for_agent(
            agent_instance,
            task_mode=method_route,
            extra_context="\n".join(
                [
                    str(context.get("memory", "") or ""),
                    str(context.get("execution_output", "") or ""),
                ]
            ),
        ),
        expansion_control=str(context.get("expansion_control", "") or ""),
    )

    step_agents = create_default_step_agents(task_mode=task_mode)
    meta_agent = MetaAgent()
    accumulate_context = bool(
        getattr(
            getattr(agent_instance.acfg, "draft", None),
            "stepwise_accumulate_context",
            True,
        )
    )
    conversation = (
        _build_stepwise_conversation(
            task_desc=prompt_base["Task description"],
            data_preview=data_preview,
            prompt_base=prompt_base,
            step_agents=step_agents,
            context=stepwise_context,
            agent_instance=agent_instance,
        )
        if accumulate_context
        else None
    )
    logger.info(
        "Stepwise cumulative context: enabled=%s stable_chars=%s max_tokens=%s",
        accumulate_context,
        len(_message_text(conversation.base_messages)) if conversation else 0,
        getattr(getattr(agent_instance.acfg, "draft", None), "stepwise_context_max_tokens", 0),
    )

    step_results: List[Dict[str, str]] = []
    for idx, agent in enumerate(step_agents, 1):
        logger.info(f"Step {idx}/{len(step_agents)}: {agent.name}")

        plan, code = agent.generate(
            task_desc=prompt_base["Task description"],
            data_preview=data_preview,
            previous_steps=step_results,
            prompt_base=prompt_base,
            agent_instance=agent_instance,
            context=stepwise_context,
            conversation=conversation,
        )

        step_results.append({
            "name": agent.name,
            "plan": plan,
            "code": code,
        })

    logger.info("Merging all steps...")
    final_plan, final_code = meta_agent.merge(
        task_desc=prompt_base["Task description"],
        data_preview_str=data_preview,
        step_results=step_results,
        prompt_base=prompt_base,
        agent_instance=agent_instance,
        context=stepwise_context,
        conversation=conversation,
    )

    logger.info("Stepwise generation completed.")

    return final_plan, final_code
