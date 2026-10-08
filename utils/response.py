import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract complete scripts, including Markdown fences inside Python strings."""
    if is_valid_python_script(text):
        return format_code(text)
    if "```" not in text:
        return ""

    opening = re.compile(r"(?<!`)(`{3,})(?:python3?|py)?[ \t]*\r?\n", re.IGNORECASE)
    parsed_codes = []
    position = 0
    while match := opening.search(text, position):
        closing = re.compile(r"^[ \t]*`{" + str(len(match[1])) + r",}[ \t]*\r?$", re.MULTILINE)
        # An apparent fence may belong to a README inside a triple-quoted
        # string. Accept a boundary only when the complete prefix compiles.
        for end in closing.finditer(text, match.end()):
            candidate = text[match.end():end.start()].strip()
            if candidate and is_valid_python_script(candidate):
                parsed_codes.append(candidate)
                position = end.end()
                break
        else:
            return ""
    code = "\n\n".join(parsed_codes)
    return format_code(code) if code and is_valid_python_script(code) else ""


def extract_text_up_to_code(s):
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def extract_plan_and_code(text: str, default_plan: str) -> tuple[str, str]:
    """Extract valid Python code while treating the prose plan as optional.

    Prompts request a short plan for readability, but a response that starts
    directly with a valid Python fence is still a complete coding result and
    must not trigger an expensive full-generation retry.
    """

    code = extract_code(text or "")
    if not code:
        return "", ""
    plan = extract_text_up_to_code(text or "").strip() or default_plan.strip()
    return plan, code


def extract_plan_from_diff_response(text: str) -> str:
    if not text:
        return ""

    stop_tokens = [
        "<<<<<<< SEARCH",
        "< SEARCH",
        ">>>>>>> REPLACE",
        "=======",
        "```",
    ]

    def cut_at_stop(s: str) -> str:
        indices = [s.find(token) for token in stop_tokens if s.find(token) != -1]
        if indices:
            return s[: min(indices)]
        return s

    if "Fixed Code Plan:" in text:
        candidate = text.split("Fixed Code Plan:", 1)[1]
        return cut_at_stop(candidate).strip()

    if "Plan:" in text:
        candidate = text.split("Plan:", 1)[1]
        return cut_at_stop(candidate).strip()

    return cut_at_stop(text).strip()


def extract_review(text):
    parsed_codes = []

    matches = re.findall(r"```(json)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(json)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    if len(parsed_codes) == 0 or not parsed_codes[0].strip():
        json_objects = extract_jsons(text)
        if len(json_objects) > 0:
            return json_objects[0]
        raise ValueError(f"No JSON found in text")

    try:
        review = json.loads(parsed_codes[0].strip())
        return review
    except json.JSONDecodeError:
        json_objects = extract_jsons(text)
        if len(json_objects) > 0:
            return json_objects[0]
        raise


def format_code(code) -> str:
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
