"""Answer and code extraction utilities shared across all solutions."""

import re
from typing import List, Optional


def extract_answer(text: str) -> Optional[int]:
    """Extract the final \\boxed{} answer from model output."""
    patterns = [
        r"\\boxed\s*\{\s*([0-9,]+)\s*\}",
        r"final\s+answer\s+is\s*[:\s]*([0-9,]+)",
        r"answer\s*[:=]\s*\**\s*([0-9,]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            try:
                val = int(matches[-1].replace(",", ""))
                if 0 <= val <= 99999:
                    return val
            except ValueError:
                pass
    return None


def extract_code_blocks(text: str) -> List[str]:
    """Extract ```python ... ``` code blocks from model output."""
    return re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from DeepSeek-R1 style output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ensure_print(code: str) -> str:
    """Ensure the last expression in a code block is printed."""
    lines = code.strip().split("\n")
    if not lines:
        return code
    last = lines[-1].strip()
    if (
        "print" in last
        or "import" in last
        or not last
        or last.startswith("#")
        or last.startswith("def ")
        or last.startswith("class ")
        or "=" in last
    ):
        return code
    lines[-1] = f"print({last})"
    return "\n".join(lines)
