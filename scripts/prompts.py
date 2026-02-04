"""Prompt variants registry for AIMO3 trace generation.

Exact prompt text copied from kaggle_submissions/improv1_entropy_plus/kaggle_submission.ipynb.
"""

PROMPTS = {
    "reasoning": (
        "You are a world-class International Mathematical Olympiad (IMO) competitor. "
        "The final answer must be a non-negative integer between 0 and 99999. "
        "You must place the final integer answer inside \\boxed{}. /no_think"
    ),
    "code_first": (
        "You are a world-class IMO competitor who excels at computational verification. "
        "Write Python code to explore and verify your reasoning whenever possible. "
        "The final answer must be a non-negative integer between 0 and 99999. "
        "You must place the final integer answer inside \\boxed{}. /no_think"
    ),
    "case_analysis": (
        "You are a world-class IMO competitor. Break the problem into cases, "
        "solve each case carefully, and use Python code to check your work. "
        "The final answer must be a non-negative integer between 0 and 99999. "
        "You must place the final integer answer inside \\boxed{}. /no_think"
    ),
}

PREFERENCE_PROMPT = "You have access to `math`, `numpy` and `sympy` to solve the problem."

DEFAULT_MIX = {"reasoning": 8, "code_first": 2, "case_analysis": 2}


def get_prompt_sequence(mix=None):
    """Build a list of (prompt_type, system_prompt_text) from a mix dict.

    Args:
        mix: dict mapping prompt type name to count, e.g. {"reasoning": 8, "code_first": 2}.
             Defaults to DEFAULT_MIX.

    Returns:
        List of (prompt_type, prompt_text) tuples.
    """
    if mix is None:
        mix = DEFAULT_MIX
    seq = []
    for name, count in mix.items():
        if name not in PROMPTS:
            raise ValueError(f"Unknown prompt type: {name!r}. Available: {list(PROMPTS)}")
        for _ in range(count):
            seq.append((name, PROMPTS[name]))
    return seq


def parse_prompt_mix(s):
    """Parse a CLI string like 'reasoning:8,code_first:2,case_analysis:2' into a dict."""
    mix = {}
    for part in s.split(","):
        name, count = part.strip().split(":")
        mix[name.strip()] = int(count.strip())
    return mix
