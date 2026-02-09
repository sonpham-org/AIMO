"""
AIMO Trace Viewer — Flask app for visualizing math problem traces.

Supports two trace schemas:
  - Schema A (generate_traces.py): full_response_text with <think> blocks, code_executions
  - Schema B (local execution): structured turns with type: text/code/code_output
"""

import json
import re
from pathlib import Path

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Simple in-memory cache
_cache = {}

# Root directories containing trace data
TRACE_DIRS = [
    Path("/home/son/GitHub/AIMO/trace_gen/output/traces"),
    Path("/home/son/GitHub/AIMO/traces"),
]


def safe_json_load(filepath):
    """Load JSON file, handling Infinity/NaN values."""
    text = Path(filepath).read_text()
    text = text.replace(': Infinity', ': null').replace(':Infinity', ':null')
    text = text.replace(': -Infinity', ': null').replace(':-Infinity', ':null')
    text = text.replace(': NaN', ': null').replace(':NaN', ':null')
    return json.loads(text)


def discover_trace_sources():
    """Discover all trace directories and organize them."""
    sources = []
    for root in TRACE_DIRS:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            # Skip archive dirs
            if child.name.startswith("_"):
                continue
            # Check for trace files
            trace_files = list(child.glob("*.json"))
            problem_files = [
                f for f in trace_files
                if f.name.startswith("problem_") or re.match(r"\d{8}_\d{6}_problem_", f.name)
            ]
            if not problem_files:
                continue

            # Try to read config for model name
            model = "unknown"
            config_path = child / "config.json"
            run_config_path = child / "run_config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text())
                    model = cfg.get("model", "unknown")
                except Exception:
                    pass
            elif run_config_path.exists():
                try:
                    cfg = json.loads(run_config_path.read_text())
                    model = cfg.get("model", "unknown")
                except Exception:
                    pass

            # Fallback: read model from first problem file (Schema B)
            if model == "unknown" and problem_files:
                try:
                    pdata = json.loads(problem_files[0].read_text())
                    if "config" in pdata and "model" in pdata["config"]:
                        model = pdata["config"]["model"]
                    elif "phases" in pdata:
                        for phase in pdata["phases"].values():
                            if phase and "attempts" in phase:
                                for att in phase["attempts"]:
                                    if "model" in att:
                                        model = att["model"]
                                        break
                                if model != "unknown":
                                    break
                except Exception:
                    pass

            # Detect schema from first problem file
            schema = detect_schema(problem_files[0])

            sources.append({
                "id": f"{root.name}/{child.name}",
                "name": child.name,
                "path": str(child),
                "model": model,
                "n_problems": len(problem_files),
                "schema": schema,
            })
    return sources


def detect_schema(filepath):
    """Detect if a trace file uses Schema A or Schema B."""
    try:
        data = safe_json_load(filepath)
        if "phases" in data:
            return "B"
        if "attempts" in data:
            return "A"
    except Exception:
        pass
    return "unknown"


def load_problems_for_source(source_path):
    """Load all problem files from a source directory. Uses cache."""
    cache_key = f"problems:{source_path}"
    if cache_key in _cache:
        return _cache[cache_key]

    source = Path(source_path)
    problems = []
    for f in sorted(source.glob("*.json")):
        if f.name in ("config.json", "run_config.json", "summary.json"):
            continue
        if not (f.name.startswith("problem_") or re.match(r"\d{8}_\d{6}_problem_", f.name)):
            continue
        try:
            # Fast: only read first few KB to extract metadata
            raw = f.read_text(errors="replace")[:2000]
            # Handle Infinity/NaN for partial parse
            raw_clean = raw.replace(': Infinity', ': null').replace(': -Infinity', ': null').replace(': NaN', ': null')

            # Quick regex extraction instead of full JSON parse
            pid_match = re.search(r'"problem_id"\s*:\s*"([^"]*)"', raw_clean)
            pidx_match = re.search(r'"problem_index"\s*:\s*(\d+)', raw_clean)
            gt_match = re.search(r'"ground_truth"\s*:\s*(\d+(?:\.\d+)?|null)', raw_clean)
            text_match = re.search(r'"problem_text"\s*:\s*"((?:[^"\\]|\\.){0,200})', raw_clean)

            pid = pid_match.group(1) if pid_match else (pidx_match.group(1) if pidx_match else f.stem)
            gt = None
            if gt_match and gt_match.group(1) != "null":
                try:
                    gt = int(float(gt_match.group(1)))
                except ValueError:
                    gt = None
            text_preview = text_match.group(1)[:120] + "..." if text_match else ""
            # Unescape JSON string
            text_preview = text_preview.replace('\\"', '"').replace('\\n', ' ').replace('\\\\', '\\')

            problems.append({
                "file": f.name,
                "id": str(pid),
                "text_preview": text_preview,
                "ground_truth": gt,
            })
        except Exception:
            continue
    _cache[cache_key] = problems
    return problems


def load_trace(source_path, filename):
    """Load a single trace file and normalize to a common format."""
    filepath = Path(source_path) / filename
    data = safe_json_load(filepath)

    problem_text = data.get("problem_text", "")
    ground_truth = data.get("ground_truth")
    problem_id = data.get("problem_id", data.get("problem_index", ""))

    # Normalize attempts to a common format
    attempts = []

    if "phases" in data:
        # Schema B: local execution with structured turns
        config = data.get("config", {})
        for phase_name, phase in data.get("phases", {}).items():
            if not isinstance(phase, dict):
                continue
            for att in phase.get("attempts", []):
                turns = normalize_turns_schema_b(att.get("turns", []))
                attempts.append({
                    "index": att.get("Attempt", len(attempts) + 1),
                    "answer": att.get("Answer"),
                    "entropy": att.get("Entropy"),
                    "duration": att.get("Duration"),
                    "model": att.get("model", config.get("model", "unknown")),
                    "mode": att.get("mode", "tir"),
                    "python_calls": att.get("Python Calls", 0),
                    "python_errors": att.get("Python Errors", 0),
                    "response_length": att.get("Response Length", 0),
                    "token_budget": att.get("token_budget"),
                    "tokens_remaining": att.get("tokens_remaining"),
                    "turns": turns,
                    "phase": phase_name,
                })
    elif "attempts" in data:
        # Schema A: generate_traces.py output
        for att in data.get("attempts", []):
            turns = normalize_turns_schema_a(att)
            attempts.append({
                "index": att.get("attempt_idx", 0) + 1,
                "answer": att.get("answer"),
                "answer_source": att.get("answer_source"),
                "entropy": att.get("entropy"),
                "duration": att.get("wall_time_s"),
                "model": "gpt-oss-120b",
                "mode": att.get("prompt_type", "reasoning"),
                "python_calls": att.get("n_python_calls", 0),
                "python_errors": att.get("n_python_errors", 0),
                "response_length": att.get("total_response_tokens", 0),
                "turns": turns,
                "seed": att.get("seed"),
                "temperature": att.get("temperature"),
                "logprobs_summary": att.get("logprobs_summary"),
            })

    result = {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "ground_truth": ground_truth,
        "final_answer": data.get("final_answer", data.get("default_answer")),
        "selection_method": data.get("selection_method", data.get("default_method")),
        "wall_time": data.get("wall_time_s", data.get("total_seconds")),
        "attempts": attempts,
        "is_correct": data.get("is_correct"),
    }

    # Add vote distribution if available
    if "default_votes" in data:
        result["vote_distribution"] = data["default_votes"]

    return result


def normalize_turns_schema_b(turns):
    """Normalize Schema B turns (already structured)."""
    result = []
    for turn in turns:
        t = turn.get("type", "text")
        content = turn.get("content", "")
        if t == "text":
            result.append({"type": "text", "content": content})
        elif t == "code":
            result.append({"type": "code", "content": content})
        elif t == "code_output":
            result.append({
                "type": "code_output",
                "content": content,
                "success": turn.get("success", True),
            })
        else:
            result.append({"type": t, "content": content})
    return result


def normalize_turns_schema_a(attempt):
    """Normalize Schema A (full_response_text + code_executions) into structured turns."""
    text = attempt.get("full_response_text", "")
    code_execs = attempt.get("code_executions", [])

    turns = []

    # Parse the full_response_text for <think> blocks and code blocks
    # Split into think/content sections
    parts = parse_response_text(text)
    turns.extend(parts)

    # If there were code executions but they weren't in the text, append them
    if code_execs and not any(t["type"] == "code" for t in turns):
        for ce in code_execs:
            turns.append({"type": "code", "content": ce.get("code", "")})
            turns.append({
                "type": "code_output",
                "content": ce.get("output", ""),
                "success": not ce.get("is_error", False),
            })

    return turns


def parse_response_text(text):
    """Parse a response text with <think> blocks and code blocks into structured turns."""
    if not text:
        return [{"type": "text", "content": ""}]

    parts = []
    remaining = text

    while remaining:
        # Look for <think> block
        think_start = remaining.find("<think>")
        if think_start == 0:
            think_end = remaining.find("</think>")
            if think_end != -1:
                think_content = remaining[7:think_end]  # skip <think>
                parts.append({"type": "thinking", "content": think_content})
                remaining = remaining[think_end + 8:]  # skip </think>
                continue
            else:
                # Unclosed think tag - treat rest as thinking
                parts.append({"type": "thinking", "content": remaining[7:]})
                remaining = ""
                continue

        if think_start > 0:
            # Content before <think>
            before = remaining[:think_start]
            if before.strip():
                parts.extend(parse_code_blocks(before))
            remaining = remaining[think_start:]
            continue

        # No more think blocks - parse code blocks in remaining
        parts.extend(parse_code_blocks(remaining))
        remaining = ""

    return parts if parts else [{"type": "text", "content": text}]


def parse_code_blocks(text):
    """Parse text for ```python ... ``` code blocks."""
    parts = []
    pattern = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.DOTALL)

    last_end = 0
    for match in pattern.finditer(text):
        # Text before code block
        before = text[last_end:match.start()]
        if before.strip():
            parts.append({"type": "text", "content": before.strip()})
        # Code block
        parts.append({"type": "code", "content": match.group(1).strip()})
        last_end = match.end()

    # Remaining text after last code block
    after = text[last_end:]
    if after.strip():
        parts.append({"type": "text", "content": after.strip()})

    if not parts:
        parts.append({"type": "text", "content": text})

    return parts


# --- Routes ---

@app.route("/")
def index():
    sources = discover_trace_sources()
    return render_template("index.html", sources=sources)


@app.route("/api/sources")
def api_sources():
    return jsonify(discover_trace_sources())


@app.route("/api/problems")
def api_problems():
    source_path = request.args.get("source")
    if not source_path:
        return jsonify({"error": "source parameter required"}), 400
    problems = load_problems_for_source(source_path)
    return jsonify(problems)


@app.route("/api/trace")
def api_trace():
    source_path = request.args.get("source")
    filename = request.args.get("file")
    if not source_path or not filename:
        return jsonify({"error": "source and file parameters required"}), 400
    try:
        trace = load_trace(source_path, filename)
        return jsonify(trace)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
