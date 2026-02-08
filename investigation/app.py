#!/usr/bin/env python3
"""
Trace Comparison Viewer — Flask app to compare original vs shortened traces.
Dark mode, keyboard navigation, side-by-side view.

Usage:
    python investigation/app.py
    # Opens at http://localhost:5050
"""

import json
import re
import sys
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[1]
ORIG_PATH = PROJECT / "dataset_generation" / "output" / "curated_sft_5000.jsonl"
SHORT_PATH = PROJECT / "dataset_generation" / "output" / "shortened_flash_5000.jsonl"


def parse_harmony(text: str) -> list[dict]:
    """Parse Harmony format into structured turns for display."""
    if "<|start|>" not in text:
        return [{"role": "assistant", "content": text}]

    turns = []
    segments = re.split(r'<\|end\|>', text)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Remove leading <|start|>
        seg = re.sub(r'^<\|start\|>', '', seg)

        # Detect role
        role_match = re.match(
            r'(system|user|assistant)(?:\s+to=(\w+))?<\|(?:channel\|>[^<]*)?<?\|?message\|>(.+)',
            seg, re.DOTALL
        )
        if role_match:
            role = role_match.group(1)
            target = role_match.group(2)
            content = role_match.group(3).strip()
            # Clean up return markers
            content = re.sub(r'<\|return\|>.*$', '', content, flags=re.DOTALL).strip()
            if role == "system":
                continue  # Skip system prompts
            label = role
            if target:
                label = f"{role} → {target}"
            turns.append({"role": label, "content": content})
            continue

        # Tool call pattern
        call_match = re.search(r'<\|call\|>(.+?)(?:<\|message\|>(.+))?$', seg, re.DOTALL)
        if call_match:
            code = call_match.group(1).strip()
            output = (call_match.group(2) or "").strip()
            turns.append({"role": "code", "content": code})
            if output:
                turns.append({"role": "output", "content": output[:2000]})
            continue

        # Fallback: treat as content if non-trivial
        cleaned = re.sub(r'<\|[^|]+\|>', '', seg).strip()
        if cleaned and len(cleaned) > 10:
            turns.append({"role": "unknown", "content": cleaned[:2000]})

    return turns if turns else [{"role": "raw", "content": text[:5000]}]


def load_data():
    """Load and merge original + shortened traces."""
    originals = {}
    with open(ORIG_PATH) as f:
        for line in f:
            d = json.loads(line)
            originals[d["uid"]] = d

    pairs = []
    with open(SHORT_PATH) as f:
        for line in f:
            short = json.loads(line)
            uid = short["uid"]
            orig = originals.get(uid)
            if not orig:
                continue

            pairs.append({
                "uid": uid,
                "source": short.get("source", ""),
                "topic": short.get("topic", ""),
                "answer": str(short.get("answer", "")),
                "pass_rate": short.get("pass_rate"),
                "problem_text": short["problem_text"],
                "original_turns": parse_harmony(orig["solution_text"]),
                "original_chars": len(orig["solution_text"]),
                "shortened_text": short["solution_text"],
                "shortened_chars": len(short["solution_text"]),
                "shortening": short.get("shortening", {}),
                "scores": {
                    "combined": short.get("combined_score"),
                    "difficulty": short.get("difficulty_score"),
                    "quality": short.get("quality_score"),
                },
            })

    return pairs


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
DATA = []

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace Comparison Viewer</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: #0d1117; color: #c9d1d9; line-height: 1.5;
  }
  /* Top nav */
  .topbar {
    position: sticky; top: 0; z-index: 100;
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 8px 20px; display: flex; align-items: center; gap: 12px;
  }
  .topbar button {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 14px;
    font-family: inherit;
  }
  .topbar button:hover { background: #30363d; }
  .topbar button:disabled { opacity: 0.4; cursor: default; }
  .topbar .counter { color: #8b949e; font-size: 14px; }
  .topbar .jump-input {
    background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
    padding: 5px 10px; border-radius: 6px; width: 70px; font-family: inherit;
    font-size: 14px; text-align: center;
  }
  .topbar .badge {
    padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 600;
  }
  .badge-ok { background: #238636; color: #fff; }
  .badge-err { background: #da3633; color: #fff; }
  .badge-verified { background: #1f6feb; color: #fff; }
  .badge-unverified { background: #6e7681; color: #c9d1d9; }
  .badge-topic { background: #30363d; color: #c9d1d9; }

  /* Meta row */
  .meta {
    padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d;
    display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #8b949e;
  }
  .meta span { display: inline-flex; align-items: center; gap: 4px; }
  .meta .label { color: #484f58; }

  /* Problem statement */
  .problem-box {
    margin: 16px 20px; padding: 16px; background: #161b22;
    border: 1px solid #30363d; border-radius: 8px;
  }
  .problem-box h3 { color: #58a6ff; margin-bottom: 8px; font-size: 14px; }
  .problem-box .text { white-space: pre-wrap; font-size: 14px; }

  /* Side by side */
  .columns {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0;
    margin: 0 20px 20px 20px;
  }
  .column {
    border: 1px solid #30363d; border-radius: 8px; overflow: hidden;
  }
  .column:first-child { border-right: none; border-radius: 8px 0 0 8px; }
  .column:last-child { border-radius: 0 8px 8px 0; }
  .col-header {
    padding: 10px 16px; background: #161b22;
    border-bottom: 1px solid #30363d;
    font-weight: 600; font-size: 13px; color: #8b949e;
    display: flex; justify-content: space-between;
  }
  .col-body { padding: 16px; max-height: 75vh; overflow-y: auto; }

  /* Turns (original) */
  .turn { margin-bottom: 12px; }
  .turn-role {
    font-size: 11px; text-transform: uppercase; font-weight: 700;
    margin-bottom: 4px; letter-spacing: 0.5px;
  }
  .turn-role.assistant { color: #58a6ff; }
  .turn-role.code { color: #d2a8ff; }
  .turn-role.output { color: #7ee787; }
  .turn-role.user { color: #f0883e; }
  .turn-role.unknown, .turn-role.raw { color: #6e7681; }
  .turn-content {
    white-space: pre-wrap; font-size: 13px;
    padding: 8px 12px; background: #0d1117; border-radius: 6px;
    border: 1px solid #21262d; overflow-x: auto;
  }
  .turn-content.code-block {
    background: #161b22; border-color: #30363d;
  }

  /* Shortened (right) */
  .shortened-text {
    white-space: pre-wrap; font-size: 13px; line-height: 1.6;
  }
  .shortened-text code {
    background: #161b22; padding: 2px 6px; border-radius: 4px;
    font-size: 12px;
  }

  /* Keyboard hint */
  .hint {
    text-align: center; padding: 8px; font-size: 12px; color: #484f58;
  }
  kbd {
    background: #21262d; border: 1px solid #30363d; padding: 2px 6px;
    border-radius: 4px; font-size: 11px;
  }

  /* Compression bar */
  .compression-bar {
    height: 4px; background: #21262d; border-radius: 2px; margin-top: 4px;
  }
  .compression-fill {
    height: 100%; background: #238636; border-radius: 2px;
  }
</style>
</head>
<body>
  <div class="topbar">
    <button id="btn-prev" onclick="navigate(-1)">← Prev</button>
    <button id="btn-next" onclick="navigate(1)">Next →</button>
    <input type="number" class="jump-input" id="jump" min="1" placeholder="#"
           onkeydown="if(event.key==='Enter')jumpTo(this.value)">
    <span class="counter" id="counter">Loading...</span>
    <span id="badges"></span>
  </div>
  <div class="meta" id="meta"></div>
  <div class="problem-box">
    <h3>Problem</h3>
    <div class="text" id="problem-text"></div>
  </div>
  <div class="columns">
    <div class="column">
      <div class="col-header">
        <span>Original Trace</span>
        <span id="orig-size"></span>
      </div>
      <div class="col-body" id="original"></div>
    </div>
    <div class="column">
      <div class="col-header">
        <span>Shortened (Gemini distill)</span>
        <span id="short-size"></span>
      </div>
      <div class="col-body" id="shortened"></div>
    </div>
  </div>
  <div class="hint">
    <kbd>←</kbd> / <kbd>→</kbd> navigate &nbsp;|&nbsp;
    <kbd>j</kbd> / <kbd>k</kbd> navigate &nbsp;|&nbsp;
    Type number + <kbd>Enter</kbd> to jump
  </div>

<script>
let data = [];
let idx = 0;
const total = {{ total }};

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderTurn(turn) {
  const role = turn.role || 'unknown';
  const isCode = role === 'code' || role.includes('python');
  const content = escapeHtml(turn.content || '');
  return `<div class="turn">
    <div class="turn-role ${role.split(' ')[0]}">${escapeHtml(role)}</div>
    <div class="turn-content${isCode ? ' code-block' : ''}">${content}</div>
  </div>`;
}

function renderShortened(text) {
  // Basic markdown-ish rendering: code blocks, bold, boxed
  let html = escapeHtml(text);
  // Code blocks
  html = html.replace(/```python\n([\s\S]*?)```/g,
    '<pre style="background:#161b22;padding:12px;border-radius:6px;border:1px solid #30363d;overflow-x:auto"><code class="language-python">$1</code></pre>');
  html = html.replace(/```([\s\S]*?)```/g,
    '<pre style="background:#161b22;padding:12px;border-radius:6px;border:1px solid #30363d;overflow-x:auto"><code>$1</code></pre>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong style="color:#f0883e">$1</strong>');
  // Boxed answer
  html = html.replace(/\\boxed\{([^}]+)\}/g,
    '<span style="background:#238636;color:#fff;padding:2px 8px;border-radius:4px;font-weight:700">$1</span>');
  return html;
}

async function loadItem(i) {
  const resp = await fetch(`/api/item/${i}`);
  return resp.json();
}

async function show(i) {
  idx = i;
  const item = await loadItem(i);

  document.getElementById('counter').textContent = `${i+1} / ${total}`;
  document.getElementById('jump').value = i + 1;
  document.getElementById('btn-prev').disabled = (i === 0);
  document.getElementById('btn-next').disabled = (i >= total - 1);

  // Badges
  const s = item.shortening || {};
  let badges = '';
  if (s.status === 'ok') badges += '<span class="badge badge-ok">OK</span> ';
  else badges += '<span class="badge badge-err">ERROR</span> ';
  if (s.answer_verified) badges += '<span class="badge badge-verified">Answer Verified</span> ';
  else badges += '<span class="badge badge-unverified">Not Verified</span> ';
  if (item.topic) badges += `<span class="badge badge-topic">${escapeHtml(item.topic)}</span> `;
  document.getElementById('badges').innerHTML = badges;

  // Meta
  const ratio = s.compression_ratio != null ? (s.compression_ratio * 100).toFixed(1) + '%' : '?';
  document.getElementById('meta').innerHTML = `
    <span><span class="label">UID:</span> ${escapeHtml(item.uid)}</span>
    <span><span class="label">Source:</span> ${escapeHtml(item.source)}</span>
    <span><span class="label">Answer:</span> <strong>${escapeHtml(item.answer)}</strong></span>
    <span><span class="label">Pass Rate:</span> ${item.pass_rate != null ? (item.pass_rate*100).toFixed(1)+'%' : '?'}</span>
    <span><span class="label">Compression:</span> ${ratio}
      <div class="compression-bar" style="width:80px">
        <div class="compression-fill" style="width:${ratio}"></div>
      </div>
    </span>
    <span><span class="label">Scores:</span>
      Q=${item.scores?.quality ?? '?'} D=${item.scores?.difficulty ?? '?'} C=${item.scores?.combined ?? '?'}
    </span>
  `;

  // Problem
  document.getElementById('problem-text').textContent = item.problem_text;

  // Original turns
  const turns = item.original_turns || [];
  document.getElementById('original').innerHTML = turns.map(renderTurn).join('');
  document.getElementById('orig-size').textContent =
    `${(item.original_chars/1000).toFixed(1)}K chars`;

  // Shortened
  document.getElementById('shortened').innerHTML =
    `<div class="shortened-text">${renderShortened(item.shortened_text || '(empty)')}</div>`;
  document.getElementById('short-size').textContent =
    `${(item.shortened_chars/1000).toFixed(1)}K chars`;

  // Syntax highlight code blocks
  document.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
}

function navigate(delta) {
  const next = idx + delta;
  if (next >= 0 && next < total) show(next);
}

function jumpTo(val) {
  const n = parseInt(val) - 1;
  if (n >= 0 && n < total) show(n);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight' || e.key === 'j') navigate(1);
  if (e.key === 'ArrowLeft' || e.key === 'k') navigate(-1);
});

show(0);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE, total=len(DATA))


@app.route("/api/item/<int:idx>")
def api_item(idx):
    if idx < 0 or idx >= len(DATA):
        return jsonify({"error": "out of range"}), 404
    return jsonify(DATA[idx])


def main():
    global DATA
    print("Loading trace data...")
    DATA = load_data()
    print(f"Loaded {len(DATA)} trace pairs")
    if not DATA:
        print("No data found! Check file paths.")
        sys.exit(1)
    print("Starting viewer at http://localhost:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
