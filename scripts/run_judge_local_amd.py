#!/usr/bin/env python3
"""
AIMO3 Local Test Runner — AMD/llama.cpp version

Same 8+6+1 judge architecture as the Kaggle notebook, but adapted for:
- llama.cpp server (Vulkan backend for AMD GPUs)
- Standard OpenAI chat completions API (no openai_harmony)
- Code blocks parsed from model output (no harmony tool calling)

Usage:
    # Start llama-server first (in another terminal):
    ~/llama.cpp/build/bin/llama-server \
        -m ~/models/Qwen3-8B-Q4_K_M.gguf \
        --port 8081 -ngl 99 -c 32768 --parallel 4

    # Then run:
    .venv/bin/python scripts/run_judge_local_amd.py \
        --problems data/curated_union_problems.jsonl \
        --limit 3

    # Single problem:
    .venv/bin/python scripts/run_judge_local_amd.py \
        --problem "Find the sum of all positive integers n < 100 such that n^2 + n + 1 is prime."

    # With custom config:
    .venv/bin/python scripts/run_judge_local_amd.py \
        --problems data/curated_union_problems.jsonl \
        --phase1-attempts 4 --phase2-attempts 2 \
        --workers 2 --limit 3 --no-think
"""

import argparse
import json
import os
import sys
import re
import math
import time
import queue
import threading
import contextlib
import csv
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from jupyter_client import KernelManager


# ─── System Prompts ──────────────────────────────────────────────────────────

TIR_SYSTEM_PROMPT = (
    'You are a world-class International Mathematical Olympiad (IMO) competitor.\n'
    'Solve the problem step by step. When you need to compute something, write '
    'Python code in ```python blocks. After each code block, wait for the execution '
    'result before continuing.\n'
    'The environment has math, numpy, sympy, mpmath, itertools, collections available.\n'
    'The final answer must be a non-negative integer between 0 and 99999.\n'
    'You must place the final integer answer inside \\boxed{}.'
)

TEXT_ONLY_SYSTEM_PROMPT = (
    'You are a world-class International Mathematical Olympiad (IMO) competitor.\n'
    'Solve the problem step by step using mathematical reasoning.\n'
    'The final answer must be a non-negative integer between 0 and 99999.\n'
    'You must place the final integer answer inside \\boxed{}.'
)

JUDGE_SYSTEM_PROMPT = (
    'You are reviewing solutions to a math competition problem.\n'
    'The final answer must be a non-negative integer between 0 and 99999.\n'
    'Place each answer inside \\boxed{}.'
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_python_blocks(text):
    """Extract all ```python code blocks from text."""
    return re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)


# ─── Sandbox ──────────────────────────────────────────────────────────────────

class AIMO3Sandbox:
    _port_lock, _next_port = threading.Lock(), 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout):
        self._default_timeout, self._owns_kernel, self._client, self._km = timeout, False, None, None
        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env.update({'PYDEVD_DISABLE_FILE_VALIDATION': '1', 'PYDEVD_WARN_EVALUATION_TIMEOUT': '0',
                   'JUPYTER_PLATFORM_DIRS': '1', 'PYTHONWARNINGS': 'ignore', 'MPLBACKEND': 'Agg'})
        self._km = KernelManager()
        self._km.shell_port, self._km.iopub_port, self._km.stdin_port, self._km.hb_port, self._km.control_port = ports
        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True
        self.execute('import math, numpy, sympy, mpmath, itertools, collections\nmpmath.mp.dps = 64\n')

    def _format_error(self, tb):
        return ''.join(re.sub(r'\x1b\[[0-9;]*m', '', f) for f in tb
                      if 'File "' not in f or 'ipython-input' in f)

    def execute(self, code, timeout=None):
        effective_timeout = timeout or self._default_timeout
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout, stderr, start = [], [], time.time()
        while True:
            if time.time() - start > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id: continue
            mt, c = msg.get('msg_type'), msg.get('content', {})
            if mt == 'stream':
                (stdout if c.get('name') == 'stdout' else stderr).append(c.get('text', ''))
            elif mt == 'error':
                stderr.append(self._format_error(c.get('traceback', [])))
            elif mt in {'execute_result', 'display_data'}:
                if txt := c.get('data', {}).get('text/plain'):
                    stdout.append(txt if txt.endswith('\n') else f'{txt}\n')
            elif mt == 'status' and c.get('execution_state') == 'idle':
                break
        out, err = ''.join(stdout), ''.join(stderr)
        return f'{out.rstrip()}\n{err}' if err and out else (err or out or '[WARN] No output. Use print().')

    def close(self):
        with contextlib.suppress(Exception):
            if self._client: self._client.stop_channels()
        if self._owns_kernel and self._km:
            with contextlib.suppress(Exception): self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception): self._km.cleanup_resources()

    def reset(self):
        self.execute('%reset -f\nimport math, numpy, sympy, mpmath, itertools, collections\nmpmath.mp.dps = 64\n')

    def __del__(self):
        self.close()


# ─── Solver ──────────────────────────────────────────────────────────────────

class LocalSolverAMD:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(
            base_url=f'http://localhost:{args.port}/v1',
            api_key='sk-local',
            timeout=600
        )
        # Verify connection & detect model
        try:
            models = self.client.models.list()
            self.model_name = models.data[0].id if models.data else 'unknown'
        except Exception as e:
            raise RuntimeError(
                f'Cannot connect to llama.cpp server on port {args.port}: {e}\n'
                f'Start it first:\n'
                f'  ~/llama.cpp/build/bin/llama-server -m ~/models/Qwen3-8B-Q4_K_M.gguf '
                f'--port {args.port} -ngl 99 -c 32768 --parallel {args.workers}'
            )

        print(f'Connected to llama.cpp on port {args.port}')
        print(f'Model: {self.model_name}')

        # Jupyter kernels
        self._initialize_kernels()

        os.makedirs(args.output, exist_ok=True)
        self.problem_counter = 0

    def _initialize_kernels(self):
        n = self.args.workers
        print(f'Starting {n} Jupyter kernels...')
        self.sandbox_pool = queue.Queue()
        for i in range(n):
            try:
                self.sandbox_pool.put(AIMO3Sandbox(timeout=10))
                print(f'  Kernel {i+1}/{n} ready')
            except Exception as e:
                print(f'  Kernel {i+1} failed: {e}')
        print()

    # ── Answer scanning ──────────────────────────────────────────────────

    def _scan_for_answer(self, text):
        for pattern in [r'\\boxed\s*\{\s*([0-9,]+)\s*\}', r'final\s+answer\s+is\s*([0-9,]+)']:
            if matches := re.findall(pattern, text, re.IGNORECASE):
                try:
                    val = int(matches[-1].replace(',', ''))
                    if 0 <= val <= 99999: return val
                except ValueError: pass
        return None

    def _scan_all_answers(self, text):
        answers = []
        for m in re.finditer(r'\\boxed\s*\{\s*([0-9,]+)\s*\}', text):
            try:
                val = int(m.group(1).replace(',', ''))
                if 0 <= val <= 99999: answers.append(val)
            except ValueError: pass
        return answers

    # ── Entropy from chat completions logprobs ───────────────────────────

    def _compute_mean_entropy(self, logprobs_list):
        """logprobs_list: list of lists of ChatCompletionTokenLogprob objects."""
        if not logprobs_list: return float('inf')
        total, count = 0.0, 0
        for top_lps in logprobs_list:
            if top_lps:
                try:
                    probs = [math.exp(lp.logprob) for lp in top_lps]
                    ent = sum(-p * math.log2(p) for p in probs if p > 0)
                    total += ent
                    count += 1
                except (AttributeError, TypeError):
                    # Fallback for dict-style logprobs
                    pass
        return total / count if count else float('inf')

    # ── Single attempt ───────────────────────────────────────────────────

    def _process_attempt(self, problem, idx, stop_evt, deadline, use_tools=True):
        mode = 'tir' if use_tools else 'text_only'
        sys_prompt = TIR_SYSTEM_PROMPT if use_tools else TEXT_ONLY_SYSTEM_PROMPT

        if self.args.no_think:
            user_content = problem + '\n/no_think'
        else:
            user_content = problem

        if stop_evt.is_set() or time.time() > deadline:
            return {
                'Attempt': idx + 1, 'Answer': None, 'Python Calls': 0, 'Python Errors': 0,
                'Response Length': 0, 'Entropy': float('inf'), 'turns': [], 'Duration': 0.0,
                'model': self.model_name, 'mode': mode,
                'token_budget': self.args.context_length, 'tokens_remaining': self.args.context_length,
            }

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        sandbox = None
        turns = []
        py_calls, py_errs, total_toks = 0, 0, 0
        ans = None
        all_logprobs = []
        attempt_start = time.time()

        try:
            if use_tools:
                sandbox = self.sandbox_pool.get(timeout=5)

            for turn_num in range(self.args.max_turns):
                if stop_evt.is_set() or time.time() > deadline:
                    break

                # Call model via chat completions (streaming)
                create_kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    stream=True,
                    seed=42 + idx * 100 + turn_num,
                )
                # logprobs may not be supported by all llama.cpp builds
                if not self.args.no_logprobs:
                    create_kwargs['logprobs'] = True
                    create_kwargs['top_logprobs'] = 5

                stream = self.client.chat.completions.create(**create_kwargs)

                text_parts = []
                try:
                    for chunk in stream:
                        if stop_evt.is_set() or time.time() > deadline:
                            break
                        choice = chunk.choices[0] if chunk.choices else None
                        if not choice:
                            continue
                        delta = choice.delta
                        if delta and delta.content:
                            text_parts.append(delta.content)
                            total_toks += 1  # Approximate (1 chunk ≈ 1 token)
                        # Collect logprobs if available
                        if (not self.args.no_logprobs
                            and choice.logprobs
                            and choice.logprobs.content):
                            for tinfo in choice.logprobs.content:
                                if tinfo.top_logprobs:
                                    all_logprobs.append(tinfo.top_logprobs)
                except Exception as stream_err:
                    text_parts.append(f'\n[Stream error: {stream_err}]')

                text = ''.join(text_parts)
                if not text.strip():
                    break

                # Strip <think>...</think> blocks for answer scanning
                text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

                # Check for \boxed{} answer
                ans = self._scan_for_answer(text_no_think)
                if ans is not None:
                    turns.append({'type': 'text', 'content': text})
                    break

                # TIR: check for code blocks
                if use_tools:
                    code_blocks = extract_python_blocks(text)
                    if code_blocks:
                        turns.append({'type': 'text', 'content': text})
                        code = code_blocks[-1]
                        turns.append({'type': 'code', 'content': code})
                        py_calls += 1

                        output = sandbox.execute(code)
                        success = not any(x in output for x in ['[ERROR]', 'Traceback', 'Error:'])
                        if not success:
                            py_errs += 1
                        turns.append({'type': 'code_output', 'content': output, 'success': success})

                        # Feed back result
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user",
                                        "content": f"```output\n{output}\n```\n"
                                                    f"Continue solving. Place your final answer in \\boxed{{}}."})
                        continue

                # No code and no answer → stop
                turns.append({'type': 'text', 'content': text})
                break

        except Exception as e:
            turns.append({'type': 'error', 'content': str(e)})
            py_errs += 1
        finally:
            if sandbox:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return {
            'Attempt': idx + 1, 'Answer': ans,
            'Entropy': self._compute_mean_entropy(all_logprobs),
            'Response Length': total_toks,
            'Python Calls': py_calls, 'Python Errors': py_errs,
            'turns': turns,
            'Duration': round(time.time() - attempt_start, 2),
            'model': self.model_name, 'mode': mode,
            'token_budget': self.args.context_length,
            'tokens_remaining': self.args.context_length - total_toks,
        }

    # ── Phase execution ──────────────────────────────────────────────────

    def _run_phase(self, problem, n_attempts, n_workers, early_stop, stop_evt, deadline, idx_offset=0, text_only_count=0):
        """Run N attempts. Last text_only_count use text-only mode."""
        results, valid = [], []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for i in range(n_attempts):
                use_tools = i < (n_attempts - text_only_count)
                futures.append(ex.submit(
                    self._process_attempt, problem, idx_offset + i, stop_evt, deadline, use_tools))
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r['Answer'] is not None:
                        valid.append(r['Answer'])
                    results.append(r)
                    if early_stop and (cnts := Counter(valid).most_common(1)) and cnts[0][1] >= early_stop:
                        stop_evt.set()
                        for f in futures: f.cancel()
                        return results, True
                except Exception as exc:
                    print(f'  Future failed: {exc}')
        return results, False

    # ── Turns to text (for judge prompt) ─────────────────────────────────

    def _turns_to_text(self, turns, max_chars=3000):
        parts = []
        for t in turns:
            if t['type'] == 'text':
                # Strip think blocks for judge readability
                clean = re.sub(r'<think>.*?</think>', '', t['content'], flags=re.DOTALL).strip()
                if clean:
                    parts.append(clean)
            elif t['type'] == 'code':
                parts.append(f'\n```python\n{t["content"]}\n```\n')
            elif t['type'] == 'code_output':
                status = 'OK' if t.get('success', True) else 'ERROR'
                parts.append(f'[Output ({status})]: {t["content"][:500]}\n')
        full = '\n'.join(parts)
        if len(full) > max_chars:
            full = '...' + full[-max_chars:]
        return full

    # ── Judge ─────────────────────────────────────────────────────────────

    def _build_judge_prompt(self, problem, results):
        valid = [r for r in results if r['Answer'] is not None]
        if not valid: return None

        groups = defaultdict(list)
        for r in valid:
            groups[r['Answer']].append(r)

        total = len(valid)
        parts = [
            f"{total} solvers attempted the following math competition problem:\n",
            "PROBLEM:", problem,
            f"\nTheir answers, grouped by frequency:\n"
        ]
        for answer, members in sorted(groups.items(), key=lambda x: -len(x[1])):
            count = len(members)
            best = min(members, key=lambda r: r['Entropy'])
            parts.append(f"<ANSWER {answer} ({count}/{total} solvers)>")
            trace_text = self._turns_to_text(best.get('turns', []))
            if trace_text:
                parts.append(trace_text)
            parts.append(f"</ANSWER>\n")

        parts.append(
            "Study the solutions above. Assess the reasoning quality and determine which "
            "answers are most likely correct. Output two answers you believe are correct "
            "for this problem, most confident first. Place each inside \\boxed{}.\n\n"
            "You may choose from the candidate answers or derive your own if you believe "
            "the solutions contain errors. Explain your meta-conclusion briefly."
        )
        return '\n'.join(parts)

    def _run_judge(self, problem, results, deadline, judge_idx=0):
        """Run judge. Returns (list of up to 2 picked answers, judge_trace)."""
        judge_trace = {'judge_index': judge_idx}
        if time.time() > deadline:
            return [], judge_trace

        judge_prompt = self._build_judge_prompt(problem, results)
        if judge_prompt is None:
            return [], judge_trace

        judge_trace['prompt'] = judge_prompt

        if self.args.no_think:
            judge_user = judge_prompt + '\n/no_think'
        else:
            judge_user = judge_prompt

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_user},
        ]

        start_ts = time.time()
        try:
            create_kwargs = dict(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=self.args.max_tokens,
                stream=True,
                seed=1042 + judge_idx * 100,
            )
            stream = self.client.chat.completions.create(**create_kwargs)
            text_parts = []
            try:
                for chunk in stream:
                    if time.time() > deadline: break
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice and choice.delta and choice.delta.content:
                        text_parts.append(choice.delta.content)
            except Exception:
                pass

            text = ''.join(text_parts)
            text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            duration = time.time() - start_ts

            answers = self._scan_all_answers(text_no_think)
            seen, unique = set(), []
            for a in answers:
                if a not in seen:
                    seen.add(a)
                    unique.append(a)

            judge_trace['duration_seconds'] = round(duration, 2)
            judge_trace['response'] = text
            judge_trace['response_chars'] = len(text)
            judge_trace['picked_answers'] = unique[:2]

            print(f'  Judge: picked {unique[:2]} ({len(text)} chars, {duration:.1f}s)')
            return unique[:2], judge_trace
        except Exception as e:
            print(f'  Judge error: {e}')
            judge_trace['error'] = str(e)
            return [], judge_trace

    # ── Combined scoring ─────────────────────────────────────────────────

    def _score_with_judges(self, results, all_judge_picks):
        """Score using ONLY judge picks (1st=2pts, 2nd=1pt per judge). Judges have full authority."""
        scores = defaultdict(int)
        for judge_picks in all_judge_picks:
            if len(judge_picks) >= 1:
                scores[judge_picks[0]] += 2  # Judge 1st choice = 2 pts
            if len(judge_picks) >= 2:
                scores[judge_picks[1]] += 1  # Judge 2nd choice = 1 pt
        return dict(scores)

    # ── Entropy-gated consensus (fallback) ───────────────────────────────

    def _entropy_gated_fallback(self, results):
        valid = [r for r in results if r['Answer'] is not None]
        if not valid: return 0
        confident = [r for r in valid if r['Entropy'] < 5.0]
        if confident:
            votes = Counter(r['Answer'] for r in confident)
            candidates = {a: c for a, c in votes.items() if c >= 2}
            if candidates:
                scores = defaultdict(float)
                for r in confident:
                    if r['Answer'] in candidates:
                        scores[r['Answer']] += 1.0 / max(r['Entropy'], 0.1)
                return max(scores, key=scores.get)
        return Counter(r['Answer'] for r in valid).most_common(1)[0][0]

    # ── Main solve ────────────────────────────────────────────────────────

    def solve_problem(self, problem, ground_truth=None, timeout=900):
        self.problem_counter += 1
        start = time.time()
        deadline = start + timeout
        a = self.args

        print(f'\n{"="*70}')
        print(f'Problem {self.problem_counter}: {problem[:120]}...')
        print(f'Budget: {timeout}s | Phases: {a.phase1_attempts}+{a.phase2_attempts}+{a.num_judges}')

        user_input = problem + '\nYou have access to `math`, `numpy` and `sympy` to solve the problem.'
        stop_evt = threading.Event()

        trace = {
            'problem_index': self.problem_counter,
            'problem_text': problem,
            'ground_truth': ground_truth,
            'timestamp': datetime.now().isoformat(),
            'timeout_seconds': timeout,
            'config': {
                'model': self.model_name,
                'phase1_attempts': a.phase1_attempts,
                'phase2_attempts': a.phase2_attempts,
                'phase1_early_stop': a.early_stop,
                'workers': a.workers,
                'temperature': a.temperature,
                'no_think': a.no_think,
            },
            'phases': {},
        }

        # ── Phase 1: broad search ──
        p1_text_only = min(2, a.phase1_attempts)
        print(f'\n  Phase 1: {a.phase1_attempts} attempts '
              f'({a.phase1_attempts - p1_text_only} TIR + {p1_text_only} text-only)...')
        p1_start = time.time()
        results, early_stopped = self._run_phase(
            user_input, a.phase1_attempts, a.workers, a.early_stop, stop_evt, deadline,
            text_only_count=p1_text_only)

        for r in results:
            status = f'answer={r["Answer"]}' if r['Answer'] is not None else 'no answer'
            print(f'    Attempt {r["Attempt"]} [{r["mode"]}]: {status}, '
                  f'entropy={r["Entropy"]:.2f}, code={r["Python Calls"]}, '
                  f'tokens={r["Response Length"]}, {r["Duration"]}s')

        trace['phases']['phase1'] = {
            'duration_seconds': round(time.time() - p1_start, 2),
            'early_stopped': early_stopped,
            'attempts': [dict(r) for r in results],
        }

        valid = [r['Answer'] for r in results if r['Answer'] is not None]

        if early_stopped:
            winner = Counter(valid).most_common(1)[0][0]
            print(f'\n  EARLY STOP: {winner} (consensus >= {a.early_stop})')
            trace['final_answer'] = winner
            trace['selection_method'] = 'early_stop'
            trace['total_seconds'] = round(time.time() - start, 2)
            trace['is_correct'] = (winner == ground_truth) if ground_truth is not None else None
            self._save_trace(trace)
            return winner

        if not valid:
            print('  No valid answers, returning 0')
            trace['final_answer'] = 0
            trace['selection_method'] = 'no_valid'
            trace['total_seconds'] = round(time.time() - start, 2)
            self._save_trace(trace)
            return 0

        # ── Phase 2: deep search ──
        if time.time() < deadline and a.phase2_attempts > 0:
            p2_text_only = min(2, a.phase2_attempts)
            print(f'\n  Phase 2: {a.phase2_attempts} attempts '
                  f'({a.phase2_attempts - p2_text_only} TIR + {p2_text_only} text-only)...')
            p2_start = time.time()
            stop_evt.clear()
            more, _ = self._run_phase(
                user_input, a.phase2_attempts, a.workers, None, stop_evt, deadline,
                idx_offset=a.phase1_attempts, text_only_count=p2_text_only)
            results.extend(more)

            for r in more:
                status = f'answer={r["Answer"]}' if r['Answer'] is not None else 'no answer'
                print(f'    Attempt {r["Attempt"]} [{r["mode"]}]: {status}, '
                      f'entropy={r["Entropy"]:.2f}, tokens={r["Response Length"]}, {r["Duration"]}s')

            trace['phases']['phase2'] = {
                'duration_seconds': round(time.time() - p2_start, 2),
                'attempts': [dict(r) for r in more],
            }

        all_valid = [r for r in results if r['Answer'] is not None]
        all_votes = Counter(r['Answer'] for r in all_valid)
        print(f'\n  Candidates ({len(all_valid)} valid): {dict(all_votes.most_common())}')
        trace['candidates'] = {str(a): c for a, c in all_votes.most_common()}

        # ── Phase 3: Judge(s) ──
        n_judges = a.num_judges
        all_judge_picks = []
        judge_traces = []
        if time.time() < deadline:
            print(f'\n  Phase 3: {n_judges} Judge{"s" if n_judges > 1 else ""}...')
            for j_idx in range(n_judges):
                if time.time() > deadline:
                    break
                judge_picks, judge_trace = self._run_judge(user_input, results, deadline, judge_idx=j_idx)
                judge_traces.append(judge_trace)
                if judge_picks:
                    all_judge_picks.append(judge_picks)
            trace['phases']['judges'] = judge_traces

            if all_judge_picks:
                combined = self._score_with_judges(results, all_judge_picks)
                winner = max(combined, key=combined.get)
                trace['combined_scores'] = combined
                trace['final_answer'] = winner
                trace['selection_method'] = 'judge_combined'
                trace['total_seconds'] = round(time.time() - start, 2)
                trace['is_correct'] = (winner == ground_truth) if ground_truth is not None else None
                self._save_trace(trace)
                print(f'\n  Combined scores: {combined}')
                print(f'\n  JUDGE ANSWER: {winner}')
                return winner

        # Fallback
        fallback = self._entropy_gated_fallback(results)
        trace['final_answer'] = fallback
        trace['selection_method'] = 'fallback'
        trace['total_seconds'] = round(time.time() - start, 2)
        trace['is_correct'] = (fallback == ground_truth) if ground_truth is not None else None
        self._save_trace(trace)
        print(f'\n  FALLBACK ANSWER: {fallback}')
        return fallback

    def _save_trace(self, trace_data):
        pid = trace_data['problem_index']
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.args.output, f'{ts}_problem_{pid:03d}.json')
        with open(path, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)
        is_correct = trace_data.get('is_correct')
        mark = ' OK' if is_correct else (' WRONG' if is_correct is False else '')
        print(f'  Trace saved: {path}{mark}')

    def shutdown(self):
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                with contextlib.suppress(Exception):
                    self.sandbox_pool.get_nowait().close()


# ─── Problem loading ─────────────────────────────────────────────────────────

def load_problems(path):
    """Load problems from JSONL, CSV, or JSON."""
    problems = []
    ext = Path(path).suffix.lower()

    if ext == '.jsonl':
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get('problem', d.get('question', d.get('text', '')))
                answer = d.get('answer', d.get('ground_truth', d.get('expected_answer')))
                if answer is not None:
                    try: answer = int(answer)
                    except (ValueError, TypeError): answer = None
                if text: problems.append((text, answer))
    elif ext == '.csv':
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('problem', row.get('question', ''))
                answer = row.get('answer', row.get('ground_truth'))
                if answer is not None:
                    try: answer = int(answer)
                    except (ValueError, TypeError): answer = None
                if text: problems.append((text, answer))
    elif ext == '.json':
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for d in data:
                text = d.get('problem', d.get('question', ''))
                answer = d.get('answer')
                if answer is not None:
                    try: answer = int(answer)
                    except (ValueError, TypeError): answer = None
                if text: problems.append((text, answer))
    else:
        raise ValueError(f'Unsupported format: {ext}')

    return problems


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AIMO3 Local Runner (AMD/llama.cpp)')
    parser.add_argument('--problems', help='Path to problems file (.jsonl, .csv, .json)')
    parser.add_argument('--problem', help='Single problem text to solve')
    parser.add_argument('--output', default='traces/amd_run', help='Output directory')
    parser.add_argument('--limit', type=int, default=0, help='Max problems (0=all)')
    parser.add_argument('--offset', type=int, default=0, help='Skip first N')
    parser.add_argument('--timeout', type=int, default=600, help='Per-problem timeout (seconds)')

    parser.add_argument('--port', type=int, default=8081, help='llama.cpp server port')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers (match --parallel on server)')
    parser.add_argument('--phase1-attempts', type=int, default=4, help='Phase 1 attempts')
    parser.add_argument('--phase2-attempts', type=int, default=2, help='Phase 2 attempts')
    parser.add_argument('--early-stop', type=int, default=3, help='Early stop consensus')
    parser.add_argument('--max-turns', type=int, default=16, help='Max TIR turns per attempt')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max tokens per generation')
    parser.add_argument('--context-length', type=int, default=32768, help='Model context length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--no-think', action='store_true', help='Disable Qwen3 thinking mode')
    parser.add_argument('--num-judges', type=int, default=1, help='Number of judges (e.g., 2 for 7-7-2)')
    parser.add_argument('--no-logprobs', action='store_true', help='Skip logprobs (if server fails)')
    args = parser.parse_args()

    if args.problem:
        problems = [(args.problem, None)]
    elif args.problems:
        problems = load_problems(args.problems)
        if args.offset: problems = problems[args.offset:]
        if args.limit: problems = problems[:args.limit]
    else:
        parser.error('Provide --problems or --problem')

    print(f'AIMO3 Local Runner (AMD/llama.cpp)')
    print(f'Problems: {len(problems)}')
    print(f'Architecture: {args.phase1_attempts}+{args.phase2_attempts}+{args.num_judges}')
    print(f'Workers: {args.workers} | Timeout: {args.timeout}s')
    print(f'Output: {args.output}')
    print()

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    solver = LocalSolverAMD(args)

    try:
        correct, total, results_summary = 0, 0, []
        for problem_text, ground_truth in problems:
            answer = solver.solve_problem(problem_text, ground_truth, args.timeout)
            total += 1
            if ground_truth is not None:
                is_correct = (answer == ground_truth)
                if is_correct: correct += 1
                results_summary.append({
                    'index': total, 'answer': answer, 'ground_truth': ground_truth, 'correct': is_correct,
                })
                print(f'\n  Result: {answer} (expected {ground_truth}) {"OK" if is_correct else "WRONG"}')
                print(f'  Running accuracy: {correct}/{total} = {correct/total:.1%}')
            else:
                results_summary.append({'index': total, 'answer': answer})

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_problems': total,
            'correct': correct if any(gt is not None for _, gt in problems) else None,
            'accuracy': correct/total if total > 0 else None,
            'results': results_summary,
        }
        with open(os.path.join(args.output, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\n{"="*70}')
        print(f'Done. {total} problems solved.')
        if any(gt is not None for _, gt in problems):
            print(f'Accuracy: {correct}/{total} = {correct/total:.1%}')
        print(f'Traces: {args.output}/')

    finally:
        solver.shutdown()


if __name__ == '__main__':
    main()
