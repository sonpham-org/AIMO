#!/usr/bin/env python3
"""
AIMO3 Local Test Runner — 8+6+1 Judge Architecture

Runs the same solver as the Kaggle notebook but locally, saving comprehensive
JSON traces per problem (full reasoning, code executions, judge prompt/response).

Usage:
    # Run on a few problems from JSONL
    python scripts/run_judge_local.py \
        --model-path ~/models/gpt-oss-120b \
        --problems data/curated_union_problems.jsonl \
        --output traces/local_run \
        --limit 5

    # Single problem
    python scripts/run_judge_local.py \
        --model-path ~/models/gpt-oss-120b \
        --problem "Find the sum of all positive integers n such that n^2 + n + 1 divides n^4 + 2n^3."

    # With custom config
    python scripts/run_judge_local.py \
        --model-path ~/models/gpt-oss-120b \
        --problems data/curated_union_problems.jsonl \
        --phase1-attempts 4 --phase2-attempts 2 \
        --workers 4 --limit 3

Prerequisites:
    pip install vllm openai jupyter_client

    # Install openai_harmony from Kaggle wheels:
    export $(grep KAGGLE_API_TOKEN .env) && kaggle datasets download andreasbis/aimo-3-utils
    tar -xzf aimo-3-utils.zip
    pip install --find-links wheels openai_harmony

    # Download model (if needed):
    export $(grep KAGGLE_API_TOKEN .env)
    kaggle models instances versions download danielhanchen/gpt-oss-120b/transformers/default/1
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
import subprocess
import csv
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding, SystemContent, ReasoningEffort,
    ToolNamespaceConfig, Author, Message, Role, TextContent, Conversation
)
from jupyter_client import KernelManager


# ─── Template ────────────────────────────────────────────────────────────────

class AIMO3Template:
    def get_system_content(self, prompt, tool_cfg=None):
        sc = SystemContent.new().with_model_identity(prompt).with_reasoning_effort(
            reasoning_effort=ReasoningEffort.HIGH)
        if tool_cfg is not None:
            sc = sc.with_tools(tool_cfg)
        return sc

    def apply_chat_template(self, sys_prompt, usr_prompt, tool_cfg=None):
        return [Message.from_role_and_content(Role.SYSTEM, self.get_system_content(sys_prompt, tool_cfg)),
                Message.from_role_and_content(Role.USER, usr_prompt)]


# ─── Sandbox ─────────────────────────────────────────────────────────────────

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
        return f'{out.rstrip()}\n{err}' if err and out else (err or out or '[WARN] No output.')

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


# ─── Tool ────────────────────────────────────────────────────────────────────

class AIMO3Tool:
    def __init__(self, timeout, prompt, sandbox=None):
        self._local_jupyter_timeout, self._tool_prompt, self._jupyter_session = timeout, prompt, sandbox
        self._owns_session, self._execution_lock, self._init_lock = sandbox is None, threading.Lock(), threading.Lock()

    def _ensure_last_print(self, code):
        lines = code.strip().split('\n')
        if not lines: return code
        last = lines[-1].strip()
        if any(x in last for x in ['print', 'import']) or not last or last.startswith('#'): return code
        lines[-1] = 'print(' + last + ')'
        return '\n'.join(lines)

    @property
    def tool_config(self): return ToolNamespaceConfig(name='python', description=self._tool_prompt, tools=[])

    def process_sync_plus(self, message):
        final_script = self._ensure_last_print(message.content[0].text)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        msg = Message(author=Author(role=Role.TOOL, name='python'),
                     content=[TextContent(text=output)]).with_recipient('assistant')
        return [msg.with_channel(message.channel) if message.channel else msg]


# ─── Solver ──────────────────────────────────────────────────────────────────

class LocalSolver:
    def __init__(self, args):
        self.args = args
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        # Start vLLM server
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=f'http://0.0.0.0:{args.port}/v1', api_key='sk-local', timeout=960)
        self._wait_for_server()

        # Jupyter kernels
        self._initialize_kernels()

        # Output directory
        os.makedirs(args.output, exist_ok=True)
        self.problem_counter = 0

    def _start_server(self):
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', '42',
            '--model', self.args.model_path,
            '--served-model-name', 'gpt-oss',
            '--tensor-parallel-size', str(self.args.tp),
            '--max-num-seqs', '256',
            '--gpu-memory-utilization', str(self.args.gpu_util),
            '--host', '0.0.0.0', '--port', str(self.args.port),
            '--dtype', 'auto', '--kv-cache-dtype', 'fp8_e4m3',
            '--max-model-len', '65536',
            '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching',
        ]
        print(f'Starting vLLM: {" ".join(cmd[:6])}...')
        log = open(os.path.join(self.args.output, 'vllm_server.log'), 'w')
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        for i in range(300):
            if self.server_process.poll() is not None:
                raise RuntimeError('Server died. Check vllm_server.log')
            try:
                self.client.models.list()
                print(f'Server ready ({i}s)')
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError('Server timeout')

    def _initialize_kernels(self):
        n = self.args.workers
        print(f'Starting {n} Jupyter kernels...')
        self.sandbox_pool = queue.Queue()
        for i in range(n):
            try:
                self.sandbox_pool.put(AIMO3Sandbox(timeout=6))
                print(f'  Kernel {i+1}/{n} ready')
            except Exception as e:
                print(f'  Kernel {i+1} failed: {e}')
        print()

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

    def _compute_mean_entropy(self, logprobs):
        if not logprobs: return float('inf')
        total, count = 0.0, 0
        for top_lp in logprobs:
            if isinstance(top_lp, dict) and top_lp:
                ent = sum(-math.exp(lp)*math.log2(math.exp(lp)) for lp in top_lp.values() if math.exp(lp) > 0)
                total += ent
                count += 1
        return total/count if count else float('inf')

    def _process_attempt(self, problem, idx, stop_evt, deadline, use_tools=True):
        sys_prompt = ('You are a world-class International Mathematical Olympiad (IMO) competitor. '
                     'The final answer must be a non-negative integer between 0 and 99999. '
                     'You must place the final integer answer inside \\boxed{}.')
        tool_prompt = ('Use this tool to execute Python code. The environment is a stateful Jupyter notebook. '
                      'You must use print() to output results.')
        context_tokens = 65536
        mode = 'tir' if use_tools else 'text_only'

        if stop_evt.is_set() or time.time() > deadline:
            return {'Attempt': idx+1, 'Answer': None, 'Python Calls': 0, 'Python Errors': 0,
                   'Response Length': 0, 'Entropy': float('inf'), 'turns': [], 'Duration': 0.0,
                   'model': 'gpt-oss', 'mode': mode, 'token_budget': context_tokens, 'tokens_remaining': context_tokens}

        sandbox, local_tool = None, None
        py_calls, py_errs, total_toks, ans, logprobs = 0, 0, 0, None, []
        turns = []  # Structured turns: [{type, content, ...}]
        seed = int(math.pow(42 + idx, 2))
        attempt_start = time.time()
        try:
            if use_tools:
                sandbox = self.sandbox_pool.get(timeout=3)
                local_tool = AIMO3Tool(6, tool_prompt, sandbox)
                tool_cfg = local_tool.tool_config
            else:
                tool_cfg = None

            conv = Conversation.from_messages(self.template.apply_chat_template(
                sys_prompt, problem, tool_cfg))

            for _ in range(128):
                if stop_evt.is_set() or time.time() > deadline: break
                prompt_ids = self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
                max_toks = context_tokens - len(prompt_ids)
                if max_toks < 512: break

                stream = self.client.completions.create(
                    model='gpt-oss', temperature=1.0, logprobs=5, max_tokens=max_toks,
                    prompt=prompt_ids, seed=seed, stream=True,
                    extra_body={'min_p': 0.02, 'stop_token_ids': self.stop_token_ids, 'return_token_ids': True})
                try:
                    tok_buf, txt_chunks = [], []
                    for chunk in stream:
                        if stop_evt.is_set() or time.time() > deadline: break
                        if new_toks := chunk.choices[0].token_ids:
                            tok_buf.extend(new_toks)
                            total_toks += len(new_toks)
                            txt_chunks.append(chunk.choices[0].text)
                            if (clp := chunk.choices[0].logprobs) and clp.top_logprobs:
                                logprobs.extend(clp.top_logprobs)
                        if '}' in chunk.choices[0].text and (ans := self._scan_for_answer(
                            ''.join(txt_chunks[-32:]))):
                            break
                finally:
                    stream.close()

                turn_text = ''.join(txt_chunks)
                if ans or not tok_buf:
                    if turn_text:
                        turns.append({'type': 'text', 'content': turn_text})
                    break

                new_msgs = self.encoding.parse_messages_from_completion_tokens(tok_buf, Role.ASSISTANT)
                conv.messages.extend(new_msgs)
                last = new_msgs[-1]
                if last.channel == 'final':
                    turns.append({'type': 'text', 'content': last.content[0].text})
                    ans = self._scan_for_answer(last.content[0].text)
                    break
                if use_tools and last.recipient == 'python':
                    # Text before code (from this turn's raw output)
                    # The turn_text contains the full model output including code call markup
                    turns.append({'type': 'text', 'content': turn_text})
                    py_calls += 1
                    code_text = last.content[0].text
                    resp = local_tool.process_sync_plus(last)
                    tool_output = resp[0].content[0].text
                    success = not any(x in tool_output for x in ['[ERROR]', 'Traceback', 'Error:'])
                    if not success: py_errs += 1
                    turns.append({'type': 'code', 'content': code_text})
                    turns.append({'type': 'code_output', 'content': tool_output, 'success': success})
                    conv.messages.extend(resp)
                else:
                    turns.append({'type': 'text', 'content': turn_text})
        except Exception as e:
            py_errs += 1
            turns.append({'type': 'error', 'content': str(e)})
        finally:
            if sandbox:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        tokens_remaining = context_tokens - total_toks
        return {
            'Attempt': idx+1, 'Answer': ans, 'Entropy': self._compute_mean_entropy(logprobs),
            'Response Length': total_toks, 'Python Calls': py_calls, 'Python Errors': py_errs,
            'turns': turns, 'Duration': round(time.time() - attempt_start, 2),
            'model': 'gpt-oss', 'mode': mode,
            'token_budget': context_tokens, 'tokens_remaining': tokens_remaining,
        }

    def _run_phase(self, problem, n_attempts, n_workers, early_stop, stop_evt, deadline, idx_offset=0, text_only_count=0):
        """Run N attempts in parallel. Last text_only_count attempts use text-only mode (no tools)."""
        results, valid = [], []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for i in range(n_attempts):
                use_tools = i < (n_attempts - text_only_count)
                futures.append(ex.submit(self._process_attempt, problem, idx_offset + i, stop_evt, deadline, use_tools))
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

    def _turns_to_text(self, turns, max_chars=3000):
        """Convert structured turns to readable text for the judge, truncated."""
        parts = []
        for t in turns:
            if t['type'] == 'text':
                parts.append(t['content'])
            elif t['type'] == 'code':
                parts.append(f'\n```python\n{t["content"]}\n```\n')
            elif t['type'] == 'code_output':
                status = 'OK' if t.get('success', True) else 'ERROR'
                parts.append(f'[Output ({status})]: {t["content"][:500]}\n')
            elif t['type'] == 'error':
                parts.append(f'[Error: {t["content"]}]\n')
        full = ''.join(parts)
        if len(full) > max_chars:
            full = '...' + full[-max_chars:]
        return full

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

        judge_sys = ('You are reviewing solutions to a math competition problem. '
                    'The final answer must be a non-negative integer between 0 and 99999. '
                    'Place each answer inside \\boxed{}.')

        conv = Conversation.from_messages(self.template.apply_chat_template(judge_sys, judge_prompt))
        prompt_ids = self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
        max_toks = min(4096, 65536 - len(prompt_ids))

        judge_trace['prompt_tokens'] = len(prompt_ids)
        judge_trace['prompt'] = judge_prompt

        if max_toks < 256:
            judge_trace['error'] = 'prompt_too_long'
            return [], judge_trace

        start_ts = time.time()
        try:
            stream = self.client.completions.create(
                model='gpt-oss', temperature=0.7, max_tokens=max_toks,
                prompt=prompt_ids, seed=1042 + judge_idx * 100, stream=True,
                extra_body={'min_p': 0.02, 'stop_token_ids': self.stop_token_ids})
            text_chunks = []
            try:
                for chunk in stream:
                    if time.time() > deadline: break
                    text_chunks.append(chunk.choices[0].text)
            finally:
                stream.close()
            text = ''.join(text_chunks)

            answers = self._scan_all_answers(text)
            seen, unique = set(), []
            for a in answers:
                if a not in seen:
                    seen.add(a)
                    unique.append(a)

            judge_trace['duration_seconds'] = round(time.time() - start_ts, 2)
            judge_trace['response'] = text
            judge_trace['response_chars'] = len(text)
            judge_trace['picked_answers'] = unique[:2]

            print(f'  Judge {judge_idx+1}: picked {unique[:2]} ({len(text)} chars, {judge_trace["duration_seconds"]}s)')
            return unique[:2], judge_trace
        except Exception as e:
            judge_trace['error'] = str(e)
            return [], judge_trace

    def _score_with_judges(self, results, all_judge_picks):
        """Score using ONLY judge picks (1st=2pts, 2nd=1pt per judge). Judges have full authority."""
        scores = defaultdict(int)
        for judge_picks in all_judge_picks:
            if len(judge_picks) >= 1:
                scores[judge_picks[0]] += 2
            if len(judge_picks) >= 2:
                scores[judge_picks[1]] += 1
        return dict(scores)

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

    def solve_problem(self, problem, ground_truth=None, timeout=900):
        self.problem_counter += 1
        start = time.time()
        deadline = start + timeout
        a = self.args

        print(f'\n{"="*70}')
        print(f'Problem {self.problem_counter}: {problem[:120]}...')
        print(f'Budget: {timeout}s | Phases: {a.phase1_attempts}+{a.phase2_attempts}+{a.num_judges}')

        user_input = f'{problem} You have access to `math`, `numpy` and `sympy` to solve the problem.'
        stop_evt = threading.Event()

        trace = {
            'problem_index': self.problem_counter,
            'problem_text': problem,
            'ground_truth': ground_truth,
            'timestamp': datetime.now().isoformat(),
            'timeout_seconds': timeout,
            'config': {
                'model_path': a.model_path,
                'phase1_attempts': a.phase1_attempts,
                'phase2_attempts': a.phase2_attempts,
                'phase1_early_stop': a.early_stop,
                'workers': a.workers,
                'temperature': 1.0,
                'judge_temperature': 0.7,
            },
            'phases': {},
        }

        # ── Phase 1: 6 TIR + 2 text-only ──
        p1_text_only = min(2, a.phase1_attempts)
        print(f'\n  Phase 1: {a.phase1_attempts} attempts ({a.phase1_attempts - p1_text_only} TIR + {p1_text_only} text-only)...')
        p1_start = time.time()
        results, early_stopped = self._run_phase(
            user_input, a.phase1_attempts, a.workers, a.early_stop, stop_evt, deadline,
            text_only_count=p1_text_only)

        for r in results:
            status = f'answer={r["Answer"]}' if r['Answer'] is not None else 'no answer'
            print(f'    Attempt {r["Attempt"]} [{r.get("mode","tir")}]: {status}, entropy={r["Entropy"]:.2f}, '
                  f'code={r["Python Calls"]}, tokens={r["Response Length"]}, {r["Duration"]}s')

        trace['phases']['phase1'] = {
            'duration_seconds': round(time.time() - p1_start, 2),
            'early_stopped': early_stopped,
            'attempts': [dict(r) for r in results],  # Full data
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
            trace['final_answer'] = 0
            trace['selection_method'] = 'no_valid'
            trace['total_seconds'] = round(time.time() - start, 2)
            self._save_trace(trace)
            return 0

        # ── Phase 2 ──
        if time.time() < deadline and a.phase2_attempts > 0:
            p2_text_only = min(2, a.phase2_attempts)
            print(f'\n  Phase 2: {a.phase2_attempts} attempts ({a.phase2_attempts - p2_text_only} TIR + {p2_text_only} text-only)...')
            p2_start = time.time()
            stop_evt.clear()
            more, _ = self._run_phase(
                user_input, a.phase2_attempts, a.workers, None, stop_evt, deadline,
                idx_offset=a.phase1_attempts, text_only_count=p2_text_only)
            results.extend(more)

            for r in more:
                status = f'answer={r["Answer"]}' if r['Answer'] is not None else 'no answer'
                print(f'    Attempt {r["Attempt"]} [{r.get("mode","tir")}]: {status}, entropy={r["Entropy"]:.2f}, '
                      f'tokens={r["Response Length"]}, {r["Duration"]}s')

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
        mark = ' ✓' if is_correct else (' ✗' if is_correct is False else '')
        print(f'  Trace saved: {path}{mark}')

    def shutdown(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                with contextlib.suppress(Exception):
                    self.sandbox_pool.get_nowait().close()


# ─── Problem loading ─────────────────────────────────────────────────────────

def load_problems(path):
    """Load problems from JSONL or CSV. Returns list of (problem_text, answer_or_None)."""
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
                if text:
                    problems.append((text, answer))
    elif ext == '.csv':
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('problem', row.get('question', ''))
                answer = row.get('answer', row.get('ground_truth'))
                if answer is not None:
                    try: answer = int(answer)
                    except (ValueError, TypeError): answer = None
                if text:
                    problems.append((text, answer))
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
                if text:
                    problems.append((text, answer))
    else:
        raise ValueError(f'Unsupported format: {ext}. Use .jsonl, .csv, or .json')

    return problems


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AIMO3 Local Test Runner (8+6+1 Judge)')
    parser.add_argument('--model-path', required=True, help='Path to model (e.g., ~/models/gpt-oss-120b)')
    parser.add_argument('--problems', help='Path to problems file (.jsonl, .csv, .json)')
    parser.add_argument('--problem', help='Single problem text to solve')
    parser.add_argument('--output', default='traces/local_run', help='Output directory for traces')
    parser.add_argument('--limit', type=int, default=0, help='Max problems to solve (0=all)')
    parser.add_argument('--offset', type=int, default=0, help='Skip first N problems')
    parser.add_argument('--timeout', type=int, default=900, help='Timeout per problem (seconds)')
    parser.add_argument('--phase1-attempts', type=int, default=8, help='Phase 1 attempts')
    parser.add_argument('--phase2-attempts', type=int, default=6, help='Phase 2 attempts')
    parser.add_argument('--early-stop', type=int, default=4, help='Early stop consensus threshold')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--port', type=int, default=8000, help='vLLM server port')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--gpu-util', type=float, default=0.96, help='GPU memory utilization')
    parser.add_argument('--num-judges', type=int, default=1, help='Number of judges (e.g., 2 for 7-7-2)')
    args = parser.parse_args()

    # Load problems
    if args.problem:
        problems = [(args.problem, None)]
    elif args.problems:
        problems = load_problems(args.problems)
        if args.offset:
            problems = problems[args.offset:]
        if args.limit:
            problems = problems[:args.limit]
    else:
        parser.error('Provide --problems or --problem')

    print(f'AIMO3 Local Test Runner')
    print(f'Model: {args.model_path}')
    print(f'Problems: {len(problems)}')
    print(f'Architecture: {args.phase1_attempts}+{args.phase2_attempts}+{args.num_judges}')
    print(f'Output: {args.output}')
    print()

    # Save run config
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    solver = LocalSolver(args)

    try:
        correct, total, results_summary = 0, 0, []
        for problem_text, ground_truth in problems:
            answer = solver.solve_problem(problem_text, ground_truth, args.timeout)
            total += 1
            if ground_truth is not None:
                is_correct = (answer == ground_truth)
                if is_correct: correct += 1
                results_summary.append({
                    'index': total, 'answer': answer, 'ground_truth': ground_truth,
                    'correct': is_correct,
                })
                print(f'\n  Result: {answer} (expected {ground_truth}) {"✓" if is_correct else "✗"}')
                print(f'  Running accuracy: {correct}/{total} = {correct/total:.1%}')
            else:
                results_summary.append({'index': total, 'answer': answer})

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_problems': total,
            'correct': correct if any(r.get('ground_truth') is not None for _, gt in problems if gt is not None) else None,
            'accuracy': correct/total if total > 0 and correct > 0 else None,
            'results': results_summary,
        }
        summary_path = os.path.join(args.output, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\n{"="*70}')
        print(f'Done. {total} problems solved.')
        if correct > 0:
            print(f'Accuracy: {correct}/{total} = {correct/total:.1%}')
        print(f'Traces: {args.output}/')

    finally:
        solver.shutdown()


if __name__ == '__main__':
    main()
