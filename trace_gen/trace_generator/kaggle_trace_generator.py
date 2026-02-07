#!/usr/bin/env python3
"""
AIMO3 Trace Generator for Kaggle H100
=====================================

PURPOSE:
    Generate rich TIR traces for all 53 AIMO3 reference problems using gpt-oss-120b
    on Kaggle H100 GPUs. These traces contain everything needed to experiment with
    different answer selection strategies OFFLINE (without re-running inference).

OUTPUT:
    /kaggle/working/traces/
        problem_{id}.json  -- One file per problem with all attempt data
        summary.json       -- Overall stats and quick reference
        config.json        -- Generation parameters for reproducibility

TRACE FORMAT (per problem):
    {
        "problem_id": "abc123",
        "problem_text": "...",
        "ground_truth": 42,
        "wall_time_s": 300.5,
        "attempts": [
            {
                "attempt_idx": 0,
                "answer": 42,
                "answer_source": "boxed" | "code_fallback" | null,
                "entropy": 0.523,
                "prompt_type": "reasoning" | "code_first" | "case_analysis",
                "turns_used": 3,
                "n_python_calls": 2,
                "n_python_errors": 0,
                "total_response_tokens": 1500,
                "code_executions": [
                    {"turn": 0, "code": "...", "output": "...", "is_error": false}
                ],
                "wall_time_s": 45.2,
                "seed": 1849,
                "logprobs_summary": {"mean": 0.5, "min": 0.1, "max": 1.2, "count": 100}
            },
            ...
        ],
        "default_answer": 42,
        "default_method": "majority_vote",
        "default_votes": {"42": 8, "0": 2}
    }

USAGE:
    1. Create a new Kaggle notebook
    2. Add data sources:
       - danielhanchen/gpt-oss-120b (model)
       - andreasbis/aimo-3-utils (vLLM wheels)
       - sonphamorg/aimo3-reference-problems (reference CSV, or paste inline)
    3. Copy this code into cells
    4. Run with H100 GPU, 9h timeout
    5. Download traces/ folder when done

COMPATIBLE WITH:
    scripts/replay_selection.py -- Run 138+ selection strategies on these traces
"""

# =============================================================================
# Cell 1: Uninstall conflicting packages
# =============================================================================
# %pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'

# =============================================================================
# Cell 2: Imports (standard library)
# =============================================================================
import warnings; warnings.simplefilter('ignore')
import os, sys, subprocess, gc, re, math, time, queue, threading, contextlib, json
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Optional, Dict, List, Any

# =============================================================================
# Cell 3: Install dependencies from aimo-3-utils wheels
# =============================================================================
def set_env(archive, tmp):
    if not os.path.exists(tmp):
        os.makedirs(tmp, exist_ok=True)
        subprocess.run(['tar', '-xzf', archive, '-C', tmp], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', f'{tmp}/wheels',
                    'unsloth', 'trl', 'vllm', 'openai_harmony'], check=True)

# Uncomment on Kaggle:
# set_env('/kaggle/input/aimo-3-utils/wheels.tar.gz', '/kaggle/tmp/setup')

# =============================================================================
# Cell 4: Environment variables
# =============================================================================
for k, v in [('TRANSFORMERS_NO_TF', '1'), ('TRANSFORMERS_NO_FLAX', '1'), ('CUDA_VISIBLE_DEVICES', '0'),
             ('TOKENIZERS_PARALLELISM', 'false'), ('TRITON_PTXAS_PATH', '/usr/local/cuda/bin/ptxas'),
             ('TIKTOKEN_ENCODINGS_BASE', '/kaggle/tmp/setup/tiktoken_encodings')]:
    os.environ[k] = v

# =============================================================================
# Cell 5: Imports (after pip install)
# =============================================================================
import pandas as pd
from jupyter_client import KernelManager
from openai import OpenAI
from openai_harmony import (HarmonyEncodingName, load_harmony_encoding, SystemContent, ReasoningEffort,
                             ToolNamespaceConfig, Author, Message, Role, TextContent, Conversation)
from transformers import set_seed

# =============================================================================
# Cell 6: Configuration
# =============================================================================
class CFG:
    """Configuration for trace generation."""

    # --- Prompts (matching improv1_entropy_plus) ---
    system_prompt_reasoning = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    system_prompt_code_first = (
        'You are a world-class IMO competitor who excels at computational verification. '
        'Write Python code to explore and verify your reasoning whenever possible. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    system_prompt_cases = (
        'You are a world-class IMO competitor. Break the problem into cases, '
        'solve each case carefully, and use Python code to check your work. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )

    # 12 prompts: 8 reasoning + 2 code-first + 2 case-analysis
    prompt_configs = (
        [('reasoning', system_prompt_reasoning)] * 8 +
        [('code_first', system_prompt_code_first)] * 2 +
        [('case_analysis', system_prompt_cases)] * 2
    )

    tool_prompt = ('Use this tool to execute Python code. The environment is a stateful Jupyter notebook. '
                  'You must use print() to output results.')
    preference_prompt = 'You have access to `math`, `numpy` and `sympy` to solve the problem.'

    # Model
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    # Timing
    problem_timeout = 600  # 10 minutes per problem (conservative for trace generation)
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3

    # Generation
    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256

    # Attempts
    n_samples = 12
    max_turns = 128  # Same as competition notebook
    workers = 16
    early_stop_threshold = 6  # Stop if 6+ agree (just for speed, we still save all attempts)

    # Sampling
    gpu_memory_utilization = 0.96
    temperature = 1.0
    min_p = 0.02
    seed = 42

    # Output
    output_dir = '/kaggle/working/traces'

    # Code fallback entropy penalty
    code_fallback_entropy = 8.0


# =============================================================================
# Cell 7: Reference problems (inline fallback)
# =============================================================================
REFERENCE_PROBLEMS = """id,problem,answer
0e644e,"Let $ABC$ be an acute-angled triangle with integer side lengths...",336
26de63,"Define a function $f \\colon \\mathbb{Z}_{\\geq 1} \\to...",54321
"""
# TODO: Replace with actual problems or load from Kaggle dataset

def load_reference_problems():
    """Load reference problems from Kaggle dataset or inline."""
    kaggle_path = '/kaggle/input/aimo3-reference-problems/reference.csv'
    local_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv'

    for path in [kaggle_path, local_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} problems from {path}")
            return df

    # Fallback to inline (for testing)
    from io import StringIO
    df = pd.read_csv(StringIO(REFERENCE_PROBLEMS))
    print(f"WARNING: Using inline fallback ({len(df)} problems)")
    return df


# =============================================================================
# Cell 8: Sandbox (Jupyter kernel for code execution)
# =============================================================================
class AIMO3Sandbox:
    """Stateful Jupyter kernel for Python code execution."""
    _port_lock, _next_port = threading.Lock(), 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env.update({
            'PYDEVD_DISABLE_FILE_VALIDATION': '1',
            'PYDEVD_WARN_EVALUATION_TIMEOUT': '0',
            'JUPYTER_PLATFORM_DIRS': '1',
            'PYTHONWARNINGS': 'ignore',
            'MPLBACKEND': 'Agg'
        })

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

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

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
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        self.execute('%reset -f\nimport math, numpy, sympy, mpmath, itertools, collections\nmpmath.mp.dps = 64\n')

    def __del__(self):
        self.close()


# =============================================================================
# Cell 9: Tool wrapper
# =============================================================================
class AIMO3Tool:
    """Wrapper for code execution tool."""
    def __init__(self, timeout, prompt, sandbox=None):
        self._local_jupyter_timeout = timeout
        self._tool_prompt = prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code):
        lines = code.strip().split('\n')
        if not lines:
            return code
        last = lines[-1].strip()
        if any(x in last for x in ['print', 'import']) or not last or last.startswith('#'):
            return code
        lines[-1] = 'print(' + last + ')'
        return '\n'.join(lines)

    @property
    def instruction(self):
        return self._tool_prompt

    @property
    def tool_config(self):
        return ToolNamespaceConfig(name='python', description=self.instruction, tools=[])

    def _make_response(self, output, channel=None):
        msg = Message(
            author=Author(role=Role.TOOL, name='python'),
            content=[TextContent(text=output)]
        ).with_recipient('assistant')
        return msg.with_channel(channel) if channel else msg

    def process_sync_plus(self, message):
        self._ensure_session()
        final_script = self._ensure_last_print(message.content[0].text)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=message.channel)]


# =============================================================================
# Cell 10: Template
# =============================================================================
class AIMO3Template:
    def get_system_content(self, prompt, tool_cfg):
        return SystemContent.new().with_model_identity(prompt).with_reasoning_effort(
            reasoning_effort=ReasoningEffort.HIGH).with_tools(tool_cfg)

    def apply_chat_template(self, sys_prompt, usr_prompt, tool_cfg):
        return [
            Message.from_role_and_content(Role.SYSTEM, self.get_system_content(sys_prompt, tool_cfg)),
            Message.from_role_and_content(Role.USER, usr_prompt)
        ]


# =============================================================================
# Cell 11: Trace Generator
# =============================================================================
class TraceGenerator:
    """Generate rich traces for offline selection strategy experiments."""

    def __init__(self, cfg: CFG, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        # Initialize
        self._preload_model_weights()
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)
        self._wait_for_server()
        self._initialize_kernels()

        # Output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # Stats
        self.start_time = time.time()
        self.problems_processed = 0
        self.total_correct = 0

    def _preload_model_weights(self):
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start = time.time()
        files, total = [], 0
        for root, _, fnames in os.walk(self.cfg.model_path):
            for fn in fnames:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    files.append(fp)
                    total += os.path.getsize(fp)
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            list(ex.map(lambda p: open(p, 'rb').read(), files))
        print(f'Processed {len(files)} files ({total/1e9:.2f} GB) in {time.time()-start:.2f} seconds.\n')

    def _start_server(self):
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', '1',
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0',
            '--port', str(self.port),
            '--dtype', self.cfg.dtype,
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling',
            '--disable-log-stats',
            '--enable-prefix-caching'
        ]
        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        start = time.time()
        for _ in range(self.cfg.server_timeout):
            if (rc := self.server_process.poll()) is not None:
                self.log_file.flush()
                raise RuntimeError(f'Server died with code {rc}. Logs:\n{open("vllm_server.log").read()}\n')
            try:
                self.client.models.list()
                print(f'Server ready ({time.time()-start:.2f}s)\n')
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError('Server failed to start (timeout)')

    def _initialize_kernels(self):
        print(f'Initializing {self.cfg.workers} Jupyter kernels...')
        start = time.time()
        self.sandbox_pool = queue.Queue()
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            for future in as_completed([
                ex.submit(lambda: AIMO3Sandbox(timeout=self.cfg.jupyter_timeout))
                for _ in range(self.cfg.workers)
            ]):
                self.sandbox_pool.put(future.result())
        print(f'Kernels ready ({time.time()-start:.2f}s)\n')

    def _scan_for_answer(self, text: str) -> Optional[int]:
        """Extract answer from \\boxed{}."""
        for pattern in [r'\\boxed\s*\{\s*([0-9,]+)\s*\}', r'final\s+answer\s+is\s*([0-9,]+)']:
            if matches := re.findall(pattern, text, re.IGNORECASE):
                try:
                    val = int(matches[-1].replace(',', ''))
                    if 0 <= val <= 99999:
                        return val
                except ValueError:
                    pass
        return None

    def _scan_code_output_for_answer(self, text: str) -> Optional[int]:
        """Extract last integer from code output as fallback."""
        candidates = []
        for num_str in re.findall(r'\b(\d{1,5})\b', text):
            val = int(num_str)
            if 0 <= val <= 99999:
                candidates.append(val)
        return candidates[-1] if candidates else None

    def _compute_entropy(self, logprobs: List[Dict]) -> float:
        """Shannon entropy from top logprobs."""
        if not logprobs:
            return float('inf')
        total, count = 0.0, 0
        for top_lp in logprobs:
            if isinstance(top_lp, dict) and top_lp:
                ent = sum(-math.exp(lp) * math.log2(math.exp(lp))
                         for lp in top_lp.values() if math.exp(lp) > 0)
                total += ent
                count += 1
        return total / count if count else float('inf')

    def _summarize_logprobs(self, logprobs: List[Dict]) -> Dict:
        """Compute summary stats for logprobs."""
        if not logprobs:
            return {'mean': float('inf'), 'min': float('inf'), 'max': float('inf'), 'count': 0}

        entropies = []
        for top_lp in logprobs:
            if isinstance(top_lp, dict) and top_lp:
                ent = sum(-math.exp(lp) * math.log2(math.exp(lp))
                         for lp in top_lp.values() if math.exp(lp) > 0)
                entropies.append(ent)

        if not entropies:
            return {'mean': float('inf'), 'min': float('inf'), 'max': float('inf'), 'count': 0}

        return {
            'mean': sum(entropies) / len(entropies),
            'min': min(entropies),
            'max': max(entropies),
            'count': len(entropies)
        }

    def _process_single_attempt(
        self,
        problem_text: str,
        prompt_type: str,
        system_prompt: str,
        attempt_idx: int,
        stop_event: threading.Event,
        deadline: float
    ) -> Dict[str, Any]:
        """Run one TIR attempt, return rich trace data."""

        # Early exit check
        if stop_event.is_set() or time.time() > deadline:
            return {
                'attempt_idx': attempt_idx,
                'answer': None,
                'answer_source': 'skipped',
                'entropy': float('inf'),
                'prompt_type': prompt_type,
                'turns_used': 0,
                'n_python_calls': 0,
                'n_python_errors': 0,
                'total_response_tokens': 0,
                'code_executions': [],
                'wall_time_s': 0.0,
                'seed': 0,
                'logprobs_summary': {'mean': float('inf'), 'min': float('inf'), 'max': float('inf'), 'count': 0},
                'error': 'skipped'
            }

        start_time = time.time()
        seed = int(math.pow(self.cfg.seed + attempt_idx, 2))

        # Initialize
        local_tool = None
        sandbox = None
        n_python_calls = 0
        n_python_errors = 0
        total_tokens = 0
        answer = None
        answer_source = None
        all_logprobs = []
        code_executions = []
        last_code_output = None
        turns_used = 0
        error_msg = None

        try:
            # Get sandbox from pool
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(self.cfg.jupyter_timeout, self.cfg.tool_prompt, sandbox)

            # Build conversation
            user_input = f'{problem_text} {self.cfg.preference_prompt}'
            conv = Conversation.from_messages(
                self.template.apply_chat_template(system_prompt, user_input, local_tool.tool_config)
            )

            for turn in range(self.cfg.max_turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                turns_used = turn + 1

                # Tokenize conversation
                prompt_ids = self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
                max_toks = self.cfg.context_tokens - len(prompt_ids)
                if max_toks < self.cfg.buffer_tokens:
                    break

                # Stream completion
                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    logprobs=self.cfg.top_logprobs,
                    max_tokens=max_toks,
                    prompt=prompt_ids,
                    seed=seed,
                    stream=True,
                    extra_body={
                        'min_p': self.cfg.min_p,
                        'stop_token_ids': self.stop_token_ids,
                        'return_token_ids': True
                    }
                )

                try:
                    tok_buf = []
                    txt_chunks = []

                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break

                        if new_toks := chunk.choices[0].token_ids:
                            tok_buf.extend(new_toks)
                            total_tokens += len(new_toks)
                            txt_chunks.append(chunk.choices[0].text)

                            # Collect logprobs
                            if (clp := chunk.choices[0].logprobs) and clp.top_logprobs:
                                for tlp in clp.top_logprobs:
                                    if tlp:
                                        all_logprobs.append({t.token: t.logprob for t in tlp})

                        # Check for answer mid-stream
                        if '}' in chunk.choices[0].text:
                            recent = ''.join(txt_chunks[-self.cfg.search_tokens:])
                            if (ans := self._scan_for_answer(recent)):
                                answer = ans
                                answer_source = 'boxed'
                                break
                finally:
                    stream.close()

                if answer is not None or not tok_buf:
                    break

                # Parse messages
                new_msgs = self.encoding.parse_messages_from_completion_tokens(tok_buf, Role.ASSISTANT)
                conv.messages.extend(new_msgs)

                last = new_msgs[-1]
                if last.channel == 'final':
                    answer = self._scan_for_answer(last.content[0].text)
                    if answer is not None:
                        answer_source = 'boxed'
                    break

                # Handle tool calls
                if last.recipient == 'python':
                    n_python_calls += 1
                    code = last.content[0].text
                    resp = local_tool.process_sync_plus(last)
                    tool_output = resp[0].content[0].text

                    is_error = any(x in tool_output for x in ['[ERROR]', 'Traceback', 'Error:'])
                    if is_error:
                        n_python_errors += 1
                    else:
                        last_code_output = tool_output

                    code_executions.append({
                        'turn': turn,
                        'code': code,
                        'output': tool_output[:2000],  # Truncate long outputs
                        'is_error': is_error
                    })

                    conv.messages.extend(resp)

        except Exception as e:
            error_msg = str(e)
            n_python_errors += 1

        finally:
            if sandbox:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        # Code-output fallback
        if answer is None and last_code_output is not None:
            answer = self._scan_code_output_for_answer(last_code_output)
            if answer is not None:
                answer_source = 'code_fallback'

        # Compute entropy
        entropy = self._compute_entropy(all_logprobs)
        if answer_source == 'code_fallback':
            entropy = max(entropy, self.cfg.code_fallback_entropy)

        wall_time = time.time() - start_time

        return {
            'attempt_idx': attempt_idx,
            'answer': answer,
            'answer_source': answer_source,
            'entropy': entropy,
            'prompt_type': prompt_type,
            'turns_used': turns_used,
            'n_python_calls': n_python_calls,
            'n_python_errors': n_python_errors,
            'total_response_tokens': total_tokens,
            'code_executions': code_executions,
            'wall_time_s': round(wall_time, 2),
            'seed': seed,
            'logprobs_summary': self._summarize_logprobs(all_logprobs),
            'error': error_msg
        }

    def generate_traces_for_problem(
        self,
        problem_id: str,
        problem_text: str,
        ground_truth: Optional[int]
    ) -> Dict[str, Any]:
        """Generate all traces for a single problem."""

        print(f'\n{"="*60}')
        print(f'Problem: {problem_id}')
        if ground_truth is not None:
            print(f'Ground truth: {ground_truth}')
        print(f'{"="*60}')

        start_time = time.time()
        deadline = start_time + self.cfg.problem_timeout
        stop_event = threading.Event()

        attempts = []
        valid_answers = []

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = []
            for i in range(self.cfg.n_samples):
                prompt_type, system_prompt = self.cfg.prompt_configs[i % len(self.cfg.prompt_configs)]
                futures.append(executor.submit(
                    self._process_single_attempt,
                    problem_text,
                    prompt_type,
                    system_prompt,
                    i,
                    stop_event,
                    deadline
                ))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    attempts.append(result)

                    if result['answer'] is not None:
                        valid_answers.append(result['answer'])

                    # Log progress
                    ans = result['answer']
                    ent = result['entropy']
                    src = result['answer_source'] or 'none'
                    err = result.get('error', '')
                    print(f"  Sample {result['attempt_idx']+1}/{self.cfg.n_samples}: "
                          f"answer={ans} (src={src}, entropy={ent:.3f})"
                          f"{' ERROR=' + err if err else ''}")

                    # Early stop if enough agreement (but keep all traces)
                    if valid_answers:
                        counts = Counter(valid_answers).most_common(1)
                        if counts[0][1] >= self.cfg.early_stop_threshold:
                            stop_event.set()
                            # Don't break - let remaining futures finish/cancel

                except Exception as e:
                    print(f"  Future failed: {e}")

        wall_time = time.time() - start_time

        # Default answer: majority vote
        if valid_answers:
            counter = Counter(valid_answers)
            default_answer = counter.most_common(1)[0][0]
            default_method = 'majority_vote'
        else:
            default_answer = 0
            default_method = 'no_valid_answers'

        is_correct = (default_answer == ground_truth) if ground_truth is not None else None
        tag = 'CORRECT' if is_correct else ('WRONG' if is_correct is False else '?')
        print(f'\n  >> [{tag}] Default={default_answer}, Expected={ground_truth}, Time={wall_time:.1f}s')
        print(f'  >> Votes: {dict(Counter(valid_answers))}')

        # Sort attempts by index
        attempts.sort(key=lambda a: a['attempt_idx'])

        return {
            'problem_id': problem_id,
            'problem_text': problem_text,
            'ground_truth': ground_truth,
            'wall_time_s': round(wall_time, 2),
            'attempts': attempts,
            'default_answer': default_answer,
            'default_method': default_method,
            'default_votes': dict(Counter(valid_answers))
        }

    def generate_all_traces(self, df: pd.DataFrame):
        """Generate traces for all problems and save to disk."""

        print(f'\n{"#"*60}')
        print(f'# TRACE GENERATION START')
        print(f'# Problems: {len(df)}')
        print(f'# Samples per problem: {self.cfg.n_samples}')
        print(f'# Max turns: {self.cfg.max_turns}')
        print(f'# Output: {self.cfg.output_dir}')
        print(f'{"#"*60}\n')

        # Save config
        config = {
            'model': self.cfg.served_model_name,
            'model_path': self.cfg.model_path,
            'n_samples': self.cfg.n_samples,
            'max_turns': self.cfg.max_turns,
            'temperature': self.cfg.temperature,
            'min_p': self.cfg.min_p,
            'seed': self.cfg.seed,
            'top_logprobs': self.cfg.top_logprobs,
            'problem_timeout': self.cfg.problem_timeout,
            'prompt_mix': {
                'reasoning': sum(1 for p, _ in self.cfg.prompt_configs if p == 'reasoning'),
                'code_first': sum(1 for p, _ in self.cfg.prompt_configs if p == 'code_first'),
                'case_analysis': sum(1 for p, _ in self.cfg.prompt_configs if p == 'case_analysis'),
            },
            'n_problems': len(df),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(self.cfg.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        all_results = []

        for idx, row in df.iterrows():
            problem_id = str(row['id'])
            problem_text = row['problem'] if 'problem' in row else row.get('question', '')
            ground_truth = int(row['answer']) if 'answer' in row and pd.notna(row['answer']) else None

            # Generate traces
            trace = self.generate_traces_for_problem(problem_id, problem_text, ground_truth)

            # Save per-problem trace
            trace_path = os.path.join(self.cfg.output_dir, f'problem_{problem_id}.json')
            with open(trace_path, 'w') as f:
                json.dump(trace, f, indent=2, default=self._json_safe)

            # Track results
            is_correct = trace['default_answer'] == ground_truth if ground_truth is not None else None
            all_results.append({
                'problem_id': problem_id,
                'ground_truth': ground_truth,
                'default_answer': trace['default_answer'],
                'correct': is_correct,
                'wall_time_s': trace['wall_time_s'],
                'n_attempts': len(trace['attempts']),
                'n_valid_answers': len([a for a in trace['attempts'] if a['answer'] is not None])
            })

            if is_correct:
                self.total_correct += 1
            self.problems_processed += 1

            # Progress update
            elapsed = time.time() - self.start_time
            print(f'\n>>> Progress: {self.problems_processed}/{len(df)} problems '
                  f'({self.total_correct} correct so far) | Elapsed: {elapsed/60:.1f}min\n')

        # Save summary
        total_with_answer = sum(1 for r in all_results if r['ground_truth'] is not None)
        accuracy = self.total_correct / total_with_answer if total_with_answer else 0

        summary = {
            'model': self.cfg.served_model_name,
            'n_samples': self.cfg.n_samples,
            'max_turns': self.cfg.max_turns,
            'temperature': self.cfg.temperature,
            'correct': self.total_correct,
            'total': total_with_answer,
            'accuracy': round(accuracy, 4),
            'total_time_s': round(time.time() - self.start_time, 1),
            'per_problem': all_results
        }
        with open(os.path.join(self.cfg.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\n{"#"*60}')
        print(f'# TRACE GENERATION COMPLETE')
        print(f'# Accuracy: {self.total_correct}/{total_with_answer} ({accuracy*100:.1f}%)')
        print(f'# Total time: {(time.time()-self.start_time)/60:.1f} minutes')
        print(f'# Traces saved to: {self.cfg.output_dir}')
        print(f'{"#"*60}\n')

        return summary

    def _json_safe(self, obj):
        """Handle inf/nan for JSON."""
        if isinstance(obj, float):
            if math.isinf(obj):
                return 'Infinity' if obj > 0 else '-Infinity'
            if math.isnan(obj):
                return 'NaN'
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                with contextlib.suppress(Exception):
                    self.sandbox_pool.get_nowait().close()


# =============================================================================
# Cell 12: Main execution
# =============================================================================
if __name__ == '__main__':
    # Set seed
    set_seed(CFG.seed)

    # Load problems
    df = load_reference_problems()

    # Initialize generator
    generator = TraceGenerator(CFG)

    # Generate all traces
    summary = generator.generate_all_traces(df)

    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total problems: {summary['total']}")
    print(f"Correct (majority vote): {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']*100:.1f}%")
    print(f"Total time: {summary['total_time_s']/60:.1f} minutes")
    print(f"\nTraces saved to: {CFG.output_dir}")
    print("\nTo analyze with replay_selection.py:")
    print(f"  python scripts/replay_selection.py sweep --traces-dir {CFG.output_dir}")
