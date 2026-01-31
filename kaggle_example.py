import os
import sys
import re
import math
import time
import queue
import threading
import contextlib
import subprocess
import argparse
import gc
from typing import Optional, List, Dict, Any
from concurrent.futures import as_completed, ThreadPoolExecutor
from collections import Counter

import pandas as pd
import polars as pl
from openai import OpenAI
from jupyter_client import KernelManager
from transformers import set_seed

# Attempt to import specialized libraries from the notebook
try:
    from openai_harmony import (
        HarmonyEncodingName, 
        load_harmony_encoding, 
        SystemContent, 
        ReasoningEffort, 
        ToolNamespaceConfig, 
        Author, 
        Message, 
        Role, 
        TextContent, 
        Conversation
    )
except ImportError:
    print("Warning: openai_harmony not found. Some functionality may be limited.")

try:
    import kaggle_evaluation.aimo_3_inference_server
except ImportError:
    print("Warning: kaggle_evaluation not found. Inference server mode will be unavailable.")

# ==============================================================================
# Step: Configuration and Hyperparameters
# ==============================================================================

# Default values for all tweakable parameters
DEFAULT_MODEL_PATH = "/kaggle/input/gpt-oss-120b/transformers/default/1"
DEFAULT_SERVED_MODEL_NAME = "gpt-oss"
DEFAULT_TIKTOKEN_PATH = "/kaggle/tmp/setup/tiktoken_encodings"
DEFAULT_GPU_UTIL = 0.96
DEFAULT_KV_CACHE_DTYPE = "fp8_e4m3"
DEFAULT_DTYPE = "auto"
DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 16
DEFAULT_CONTEXT_TOKENS = 65536
DEFAULT_ATTEMPTS = 12
DEFAULT_EARLY_STOP = 5
DEFAULT_TURNS = 160
DEFAULT_TEMP = 0.9
DEFAULT_MIN_P = 0.04
DEFAULT_SEED = 42
DEFAULT_GPU_TYPE = "amd"  # Options: "amd", "nvidia"

def get_args():
    parser = argparse.ArgumentParser(description="AIMO-3 Local Inference Script")
    
    # Model and Path Arguments
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the model weights")
    parser.add_argument("--served_model_name", type=str, default=DEFAULT_SERVED_MODEL_NAME, help="Name used for the vLLM server")
    parser.add_argument("--tiktoken_encodings_path", type=str, default=DEFAULT_TIKTOKEN_PATH, help="Path to tiktoken encodings")
    
    # Hardware/Performance Arguments
    parser.add_argument("--gpu_type", type=str, default=DEFAULT_GPU_TYPE, choices=["amd", "nvidia"], help="GPU manufacturer (amd or nvidia)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=DEFAULT_GPU_UTIL, help="GPU memory utilization for vLLM")
    parser.add_argument("--kv_cache_dtype", type=str, default=DEFAULT_KV_CACHE_DTYPE, help="KV cache data type")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, help="Model data type")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Maximum number of sequences per batch")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers/kernels")
    parser.add_argument("--context_tokens", type=int, default=DEFAULT_CONTEXT_TOKENS, help="Maximum context length")
    
    # Solver Tuning Parameters
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS, help="Number of solution attempts per problem")
    parser.add_argument("--early_stop", type=int, default=DEFAULT_EARLY_STOP, help="Number of identical answers required for early stopping")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="Maximum turns per attempt")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
    parser.add_argument("--min_p", type=float, default=DEFAULT_MIN_P, help="Min-P sampling parameter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    
    # Timeouts (in seconds)
    parser.add_argument("--high_problem_timeout", type=int, default=900)
    parser.add_argument("--base_problem_timeout", type=int, default=270)
    parser.add_argument("--notebook_limit", type=int, default=17400)
    parser.add_argument("--server_timeout", type=int, default=180)
    parser.add_argument("--session_timeout", type=int, default=960)
    parser.add_argument("--jupyter_timeout", type=int, default=6)
    parser.add_argument("--sandbox_timeout", type=int, default=3)
    
    # Prompts
    parser.add_argument("--system_prompt", type=str, default=(
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'Think step by step. Be extremely careful and double-check every calculation. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'Place ONLY the final answer inside \\boxed{} — nothing else after it.'
    ))
    parser.add_argument("--tool_prompt", type=str, default=(
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook with sympy, numpy, mpmath, etc. '
        'Always use print() to output the final numerical result.'
    ))
    parser.add_argument("--preference_prompt", type=str, default=(
        'You have access to `math`, `numpy`, `sympy`, `mpmath`. '
        'Use exact fractions / symbolic computation whenever possible.'
    ))

    # Miscellaneous
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server")
    parser.add_argument("--input_csv", type=str, default="test.csv", help="Input CSV file for inference")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Output CSV file")
    
    return parser.parse_args()

args = get_args()

# ==============================================================================
# Step: Environment Setup
# ==============================================================================

def setup_environment(arguments):
    """Sets up environment variables based on GPU type and system paths."""
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['TRANSFORMERS_NO_FLAX'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TIKTOKEN_ENCODINGS_BASE'] = arguments.tiktoken_encodings_path
    
    if arguments.gpu_type == "nvidia":
        # NVIDIA specific optimizations/paths
        os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
        print("Configured environment for NVIDIA GPU.")
    elif arguments.gpu_type == "amd":
        # AMD ROCm specific logic
        # Note: CUDA_VISIBLE_DEVICES is still honored by most ROCm-PyTorch builds
        if 'TRITON_PTXAS_PATH' in os.environ:
            del os.environ['TRITON_PTXAS_PATH']
        
        # Example: GFX version override for consumer cards if necessary
        # os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0' 
        print("Configured environment for AMD GPU (ROCm).")

setup_environment(args)

# Set random seed
set_seed(args.seed)

# ==============================================================================
# Step: AIMO3Template Class
# ==============================================================================
# Handles building the conversation structure for the LLM.

class AIMO3Template:
    def __init__(self):
        pass

    def get_system_content(self, system_prompt: str, tool_config: Any) -> Any:
        # Construct system message content with model identity and tool configuration
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: Any
    ) -> List[Any]:
        # Formulate the initial message list with system and user messages
        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]

# ==============================================================================
# Step: AIMO3Sandbox Class
# ==============================================================================
# Manages isolated Jupyter kernels for Python code execution by the model.

class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> List[int]:
        # Thread-safe port allocation for multiple kernels
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        # Initialize the kernel manager with specific ports
        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        # Pre-import common math libraries
        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import mpmath\n'
            'import itertools\n'
            'import collections\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: List[str]) -> str:
        # Clean up traceback to make it more readable for the LLM
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        # Execute code in the kernel and capture stdout/stderr
        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])
                stderr_parts.append(self._format_error(traceback_list))
            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')
            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr
        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        # Shutdown the kernel and cleanup resources
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        # Reset the namespace of the current kernel
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import mpmath\n'
            'import itertools\n'
            'import collections\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()

# ==============================================================================
# Step: AIMO3Tool Class
# ==============================================================================
# Bridges the LLM's tool calls to the sandbox execution.

class AIMO3Tool:
    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        # Lazy initialization of the sandbox session
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        # Ensures that the last line of code is wrapped in a print statement if not already
        lines = code.strip().split('\n')
        if not lines: return code
        last_line = lines[-1].strip()
        if 'print' in last_line or 'import' in last_line or not last_line or last_line.startswith('#'):
            return code
        lines[-1] = 'print(' + last_line + ')'
        return '\n'.join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name='python', description=self.instruction, tools=[])

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        # Wrap execution output into a Message object
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> List[Message]:
        # Primary interface for processing a code execution request from the LLM
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except Exception as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

# ==============================================================================
# Step: AIMO3Solver Class
# ==============================================================================
# The core orchestration class that manages the vLLM server, sandbox pool, 
# and the solving logic across multiple attempts.

class AIMO3Solver:
    def __init__(self, config, port: int = 8000):
        self.cfg = config
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        
        # Initialize encoding and stop tokens
        try:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        except:
            print("Warning: Could not load harmony encoding.")
            self.encoding = None
            self.stop_token_ids = []
    
        # Pre-load model weights into OS page cache for faster startup
        self._preload_model_weights()
        
        # Start the vLLM server as a subprocess
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)
    
        # Wait for the server to be ready
        self._wait_for_server()
        
        # Initialize a pool of persistent sandboxes
        self._initialize_kernels()
    
        self.notebook_start_time = time.time()
        self.problems_remaining = 50
    
    def _preload_model_weights(self) -> None:
        # Reads model files into memory to speed up initial loading by vLLM
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        files_to_load = []
        total_size = 0
    
        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)
    
        def _read_file(path: str) -> None:
            with open(path, 'rb') as f:
                while f.read(1024 * 1024 * 1024): pass
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')
    
    def _start_server(self) -> subprocess.Popen:
        # Command construction for starting vLLM
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
            '--stream-interval', '200',
            '--async-scheduling',
            '--disable-log-stats',
            '--enable-prefix-caching'
        ]
        
        # Redirect server logs to a file
        os.makedirs('logs', exist_ok=True)
        self.log_file = open('logs/vllm_server.log', 'w')
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)
    
    def _wait_for_server(self):
        # Health check polling for the vLLM server
        print('Waiting for vLLM server...')
        start_time = time.time()
        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
            if return_code is not None:
                self.log_file.flush()
                with open('logs/vllm_server.log', 'r') as log_file: logs = log_file.read()
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
            try:
                self.client.models.list()
                print(f'Server is ready (took {time.time() - start_time:.2f} seconds).\n')
                return
            except:
                time.sleep(1)
        raise RuntimeError('Server failed to start (timeout).\n')
    
    def _initialize_kernels(self) -> None:
        # Bootstrapping the sandbox pool
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()
        self.sandbox_pool = queue.Queue()
        def _create_sandbox(): return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())
        print(f'Kernels initialized in {time.time() - start_time:.2f} seconds.\n')
    
    def _scan_for_answer(self, text: str) -> Optional[int]:
        # Regex extraction of the final boxed answer or descriptive answer
        patterns = [
            r'\\boxed\s*\{\s*([0-9,]+)\s*\}',
            r'final\s+answer\s+is\s*([0-9,]+)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = int(matches[-1].replace(',', ''))
                    if 0 <= value <= 99999: return value
                except ValueError: pass
        return None
    
    def _compute_mean_entropy(self, logprobs_buffer: List) -> float:
        # Calculates average token entropy for quality selection
        if not logprobs_buffer: return float('inf')
        total_entropy, token_count = 0.0, 0
        for top_logprobs_dict in logprobs_buffer:
            if not isinstance(top_logprobs_dict, dict) or not top_logprobs_dict: continue
            token_entropy = 0.0
            for prob_val in top_logprobs_dict.values():
                prob = math.exp(prob_val)
                if prob > 0: token_entropy -= prob * math.log2(prob)
            total_entropy += token_entropy
            token_count += 1
        return total_entropy / token_count if token_count > 0 else float('inf')
    
    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float
    ) -> Dict[str, Any]:
        # Individual attempt logic: conversation management and tool usage
        if stop_event.is_set() or time.time() > deadline:
            return {'Attempt': attempt_index + 1, 'Answer': None, 'Python Calls': 0, 'Python Errors': 0, 'Response Length': 0, 'Entropy': float('inf')}

        sandbox = None
        python_calls, python_errors, total_tokens = 0, 0, 0
        final_answer = None
        logprobs_buffer = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))
    
        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(local_jupyter_timeout=self.cfg.jupyter_timeout, tool_prompt=self.cfg.tool_prompt, sandbox=sandbox)
            
            messages = self.template.apply_chat_template(system_prompt, problem, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)
    
            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline: break
                
                # Render conversation tokens
                prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < 512: break # Buffer threshold

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, 
                    temperature=self.cfg.temperature, 
                    logprobs=self.cfg.top_logprobs, 
                    max_tokens=max_tokens, 
                    prompt=prompt_ids, 
                    seed=attempt_seed, 
                    stream=True, 
                    extra_body={'min_p': self.cfg.min_p, 'stop_token_ids': self.stop_token_ids, 'return_token_ids': True}
                )
    
                token_buffer, text_chunks = [], []
                try:
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline: break
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            if chunk.choices[0].logprobs and chunk.choices[0].logprobs.top_logprobs:
                                logprobs_buffer.extend(chunk.choices[0].logprobs.top_logprobs)
                        
                        # Early scan in chunk text
                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-32:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                break
                finally:
                    stream.close()
    
                if final_answer is not None: break
                if not token_buffer: break
    
                # Parse new messages and handle tool calls
                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
    
                if last_message.channel == 'final':
                    final_answer = self._scan_for_answer(last_message.content[0].text)
                    break
    
                if last_message.recipient == 'python':
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last_message)
                    resp_text = tool_responses[0].content[0].text
                    if resp_text.startswith('[ERROR]') or 'Traceback' in resp_text: python_errors += 1
                    conversation.messages.extend(tool_responses)
    
        except Exception as exc:
            print(f"Error in attempt {attempt_index}: {exc}")
            python_errors += 1
        finally:
            if sandbox:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)
    
        return {
            'Attempt': attempt_index + 1, 'Response Length': total_tokens, 
            'Python Calls': python_calls, 'Python Errors': python_errors, 
            'Entropy': self._compute_mean_entropy(logprobs_buffer), 'Answer': final_answer
        }
    
    def _select_answer(self, detailed_results: List[Dict]) -> int:
        # Logic to choose the best answer from multiple attempts
        answer_list = [r['Answer'] for r in detailed_results if r['Answer'] is not None]
        if not answer_list: return 0
            
        counter = Counter(answer_list)
        most_common = counter.most_common()
        
        # Scenario 1: High consensus
        if most_common[0][1] >= max(4, self.cfg.early_stop):
            return most_common[0][0]
        
        # Scenario 2: Tie-breaking based on entropy
        candidates = [most_common[0][0]]
        if len(most_common) >= 2 and most_common[1][1] == most_common[0][1]:
            candidates.append(most_common[1][0])
        
        best, best_entropy = None, float('inf')
        for res in detailed_results:
            if res['Answer'] in candidates and res['Entropy'] < best_entropy:
                best_entropy = res['Entropy']
                best = res['Answer']
        
        return best if best is not None else most_common[0][0]

    def solve_problem(self, problem: str) -> int:
        # Entry point for solving a single math problem
        print(f'\n--- Solving Problem ---\n{problem}\n')
        user_input = f'{problem} {self.cfg.preference_prompt}'
    
        # Dynamic time management
        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        budget = max(self.cfg.base_problem_timeout, min(self.cfg.high_problem_timeout, time_left - (problems_left_others * self.cfg.base_problem_timeout)))
        deadline = time.time() + budget
    
        stop_event = threading.Event()
        valid_answers, detailed_results = [], []
        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)
    
        try:
            futures = [executor.submit(self._process_attempt, user_input, self.cfg.system_prompt, i, stop_event, deadline) for i in range(self.cfg.attempts)]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)
                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])
                        counts = Counter(valid_answers).most_common(1)
                        if counts and counts[0][1] >= self.cfg.early_stop:
                            stop_event.set()
                            break
                except Exception as exc: continue
        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)
    
        final_choice = self._select_answer(detailed_results) if detailed_results else 0
        print(f"Final Prediction: {final_choice}")
        return final_choice
    
    def __del__(self):
        # Cleanup server and sandboxes
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'log_file'): self.log_file.close()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try: self.sandbox_pool.get_nowait().close()
                except: pass

# ==============================================================================
# Step: Main Execution
# ==============================================================================

def main():
    # Instantiate the solver with command-line arguments
    solver = AIMO3Solver(args, port=args.port)
    
    # Check if we should run the Kaggle inference server or local processing
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN') and 'kaggle_evaluation' in sys.modules:
        def predict(id_df, question_df, answer_df=None):
            id_value = id_df.item(0)
            question_text = question_df.item(0)
            gc.disable()
            final_answer = solver.solve_problem(question_text)
            gc.enable()
            gc.collect()
            return pl.DataFrame({'id': id_value, 'answer': final_answer})

        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        inference_server.serve()
    else:
        # Local processing of a CSV file
        if not os.path.exists(args.input_csv):
            print(f"Error: Input file {args.input_csv} not found.")
            # Example question if file is missing
            example_question = "What is 2 + 2?"
            solver.solve_problem(example_question)
            return

        print(f"Reading problems from {args.input_csv}...")
        test_df = pd.read_csv(args.input_csv)
        results = []
        
        for idx, row in test_df.iterrows():
            prob_id = row.get('id', idx)
            question = row.get('problem') or row.get('question')
            print(f"\nProcessing Problem ID: {prob_id}")
            answer = solver.solve_problem(question)
            results.append({'id': prob_id, 'answer': answer})
        
        output_df = pd.DataFrame(results)
        output_df.to_csv(args.output_csv, index=False)
        print(f"Inference complete. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
