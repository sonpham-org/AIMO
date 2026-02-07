"""Isolated Jupyter kernel for safe Python code execution.

Shared across all solution recreations. Provides a stateful Python environment
with common math libraries pre-imported.
"""

import os
import re
import time
import queue
import threading
import contextlib
from typing import List, Optional

from jupyter_client import KernelManager


class Sandbox:
    """Isolated Jupyter kernel for safe Python code execution."""

    _port_lock = threading.Lock()
    _next_port = 50000

    PREIMPORT = (
        "import math, numpy, sympy, mpmath, itertools, collections, functools\n"
        "from sympy import *\n"
        "mpmath.mp.dps = 64"
    )

    @classmethod
    def _alloc_ports(cls, n: int = 5) -> List[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + n))
            cls._next_port += n
            return ports

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        ports = self._alloc_ports()

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        env = os.environ.copy()
        env.update({
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "JUPYTER_PLATFORM_DIRS": "1",
            "PYTHONWARNINGS": "ignore",
            "MPLBACKEND": "Agg",
        })
        self._km.start_kernel(
            env=env, extra_arguments=["--Application.log_level=CRITICAL"]
        )

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._timeout)

        self.execute(self.PREIMPORT)

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        timeout = timeout or self._timeout
        msg_id = self._client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        t0 = time.time()

        while True:
            if time.time() - t0 > timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Timed out after {timeout}s"
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mt = msg.get("msg_type")
            ct = msg.get("content", {})

            if mt == "stream":
                target = stdout_parts if ct.get("name") == "stdout" else stderr_parts
                target.append(ct.get("text", ""))
            elif mt == "error":
                tb = ct.get("traceback", [])
                clean = [re.sub(r"\x1b\[[0-9;]*m", "", f) for f in tb]
                stderr_parts.append("\n".join(clean))
            elif mt in ("execute_result", "display_data"):
                text = ct.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif mt == "status" and ct.get("execution_state") == "idle":
                break

        out = "".join(stdout_parts).rstrip()
        err = "".join(stderr_parts).rstrip()
        if err:
            return f"{out}\n{err}" if out else err
        return out or "[No output — use print() to see results.]"

    def reset(self):
        self.execute(f"%reset -f\n{self.PREIMPORT}")

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)
        with contextlib.suppress(Exception):
            self._km.cleanup_resources()

    def __del__(self):
        self.close()
