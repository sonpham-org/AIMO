#!/usr/bin/env python3
"""
Monitor Kaggle kernel submissions and log their completion times.
Run in background to track how long submissions take.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Load .env for KAGGLE_API_TOKEN
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip('"').strip("'")

LOG_FILE = Path(__file__).parent.parent / "output" / "kaggle_kernel_monitor.jsonl"
CHECK_INTERVAL = 300  # 5 minutes

def get_kernel_status(kernel_ref: str) -> dict:
    """Get kernel status via Kaggle CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True, text=True, timeout=30
        )
        # Parse output like: 'sonphamorg/aimo3-entropy-gated-feb3 has status "KernelWorkerStatus.COMPLETE"'
        output = result.stdout.strip()
        if "COMPLETE" in output:
            return {"status": "complete", "raw": output}
        elif "RUNNING" in output:
            return {"status": "running", "raw": output}
        elif "QUEUED" in output:
            return {"status": "queued", "raw": output}
        elif "ERROR" in output or "CANCEL" in output:
            return {"status": "error", "raw": output}
        else:
            return {"status": "unknown", "raw": output}
    except Exception as e:
        return {"status": "error", "raw": str(e)}

def log_event(event: dict):
    """Append event to log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[{event['timestamp']}] {event['kernel']}: {event['status']}")

def monitor_kernels(kernel_refs: list[str]):
    """Monitor kernels until all complete."""
    tracking = {}

    for ref in kernel_refs:
        tracking[ref] = {
            "start_time": datetime.now().isoformat(),
            "first_seen": None,
            "completed_at": None,
        }

    print(f"Monitoring {len(kernel_refs)} kernels...")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"Log file: {LOG_FILE}")
    print()

    while True:
        all_done = True

        for ref in kernel_refs:
            if tracking[ref]["completed_at"]:
                continue

            status = get_kernel_status(ref)
            now = datetime.now().isoformat()

            if tracking[ref]["first_seen"] is None:
                tracking[ref]["first_seen"] = now

            log_event({
                "timestamp": now,
                "kernel": ref,
                "status": status["status"],
                "raw": status["raw"]
            })

            if status["status"] in ["complete", "error"]:
                tracking[ref]["completed_at"] = now

                # Calculate duration if we have start time
                if tracking[ref]["first_seen"]:
                    start = datetime.fromisoformat(tracking[ref]["first_seen"])
                    end = datetime.fromisoformat(now)
                    duration = (end - start).total_seconds()
                    print(f"  -> Duration (from first check): {duration/3600:.2f} hours")
            else:
                all_done = False

        if all_done:
            print("\nAll kernels completed!")
            break

        time.sleep(CHECK_INTERVAL)

def list_recent_kernels():
    """List recent kernels from Kaggle."""
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "-v"],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python monitor_kaggle_kernels.py list")
        print("  python monitor_kaggle_kernels.py <kernel-ref> [kernel-ref2] ...")
        print()
        print("Example:")
        print("  python monitor_kaggle_kernels.py sonphamorg/aimo3-entropy-gated-feb3")
        sys.exit(1)

    if sys.argv[1] == "list":
        list_recent_kernels()
    else:
        monitor_kernels(sys.argv[1:])
