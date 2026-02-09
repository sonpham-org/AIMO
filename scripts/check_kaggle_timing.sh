#!/bin/bash
# Check Kaggle kernel timing by downloading output and parsing logs
# Usage: ./scripts/check_kaggle_timing.sh <kernel-slug>
# Example: ./scripts/check_kaggle_timing.sh sonphamorg/aimo3-judge-552-feb9

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load Kaggle API token
export $(grep KAGGLE_API_TOKEN "$PROJECT_DIR/.env")

KERNEL="${1:?Usage: $0 <kernel-slug>}"
OUTDIR="/tmp/kaggle_timing_$(echo "$KERNEL" | tr '/' '_')"

echo "=== Checking kernel: $KERNEL ==="
echo ""

# 1. Status
echo "--- Status ---"
kaggle kernels status "$KERNEL" 2>/dev/null
echo ""

# 2. Download output
echo "--- Downloading output to $OUTDIR ---"
rm -rf "$OUTDIR"
kaggle kernels output "$KERNEL" -p "$OUTDIR" 2>/dev/null
echo ""

# 3. Parse timing from traces
if [ -d "$OUTDIR/traces" ]; then
    echo "--- Per-problem timing (from traces) ---"
    python3 -c "
import json, os, sys

trace_dir = '$OUTDIR/traces'
files = sorted(f for f in os.listdir(trace_dir) if f.endswith('.json'))
total_time = 0
methods = {}
for f in files:
    d = json.load(open(os.path.join(trace_dir, f)))
    pid = d.get('problem_index', '?')
    ans = d.get('final_answer', '?')
    method = d.get('selection_method', '?')
    secs = d.get('total_seconds', 0)
    total_time += secs
    methods[method] = methods.get(method, 0) + 1
    print(f'  P{pid:>2}: answer={ans:>5} | {method:<18} | {secs:>6.0f}s ({secs/60:.1f}m)')

n = len(files)
if n > 0:
    avg = total_time / n
    projected = avg * 50
    print(f'')
    print(f'  Problems solved: {n}/50')
    print(f'  Total time: {total_time:.0f}s ({total_time/3600:.2f}h)')
    print(f'  Average: {avg:.0f}s/problem ({avg/60:.1f}m)')
    print(f'  Projected for 50: {projected:.0f}s ({projected/3600:.2f}h)')
    print(f'  Methods: {methods}')
    if projected > 32400:
        print(f'  >>> WOULD TIMEOUT (projected {projected/3600:.1f}h > 9h limit)')
    else:
        print(f'  >>> OK (projected {projected/3600:.1f}h < 9h limit)')
"
else
    echo "  No trace files found"
fi

echo ""

# 4. Check log for timing prints
LOG="$OUTDIR/$(ls "$OUTDIR"/*.log 2>/dev/null | head -1)"
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    echo "--- Timing summaries from log ---"
    python3 -c "
import json
raw = open('$LOG').read().strip()
if raw.startswith('['): raw = raw[1:]
if raw.endswith(']'): raw = raw[:-1]
for line in raw.split('\n'):
    line = line.strip().rstrip(',')
    if not line: continue
    try:
        entry = json.loads(line)
        data = entry.get('data', '')
        if '>>> TIMING' in data or '>>> WARNING' in data:
            t = entry.get('time', 0)
            print(f'  [{t/3600:.2f}h] {data.strip()}')
    except: pass
" 2>/dev/null || echo "  Could not parse log"
fi

echo ""
echo "=== Done ==="
