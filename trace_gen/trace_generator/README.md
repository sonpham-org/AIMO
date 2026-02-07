# AIMO3 Trace Generator

Generate rich TIR (Tool-Integrated Reasoning) traces for all 53 AIMO3 reference problems using `gpt-oss-120b` on Kaggle H100 GPUs.

## Why?

Running inference is expensive (~9 hours on H100). By saving complete traces with:
- All answers from all attempts
- Entropy/logprobs for confidence weighting
- Code execution history
- Prompt type metadata

We can test **138+ selection strategies** in seconds locally using `scripts/replay_selection.py`.

## Files

```
trace_generator/
├── kaggle_trace_generator.py      # Standalone Python script
├── kaggle_trace_generator.ipynb   # Kaggle notebook version
├── kernel-metadata.json           # Kaggle push config
├── reference_dataset/             # Dataset to upload
│   ├── reference.csv              # 53 AIMO3 problems with answers
│   └── dataset-metadata.json      # Kaggle dataset config
└── README.md                      # This file
```

## Setup on Kaggle

### Step 1: Upload Reference Problems Dataset

```bash
cd kaggle_submissions/trace_generator/reference_dataset
kaggle datasets create -p .
```

This creates the `sonphamorg/aimo3-reference-problems` dataset.

### Step 2: Push the Notebook

```bash
cd kaggle_submissions/trace_generator
kaggle kernels push -p .
```

### Step 3: Run on Kaggle

1. Go to https://www.kaggle.com/code/sonphamorg/aimo3-trace-generator
2. Click "Edit" → "Settings"
3. Ensure:
   - **Accelerator**: GPU P100 or better (H100 preferred)
   - **Internet**: OFF (competition mode)
   - **Timeout**: 9 hours
4. Click "Run All"

### Step 4: Download Traces

When complete, download `/kaggle/working/traces/` folder containing:
- `problem_{id}.json` — One file per problem
- `summary.json` — Overall stats
- `config.json` — Generation parameters

## Trace Format

Each `problem_{id}.json` contains:

```json
{
  "problem_id": "abc123",
  "problem_text": "...",
  "ground_truth": 42,
  "wall_time_s": 300.5,
  "attempts": [
    {
      "attempt_idx": 0,
      "answer": 42,
      "answer_source": "boxed",
      "entropy": 0.523,
      "prompt_type": "reasoning",
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
    }
  ],
  "default_answer": 42,
  "default_method": "majority_vote",
  "default_votes": {"42": 8, "0": 2}
}
```

## Analyze Traces Locally

After downloading:

```bash
# Run all 138+ selection strategies
python scripts/replay_selection.py sweep --traces-dir ./traces/

# Results saved to results/experiments.db
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | gpt-oss-120b | MoE, 117B/5.1B active |
| Samples/problem | 12 | 8 reasoning + 2 code + 2 cases |
| Max turns | 128 | Deep TIR |
| Temperature | 1.0 | |
| min_p | 0.02 | |
| Time/problem | 600s | 10 min budget |

## Expected Runtime

- 53 problems × ~10 min = ~9 hours
- Fits within Kaggle's 9-hour GPU limit
