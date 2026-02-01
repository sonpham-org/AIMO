# Solution 2: imagination-research (2nd Place, AIMO2)

Adapted for AIMO3 (H100, 9 hours, 110 problems).

## Strategy
- **Dual prompting**: 7 CoT + 8 Code samples per problem (15 total)
- **TIR**: Execute code blocks, feed output back to model (max 3 rounds)
- **Early stopping**: Sample-level (stop at first \boxed{}) + question-level (stop when 5+ agree)
- **Dynamic speed**: Reduce samples when running behind schedule
- **Majority voting**: Most frequent valid answer wins

## Model Options

| Option | Model | Source | Notes |
|--------|-------|--------|-------|
| A (default) | `imagination-research/deepseek-14b-sft-dpo2` | HuggingFace | Their best SFT+DPO checkpoint |
| B | Custom fine-tuned model | Your upload | Train your own, upload as Kaggle dataset |
| C | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-AWQ` | HuggingFace | Off-the-shelf, no training needed |

## Kaggle Setup
1. Upload your chosen model as a Kaggle Model or Dataset
2. Update `kernel-metadata.json` with the correct `model_sources` or `dataset_sources`
3. Push: `kaggle kernels push -p kaggle_submissions/solution_2_imagination/`

## Local Testing
```bash
# With external server (llama-server, vLLM server, etc.)
python kaggle_submission.py --api-base http://localhost:8080/v1 --model-name my-model

# With vLLM directly (auto-downloads model)
python kaggle_submission.py --local-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-AWQ

# Limit to N problems for quick test
python kaggle_submission.py --api-base http://localhost:8080/v1 --n-problems 5
```

## Reference
- [imagination-research/aimo2](https://github.com/imagination-research/aimo2) — Original solution
- [recreate_solutions/solution_2_imagination/](../../recreate_solutions/solution_2_imagination/) — Our recreation
