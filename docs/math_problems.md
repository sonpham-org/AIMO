# H100 Trace Data Quality Issues

From data scientist review of 74 problems in `trace_gen/traces_h100_curated/traces/`.

## Answer Leaked in Problem Text (2 problems)

| File | GT | Issue |
|------|-----|-------|
| `problem_bigmath_53b2be09.json` | 6 | Problem text contains `$m=\boxed{6}$` — all 16/16 correct, 98 avg tokens |
| `problem_bigmath_eb57e48b.json` | 7 | Problem text contains `$\boxed{7}$ proper subsets` — all 16/16 correct |

**Action**: Remove from training data. These inflate accuracy metrics.

## Suspicious Ground Truths (7+ problems)

Problems where 14-16/16 samples agree on a different answer than ground truth:

| File | GT | Model Consensus | Agreement | Notes |
|------|-----|-----------------|-----------|-------|
| `problem_aimo3_hard_2c523883.json` | 45 | 46 | 16/16 | e^pi + pi^e = 45.5999, floor=45 vs round=46. Ambiguous. |
| `problem_aimo3_hard_640cd41d.json` | 107 | 321 | 14/16 | Meals per type (107) vs total meals (321). Interpretation difference. |
| `problem_aimo3_hard_f8503e38.json` | 718 | 720 | 14/16 | Very close, could be rounding. |
| `problem_bigmath_5419657f.json` | 4430 | 440 | 14/16 | Off by 10x — likely GT error. |
| `problem_bigmath_57b1f60d.json` | 2090 | 1390 | 16/16 | Possible GT error. |
| `problem_bigmath_e5a45f13.json` | 10 | 17 | 16/16 | Non-standard locker problem (skips Student 2). |
| `problem_bigmath_e94549ae.json` | 1 | 84 | 15/16 | GT=1 for a distance problem seems wrong. |
| `problem_bigmath_1a7d5b0f.json` | 0 | — | — | GT=0 for a bus distance problem (physically meaningless). |

**Action**: Re-validate these ground truths manually before using for training/eval.

## Other Findings

- **11.1% of samples (132/1184) have answer=None** — mostly from hitting 12-turn limit (104/132 have n_turns=12)
- **ngram_rep_4 strongly correlates with accuracy**: <0.05 → 74%, 0.05-0.10 → 55%, 0.10-0.15 → 32%, >0.15 → 20%
- **Single-turn pure reasoning (65.6%) outperforms multi-turn code (38.8%)** — reflects difficulty selection: easy problems solve in one shot
- **Only 2/1184 samples changed their answer** during multi-turn reasoning (high answer stability)
- **3,741 tokens out of 5.94M (0.063%) have logprob < -20** — likely FP8 KV cache quantization artifacts, rare enough to ignore
