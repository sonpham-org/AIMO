# Eagle3 Parameter Scaling Strategy

## Current Baseline (feb3, 40/50)
- attempts = 8, workers = 8, turns = 128, early_stop = 4
- ~50 problems in ~17400s (4h50m)
- ~348s per problem average

## Speedup Scenarios

### Scenario A: 1.3x speedup (conservative, high concurrency)
Effective budget per problem: 348s worth of work in ~268s
- Savings: ~80s per problem = ~4000s total = ~67 min saved

**Recommended scaling:**
| Param | Current | Scaled | Rationale |
|-------|---------|--------|-----------|
| attempts | 8 | 10 | +25% more votes for consensus |
| workers | 8 | 10 | Match attempts |
| turns | 128 | 128 | Keep same (depth isn't bottleneck) |
| early_stop | 4 | 4 | Keep same threshold |

### Scenario B: 1.5x speedup (moderate)
Effective budget: 348s worth of work in ~232s
- Savings: ~116s per problem = ~5800s total = ~97 min saved

**Recommended scaling:**
| Param | Current | Scaled | Rationale |
|-------|---------|--------|-----------|
| attempts | 8 | 12 | 50% more votes |
| workers | 8 | 12 | Match attempts |
| turns | 128 | 128 | Keep same |
| early_stop | 4 | 5 | Higher consensus bar with more attempts |

### Scenario C: 2.0x speedup (optimistic, batch=1 speeds)
Effective budget: 348s worth of work in ~174s
- Savings: ~174s per problem = ~8700s total = ~145 min saved

**Recommended scaling:**
| Param | Current | Scaled | Rationale |
|-------|---------|--------|-----------|
| attempts | 8 | 16 | 2x more votes |
| workers | 8 | 16 | Match attempts |
| turns | 128 | 160 | Deeper reasoning too |
| early_stop | 4 | 6 | Higher bar with 16 attempts |

## What Matters Most?

### More attempts (consensus quality)
- Going from 8→12 attempts: significant improvement in consensus reliability
- Each additional attempt is another "voter" — more voters = more robust majority
- Especially helps on problems where answers are split (e.g., 3-3-2 vote)
- Diminishing returns above ~16 attempts

### More turns (reasoning depth)
- Going from 128→160: helps on harder problems requiring longer chains
- Most problems solve in <50 turns, so this mainly helps edge cases
- Less impactful than more attempts overall

### More workers (parallelism)
- Should match `attempts` — having fewer workers than attempts creates a bottleneck
- More workers = more Jupyter kernels = more memory, but kernels are lightweight

## Decision Framework

After running the Eagle3 notebook and getting actual tokens/sec:

```
measured_speedup = eagle3_tok_per_sec / baseline_tok_per_sec

if measured_speedup >= 1.8:
    use Scenario C (16 attempts, 16 workers, 160 turns, early_stop=6)
elif measured_speedup >= 1.4:
    use Scenario B (12 attempts, 12 workers, 128 turns, early_stop=5)
elif measured_speedup >= 1.2:
    use Scenario A (10 attempts, 10 workers, 128 turns, early_stop=4)
else:
    keep current (8 attempts, 8 workers, 128 turns, early_stop=4)
    # Eagle3 not helping at high concurrency — just enjoy lower latency
```

## What Else to Do with Extra Inference Time

1. **Lower base_problem_timeout** (270→200s): Give more time budget to harder problems
2. **Increase high_problem_timeout** (900→1200s): Let hard problems run longer
3. **Multiple prompt variants**: Use different system prompts for different attempts
4. **Temperature diversity**: Use different temperatures (0.7, 0.8, 1.0, 1.1) across attempts
5. **Second pass**: Re-attempt problems where confidence was low (entropy > threshold)

## Logprobs Risk Mitigation

If Eagle3 breaks logprobs/entropy calculation:
- Fall back to **simple majority vote** (remove entropy weighting)
- Or fall back to **pure consensus** (most common answer wins)
- Monitor: if all entropies are inf or 0.0, logprobs are broken
