# E6-torch (packet e6-v11): the verified-synthetic result on PyTorch/CUDA

Replication of E6-verified (v8) on a different stack: PyTorch 2.14 / CUDA on one H100, full-precision
Qwen2.5-7B-Instruct (Apache-2.0, pinned revision), every adapter trained from scratch, and a **fresh
held-out lock** created on the node. Source commit `f69c6cb`. Receipt `E6-torch-receipt.json`,
digest `4087964f872a…`, recomputed OK by the stdlib validator (A2).

## Result: PROMOTE

d2 verified vs d2 replay (grounded generation rate): **+0.109, 95% CI [0.073, 0.144]**, threshold
0.05, guardrails held. The MLX/4-bit run (v8) gave +0.238 [0.205, 0.270].

| arm (eval consistent rate) | torch d1 | d2 | d3 | MLX v8 d1 | d2 | d3 |
| --- | --- | --- | --- | --- | --- | --- |
| no_replay | 0.096 | 0.098 | 0.068 | 0.072 | 0.035 | 0.031 |
| replay | 0.144 | 0.154 | 0.125 | 0.077 | 0.060 | 0.071 |
| verified | 0.243 | 0.263 | 0.318 | 0.282 | 0.297 | 0.314 |
| real_only | 0.300 | | | 0.351 | | |

## Reading
- The conclusion replicates in direction, decision and sign at every depth: keeping only
  evidence-consistent synthetic examples beats replaying unverified ones, and verified d3 (0.318)
  reaches the real-only level (0.300) while unverified chains stay well below it.
- The effect is **about half the size** (+0.109 vs +0.238). The difference comes from the replay
  arm, which is stronger here (0.154 vs 0.060 at d2) on the full-precision base; the verified arm is
  similar (0.263 vs 0.297). Magnitudes are not expected to match across stacks, so this is a
  replication of the conclusion, not a reproduction (ADR 0002).
- Exact-match rate also favours verified (d2 0.119 vs replay 0.050).

## Limits
- One seed, one generator size, synthetic worlds only. Not independent validation (A4).
- The torch arm uses a fresh lock, so it is a new held-out cohort, not the v8 cohort.
- Not yet reproduced in a clean environment (A3): that needs this lock's secret, which is on the
  GPU node and has not been copied to local custody (see ADR 0002 B4 follow-up).
- Cost: USD 7.43 of the USD 50 ceiling.
