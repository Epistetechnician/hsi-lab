# ADR 0004: test strength in code self-training (E8, packet v1)

Status: accepted, pre-registered before any E8 chain was trained.

## Why
E6-noisyver (ADR 0003) found in synthetic rule worlds that verifier *precision* is what pays (the
~98%-precise "tested" verifier beat the consistency-only one by +0.105). That is a synthetic result. Code
generation is the natural real domain: EvalPlus (Liu et al., arXiv 2305.01210) showed that weak tests
(the original MBPP/HumanEval assertions) admit wrong programs and that stronger tests expose them; Feng et
al. (2406.07515) show verification is what keeps self-training from collapsing. Nobody we know of has
measured the *training-time* cost of a weak test filter in this setting under a locked, recomputed protocol.

## Setup
- Data: MBPP+ (EvalPlus, 378 tasks, `data/mbppplus.jsonl`, sha256 pinned in code). Two tasks whose own reference
  fails the EvalPlus script (590, 599) are excluded. Hash split: 40% tune pool (150 tasks), 60% held-out
  (226; 225 used, 75 per family). Chains read only tune tasks (asserted); held-out tasks are only scored.
- Model: Qwen2.5-1.5B-Instruct 4-bit (Apache-2.0, pinned), local MLX, LoRA r=8. Zero GPU spend.
- Chain: generation g samples K=8 solutions per tune task at T=0.8 with the previous model, keeps those that
  pass the chain's test tier (up to 2 distinct per task), fills a fixed quota of 150 examples, trains a fresh
  adapter (150 steps). Depth 2. Generation-1 proposals are shared by all chains (common random numbers), so
  chains differ only in the filter and the selection/training seeds.
- Tiers: `none` (any extractable code), `ex` (the one assertion shown in the prompt), `base` (the original
  MBPP assertions), `plus` (EvalPlus script). Plus replicate chain `plusb`.
- Metric: greedy pass@1 under `plus` on held-out tasks (hidden tests, never used to filter).
- Sandbox: subprocess, `python -I`, scratch dir, rlimits, 10 s timeout, macOS `sandbox-exec` with network
  denied and writes limited to the scratch dir.

## Pre-registration
Primary: d2_plus minus d2_ex, threshold +0.03, guardrail: d2_plus not below d1_plus beyond 0.02.
Six secondary contrasts (Bonferroni, exploratory, never decide): plus-none, plus-base, base-ex, ex-none,
plus-frozen_base, plusb-ex. Power is limited (226 paired tasks, SE of a paired difference ~0.02), so
INCONCLUSIVE is the most likely honest outcome for a true effect near the threshold.

## Audit (frozen base, 1,576 generations; `docs/experiments/E8-audit.json`)
Held-out greedy pass@1 under plus: 0.504 (headroom). Of sampled tune proposals: plus-pass 0.466. False
positives of a tier = P(fails plus | passes tier): none 0.53, ex 0.21, base 0.14, plus 0. Capacity at 2/task:
ex 256, base 237, plus 222 (quota 150 is reachable by every tier at generation 1).

## Non-claims
One small model, one benchmark that Qwen2.5 may have seen in pretraining, and a finite public held-out pool
(the lock fixes only order; there is no secrecy). Not independent validation (A4).
