# ADR 0007: scoping the larger-model code study (E8-7B). Nothing built or provisioned.

Status: **executed 2026-10-21** (owner said go; lean design). Outcome at the end. Builds on ADR 0002 (remote GPU, USD 50 total,
USD 15 per run, USD 24.61 already spent, so **USD 25.39 remain**).

## Question
E8 (1.5B, local) found: running an execution test before self-training helps a little (+0.06 over unfiltered,
pooled), a stronger test than the one shown example does not (+0.000 [-0.043, 0.043]), and self-training does not
beat the untouched base (-0.036). Does any of that change on a model that is actually good at this task? A 7B base
solves a larger share of MBPP+ (less junk to filter, more headroom lost to drift, more signal per example). The
claim at stake is the transferable one: **does verifier strength matter once the generator is competent?**

## Model choice
Qwen2.5-7B-Instruct, bf16 (Apache-2.0, pinned revision already in `MODELS` as `qwen2.5-7b-hf`). Not 14B: roughly
2x the cost for the same question. Not a Coder checkpoint: heavier benchmark contamination, and the claim is about
self-training a general instruct model. (Both are one-line swaps later.)

## Design (same estimator and rules as E8-rep; only the model and scale change)
Chains: plus x3, ex x3, none x1 (7 chains x 2 generations, down from 10 chains), K=6, quota 100, lr 5e-5, 100
steps, tune pool 113 MBPP+ tasks. Held-out: the 261 MBPP+ tasks and the 162 HumanEval+ tasks in **one** packet
(two benchmarks = six families), so the primary and the pooled estimate are the same pre-registered number.
Primary: d2 plus minus d2 ex, threshold +0.03, guardrail d2_plus not below d1_plus beyond 0.02. Secondaries:
plus-none, ex-none, plus-frozen_base, d1 plus-ex. Fresh lock created on the node (as in E6-torch), so no custody
secret ever leaves this Mac; for E8 the held-out tasks are public and the lock fixes order only anyway.

## What must be built first (no GPU needed)
1. `TorchLLM.chat` / `complete` (sampled and greedy) and `train_lora(lead=...)`, matching the MLX `LLM` contract.
2. **Batched generation.** HF `generate` one sequence at a time would cost hours; proposals and greedy eval must be
   batched (64+ sequences). Proposals batch naturally; evaluation goes through the framework one task at a time, so
   add an eval-time prefetch: on first request for (arm, adapter), generate the greedy replies for that adapter's
   whole task bucket in one batch and serve `run_task` from the cache (results unchanged, deterministic given
   the batch order; the cache key includes the adapter digest).
3. Parity tests on CPU with the 1.5B `-hf` checkpoint: adapter identity at init, exact adapter-off, `complete`
   greedy equals `generate`, batched equals unbatched on a small set.

## Compute and cost (estimate; the node's hourly rate is read from `gman usage` at the smoke gate)
H100 bf16 7B, batch 64: proposals 678 per generation about 5 min; 14 generations (d1 shared) about 70 min;
LoRA training 14 adapters x 100 steps about 4 min each, about 55 min; eval: 7 arms x 423 tasks batched by adapter
about 40 min; HumanEval+ and test execution run on the node CPU in parallel. **About 3.5 GPU-hours plus 1 hour of
smoke and slack.** At USD 2-4 per H100-hour that is **USD 9-18**, which straddles the USD 15 per-run cap. The
design above is therefore the lean one; a hard stop at USD 14 on `gman usage`, parking the node on any gate failure.

## Gates (stop at any failure)
| Gate | Deliverable | Pass condition |
| --- | --- | --- |
| G0 | Owner approves spend (USD 15 cap) and the lean design | recorded here |
| G1 | Items 1-3 above, local | parity tests green, suite green |
| G2 | Smoke on the node: one tiny chain + batched eval, 5-10 GPU-minutes | measured seconds-per-generation and USD/hour recomputed; projected total under USD 14, else shrink K/chains |
| G3 | Freeze packet, run all stages detached with `hsi watch`, `gman usage` before and after | receipt validates (A2) |
| G4 | Writeup, ADR outcome, node deleted | `gman usage` recorded |

## Risks
- Torch results are a replication stack, not a reproduction of MLX digests (ADR 0002); A3 will likely mismatch on
  digests, as for E6-torch. Reported as such.
- If the 7B base already scores ~0.75 on MBPP+, the headroom shrinks and every contrast may be null again. That is
  a finding (verification matters less for a competent generator), not a failure.
- Cost overrun if batching under-delivers; G2 exists to catch it before the full run.
- The agent classifier blocks moving custody secrets to a node; the fresh-lock design avoids needing to.

## Decision needed from the owner
Go / no-go on USD <=15 of the remaining USD 25.39, for the lean design above (or say "full" for 10 chains at a
projected USD 12-20, which can exceed the per-run cap).

## Outcome (2026-10-21)
Ran as packet e8-v4 on one H100 (`hsi-lab-7b`, created, used and deleted; USD 10.77 including 20 idle minutes after
a failed first launch, a `~` expanded locally instead of on the node). Cumulative remote spend is now about USD 35.3
of the 50 ceiling; the per-run cap held. Result: NO_CANDIDATE with a tight interval (`docs/experiments/E8-7b.md`).
The build list was completed (batched torch generation, eval prefetch, multi-benchmark pools) and is exercised
by CPU tests. The fresh lock was created on the node and its secret was deleted with the node, so an A3 rerun of
this packet is not possible; the held-out tasks are public, so nothing secret is lost.
