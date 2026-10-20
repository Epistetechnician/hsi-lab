# ADR 0005: replicated chains for the code-domain study (E8 packet v2)

Status: accepted, pre-registered before any v2 chain was trained.

## Why
E8 v1 (NO_CANDIDATE) could not separate the test-strength effect (+0.049 primary) from training noise: two
chains with the same filter differed by 0.07, and every chain except one fell below the untouched base.
Two defects, two fixes, both chosen before looking at any v2 data:

1. **Chain variance.** Each filtered arm becomes a *mixture of replicate chains* (plus x4, ex x4, none x2).
   A held-out task is answered by one replicate, chosen by a hash of (arm, task). Task-level pairing then
   averages chain-to-chain variation instead of letting one lucky chain decide. Caveat: the CI treats the
   replicate chains as fixed, so it is conditional on these chains; the per-replicate rates are reported.
2. **Destructive recipe.** Gentler training: lr 5e-5 (was 1e-4), 100 steps (was 150), quota 100 (was 150),
   K=6 proposals per task (was 8). The v1 audit gives no basis to expect the effect to need more.

Also: tune pool 30% (113 tasks) so the held-out pool grows from 225 to 261 (87 per family).

## Unchanged
Primary d2_plus minus d2_ex, threshold +0.03, guardrail d2_plus not below d1_plus beyond 0.02, MBPP+ tiers,
sandbox, common d1 proposals, depth 2, Qwen2.5-1.5B-Instruct 4-bit. Secondaries (Bonferroni over 4,
exploratory): plus-none, ex-none, plus-frozen_base, d1 plus-ex.

## Expectations (stated in advance)
Power is still modest (261 paired tasks; task-level SE of a paired difference ~0.03). If the true effect is
near +0.05, a PROMOTE is roughly a coin flip; INCONCLUSIVE and NO_CANDIDATE are both live outcomes. If the
gentler recipe still lands below the frozen base, that is a finding about self-training of small code models,
reported as such.
