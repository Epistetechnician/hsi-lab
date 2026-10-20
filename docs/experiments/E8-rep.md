# E8-rep (packet e8-v2): replicated chains, gentler recipe

Same question as E8 (ADR 0004), redesigned per ADR 0005: plus x4, ex x4, none x2 replicate chains per tier,
each arm a task-level mixture of its replicates; lr 5e-5, 100 steps, quota 100, K=6; tune pool 113 tasks,
held-out 261 (87 per family). Source commit `039833d` (clean). Receipt `E8-rep-receipt.json`, digest
`5bae620ee0…`, recomputed OK by the stdlib validator (A2).

## Result: INCONCLUSIVE
Primary d2_plus minus d2_ex: **+0.019, 95% CI [-0.030, 0.068]**, threshold +0.03. Guardrail held (d2_plus
0.479 vs d1_plus 0.494). The interval straddles the bar and zero.

## Held-out pass@1 (greedy, EvalPlus tests, n = 261)
| arm | d1 | d2 | kept-set precision (d1 / d2) |
| --- | --- | --- | --- |
| frozen base | 0.513 | | |
| none | 0.456 | 0.379 | 44-48% / 47-55% |
| ex | 0.475 | 0.460 | 74-85% / 70-80% |
| plus | 0.494 | 0.479 | 100% / 100% |

Exploratory contrasts (Bonferroni z=2.50, never decide): d2 plus-none **+0.100 [0.023, 0.177]**; d2 ex-none
**+0.081 [0.013, 0.148]**; d2 plus-frozen_base -0.035 [-0.098, 0.029]; d1 plus-ex +0.019 [-0.043, 0.082].
Per-replicate d2 plus rates: 0.558, 0.397, 0.474, 0.500 (n = 52-78 each, so mostly binomial noise).

## Reading
- **Running any execution test before training beats not testing (about +0.08 to +0.10 at depth 2).** Unfiltered
  self-training degrades the model steadily (0.456 to 0.379); the filtered chains hold near the base.
- **Strength beyond the shown example is not demonstrated.** plus beats ex by +0.019, indistinguishable from
  zero and from the bar. With 261 tasks the study could not detect an effect of this size (about 23% power at a
  true +0.05; less at +0.02).
- The gentler recipe removed the v1 collapse: no filtered chain is distinguishable from the base model, but
  none improves on it either. Self-training here limits damage; it does not add skill.
- Replication across v1 and v2 agrees in sign for plus over ex (+0.049, +0.019) and for filtered over
  unfiltered, with different chains, tasks and recipes.

## Limits
One 1.5B model, MBPP+ only (possibly seen in pretraining), CI conditional on these replicate chains, held-out
pool public and fully used. A3 not attempted. Not independent validation (A4).

## Next
Evaluate these same adapters on HumanEval+ (163 more tasks, loader committed) to raise power; a pre-registered
packet is needed since the pool changes.
