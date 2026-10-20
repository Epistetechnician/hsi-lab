# E8-he (packet e8-v3): the v2 adapters on HumanEval+

No training. The 20 adapters of E8-rep (digest-pinned, copied from run `d55611f54cc09b94`) are scored on 162
HumanEval+ tasks (3 x 54; HumanEval/32 excluded) that no chain trained on. Source commit `beac68a` (clean).
Receipt `E8-he-receipt.json`, digest `3fc0c44315…`, recomputed OK by the stdlib validator (A2). ADR 0006.

## Result: INCONCLUSIVE
Primary d2_plus minus d2_ex: **-0.0185, 95% CI [-0.089, 0.052]**, threshold +0.03. Guardrail held (d2_plus 0.414
vs d1_plus 0.432). The interval straddles the bar and zero.

| arm | pass@1 (HumanEval+) |
| --- | --- |
| frozen base | 0.451 |
| d1 none / ex / plus | 0.389 / 0.407 / 0.432 |
| d2 none / ex / plus | 0.389 / 0.432 / 0.414 |

Secondary (Bonferroni z=2.50, exploratory): d2 plus-none +0.025 [-0.073, 0.122]; ex-none +0.043 [-0.050, 0.136];
plus-frozen_base -0.037 [-0.129, 0.055]; d1 plus-ex +0.025 [-0.072, 0.122].

## Pooled with E8-rep (exploratory, `E8-pooled.txt`, 423 paired tasks, unadjusted 95% CIs)
| contrast (d2) | diff | CI |
| --- | --- | --- |
| plus - ex | +0.000 | [-0.043, +0.043] |
| plus - none | +0.062 | [+0.013, +0.111] |
| ex - none | +0.062 | [+0.017, +0.107] |
| plus - frozen_base | -0.036 | [-0.080, +0.008] |

## Reading
- **Test strength beyond the shown example does not measurably matter in this setup.** The pooled estimate is
  zero with a CI that rules out an effect above about +0.04. The pre-registered +0.03 bar is inside that interval,
  so a small effect is not excluded, but there is no signal for one.
- **Running any execution check before training helps modestly** (+0.06 over unfiltered at depth 2, pooled across
  benchmarks); the example-only check captures all of it.
- Self-training still does not beat the untouched base (-0.036, CI includes zero).
- The benefit seen on MBPP+ transfers in sign to HumanEval+ (ex-none and plus-none positive) but not in
  significance on 162 tasks alone.

## Limits
Pooling is post hoc (pre-announced in ADR 0006, not a receipt). Adapters trained on MBPP-style prompts; HumanEval+
may be in pretraining; one 1.5B model. A3 not attempted. Not independent validation (A4).
