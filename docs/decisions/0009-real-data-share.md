# ADR 0009: does the synthetic share of the training set matter? (proposed E10)

Status: proposed. Scope only; nothing built, frozen or run. Written after E9 (ADR 0008).

## Why
E9 found no dose-response to the correctness of 40 synthetic examples, and every self-trained arm sat at or
below `real_only` (0.181). The all-wrong dose000 arm beat the pool-matched control by +0.100 while
`d2_poolctl` also had 80% valid outputs vs 93-99%, so the rest of the training set (real data, junk outputs)
explains more than synthetic correctness did. The open question is the one the repo's headline hypothesis
needs: **at a fixed total of 350 examples, does replacing real examples with verified synthetic ones help,
hurt, or do nothing?** E1-E9 never varied that share cleanly (v8-v12 fixed it at 50%; E9 at 11%).

## Question
At depth 2, fixed total examples (350) and fixed training budget, how does probe success change with the
synthetic share s of the training set, with the synthetic part drawn from the visible-verified source?

## Design (to freeze as packet e10-v1 after the build list)
- **Arms.** s in {0, 0.11, 0.25, 0.50, 0.75}, two replicate chains per level, depth 2. s=0 is `real_only`
  retrained as a chain (same seeds as the others, so it is a true chain-variance anchor, not the pinned adapter).
  s=0.11 re-uses E9's K=40 with the *visible* verifier (no ground truth), so it is a deployable setting.
- **Held constant.** Total examples, steps, lr, rank, held-out worlds and lock procedure as v12/v13.
  Real examples come from the same real pool, drawn with the chain's seed; only the real/synthetic split moves.
- **Primary (one).** Best-supported pairwise form (the evaluator takes paired contrasts only): s=0.50 chain a
  minus s=0 chain a, threshold +0.05 either direction is informative, decision rule: PROMOTE if >= +0.05 with
  CI above 0; NO_CANDIDATE if the CI is below +0.05. Replicates and other levels are exploratory secondaries
  (Bonferroni over their count). Reading the trend across s is descriptive, not decided.
- **Guardrails.** As v13 (diversity and non-collapse on the guarded chain, guarded on s=0.50).
- **Yield risk.** s=0.75 needs ~262 visible-consistent proposals; at the audited ~12% consistent rate over a
  4,307-proposal pool that is feasible at d1 but marginal at d2. Any chain that cannot fill its synthetic quota
  is reported invalid (as `dose_valid`), not topped up silently with real data.

## Build list
1. Generalise the quota in `train_chain` from `n_total // 2` to a spec field `synthetic_share`, reusing the
   existing `shortfall` / validity reporting; tests for realised share and constant total.
2. v14 template, secondaries, and an audit confirming d2 yield at s=0.50 and 0.75 using the E9 depth-1 proxies.
3. Check that the pinned `real_only` adapter and a retrained s=0 chain agree within chain variance (~0.04);
   if not, report that as a finding before reading any contrast.
4. Power: with n=155 per arm and chain spread ~0.04, a +/-0.05 contrast is near the noise floor. Raise the
   held-out world count (max_n) before freezing; do not run at n=155.

## Cost and non-goals
~10 chains at depth 2 on local MLX 7B, about 24 h as in E9, no cloud spend. Not a general scaling-law claim:
one model, one synthetic world family, one evaluator. A null (no effect of s) is a valid outcome and would
close the line: synthetic experience adds nothing measurable here.

## Decision needed
Go / no-go on ~24 h of local MLX compute after the build list, and whether to raise the held-out size (item 4).
