# ADR 0003: imperfect-verifier study (E6-noisyver, packet v12)

Status: accepted, pre-registered before any v12 run.

## Correction
Earlier summaries called the E6 verifier an "exact oracle". It is not. V_vis only checks consistency with
the *visible* evidence; the offline kept-set audit shows only 19–43% of verified kept samples are truly
correct (probe metric; verified chain d1/d2/d3 = 19.4/37.7/42.9%, poolctl 1.7/8.6/5.1%). The old "grounded"
metric was circular with V_vis; the probe metric (fresh oracle probe states no verifier saw) is not.

## Question
How imperfect can the verifier be before verified self-training stops beating a pool-matched unverified
control? (Feng et al. 2406.07515: imperfect verifiers can still prevent collapse; EvalPlus 2305.01210: weak
tests admit false positives, extra tests expose them; Gerstgrasser et al. 2404.01413: keeping real data
helps; Intuitor 2505.19590 / TTRL 2504.16084: self-certainty and majority-vote as proxy rewards.)

## Design
Verifier kinds (`hsi.verification.verifiers`): `visible`; `noisy` (V_vis plus accept unparseable-excluded
inconsistent proposals with probability `fpr`); `tested` (V_vis plus agreement on `extra_probes` oracle
probe states); `confidence` (parseable and mean token logprob >= tau, no world check). Verdicts are pure in
(spec, seed, generation, key, text, score). Ground truth is used only for audit and evaluation.

Primary: d2_noisy10 minus d2_poolctl, probe metric, depth 2, threshold +0.05, local MLX 7B with pinned v9
adapters (no spend). Seven secondary contrasts, Bonferroni z≈2.69, exploratory, never decide. Replicate
chain `noisy10b` estimates chain-to-chain variance.

## Audit results (600 base-model proposals, worlds 1600–1900; `E6-noisyver-audit.json`)
- valid 74%, consistent 7.3%, correct 0.5% (correct|consistent 6.8%).
- Confidence is *anti*-informative: AUC(logprob, correct)=0.32, AUC(logprob, consistent)=0.58. At the 80th
  percentile (tau=-0.108) kept precision is 14.6% consistent, 0% correct. Set tau=-0.108; the `conf` chain is
  expected to do poorly and is included as the self-certainty negative control.
- Noisy contamination (share of kept that is inconsistent): fpr 0.03: 27%, 0.10: 56%, 0.30: 79%.
- `tested` on consistent proposals: m=1 keeps 1.5% of proposals at 33% correct; m=2 and m=4 keep 0.5% at
  100% (n=3). Chose m=2. Yield (~20 of the 175 quota expected at d1) means the tested arm is mostly
  real-data top-up; this confound is reported, not hidden (`shortfall` is logged per chain).

## Risks
Probe-metric scale is smaller than "grounded", so NO_CANDIDATE/INCONCLUSIVE are acceptable outcomes.
tau and m were fixed from the audit worlds (disjoint from TUNE and eval).
