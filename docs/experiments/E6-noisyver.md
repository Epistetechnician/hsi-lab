# E6-noisyver (packet v12): how imperfect can the verifier be?

Local MLX 7B (4-bit) with pinned v9 adapters for the shared arms, five new verifier chains trained to
depth 2, probe metric (fresh oracle probe states no verifier saw), in-distribution held-out worlds.
Source commit `41e024e`. Receipt `E6-noisyver-receipt.json`, digest `ba3820d487a5…`, recomputed OK by the
stdlib validator (A2). Design and audit: ADR 0003, `E6-noisyver-audit.md`.

## Result: INCONCLUSIVE (pre-registered primary)
d2_noisy10 minus d2_poolctl: **+0.0675, 95% CI [0.043, 0.092]**, threshold +0.05. The interval straddles the
threshold, so the packet does not promote. Guardrails held. The direction is clear (CI excludes 0); the
magnitude is not shown to clear the bar.

## Secondary contrasts (exploratory, Bonferroni z=2.69 over 7, never decide)
| contrast | diff | adjusted CI |
| --- | --- | --- |
| d2_verified − d2_poolctl | +0.076 | [0.041, 0.112] |
| d2_noisy30 − d2_poolctl | +0.093 | [0.056, 0.129] (see caveat) |
| d2_noisy10b − d2_poolctl | +0.044 | [0.011, 0.076] |
| d2_tested − d2_verified | +0.105 | [0.062, 0.148] |
| d2_conf − d2_poolctl | +0.024 | [−0.010, 0.057] |
| d2_verified − d2_noisy10 | +0.009 | [−0.031, 0.049] |
| d2_noisy10b − d2_noisy10 | −0.024 | [−0.063, 0.015] |

## Kept-set precision (probe-correct share of synthetic examples)
| chain | d1 | d2 |
| --- | --- | --- |
| noisy10 | 10.3% | 22.3% |
| noisy10b | 11.4% | 14.9% |
| noisy30 | 6.9% | 0 synthetic (see caveat) |
| tested (m=2) | 90.8% (76 synthetic, 99 shortfall) | 97.7% |
| conf | 1.1% | 1.1% |

## Reading
- A verifier that wrongly passes 10% of inconsistent proposals still helps: noisy10 and its replicate are
  both positive over the pool-matched control, and are statistically indistinguishable from the
  exact-visible verifier (+0.009) and from each other (−0.024). Chain-to-chain variation is about 0.02.
- Confidence-only filtering (1% correct kept) does not clearly help (CI includes 0), as the audit predicted.
- The stricter "tested" verifier (≈98% precise) is the best arm by a clear margin over d2_verified (+0.105).
  Precision, not just consistency, is what pays.

## Caveats
- **noisy30 d2 is not a valid dose point.** Its d2 accepted 0 proposals, so the generation trained on 350 real
  examples only. Its +0.093 over poolctl therefore reflects real-data content, not a noisy verifier; do not
  read it as "30% noise is fine". This also shows the dose-response was not cleanly identified.
- d1_tested had a 99-example real top-up (shortfall), a confound at d1; d2_tested was full.
- One lock, one model size, synthetic worlds. Not independent validation (A4). A3 clean rerun not yet
  attempted (~12 h on MLX).
