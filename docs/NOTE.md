# Does verified self-training work? Eight pre-registered experiments, mostly negative

*A short technical note on what hsi-lab found. Every number is in a committed receipt and recomputes with
`python3 scripts/verify_all.py` (stdlib only, no install). Nothing here has been checked by an independent
party yet.*

## The question
When a language model trains on its own generated data, does filtering that data with a verifier make the
difference, and does the verifier have to be exact? Recursive self-training is known to degrade models
(Shumailov et al.); keeping real data (Gerstgrasser et al., 2404.01413) and verifying synthetic data
(Feng et al., 2406.07515; EvalPlus, 2305.01210 for tests as verifiers) are the known remedies. We tested
how much verifier quality matters, first in controlled synthetic worlds, then in code.

## How the evidence is produced
Each experiment is a frozen packet (hypothesis, arms, threshold, guardrails, identified by digest) run before
looking at results; held-out tasks come from a random salt kept outside the repository; an isolated evaluator
writes a receipt; a separate stdlib-only program recomputes every statistic from the counts (A2); a
clean-environment rerun checks the aggregates (A3). The decision rule is fixed in advance: PROMOTE only if every
baseline and guardrail clears the threshold. Most packets did not promote, and all outcomes are kept.

## What held
| Finding | Evidence |
| --- | --- |
| Verified synthetic data beats unverified replay (rule-induction worlds, 7B) | +0.238 [0.205, 0.270]; reproduced on PyTorch/CUDA at +0.109 [0.073, 0.144] and +0.168 [0.133, 0.202] |
| About half of that gain is just a bigger data pool | verified minus pool-matched control +0.141 [0.106, 0.177] |
| The verifier need not be exact | a verifier that wrongly passes 10% of inconsistent proposals is within 0.01 of the plain one (+0.009 [-0.031, 0.049]); a replicate chain differs by -0.024 |
| Precision is what pays | a verifier that also checks extra probe states (98% precise) beats the plain one by +0.105 [0.062, 0.148] |
| Model confidence is a bad verifier | kept 1% correct samples; +0.024 [-0.010, 0.057] |

## What did not hold
- The gain does not clearly transfer to new rule types (+0.091 clears the bar but a diversity guardrail fails).
- **In code it mostly vanished.** On MBPP+ and HumanEval+ with execution tests as the verifier:
  a 1.5B model is hurt by unfiltered self-training; any test filter limits the damage (+0.062 [0.013, 0.111]
  over none, pooled over 423 tasks), but test strength beyond the single example in the prompt adds nothing
  (plus minus example +0.000 [-0.043, 0.043], pooled). A competent 7B model gains nothing either way
  (-0.0025 [-0.025, 0.020] on 405 tasks; every arm within 0.04 of the base model).
- So the synthetic-world result (precision pays) does not carry over to small-scale code self-training.

## Mistakes we made and caught
We described the first verifier as an "exact oracle"; an audit showed only 19-43% of its kept samples were truly
correct, and the metric we used was circular with it, so we added a non-circular probe metric and re-ran. One
CUDA rerun failed to match the original digests (GPU nondeterminism) while the conclusion replicated; recorded as
a failed reproduction, not hidden. A first-launch path bug wasted about 20 idle GPU minutes ($0.9).

## What this does not show
Small models (1.5B-7B), synthetic worlds plus two code benchmarks that may have been seen in pretraining, a
simple recipe (supervised fine-tuning on a few hundred kept samples, not reinforcement learning), and held-out sets
that are public for the code studies. Statistical power was modest in several packets (stated in advance in
the decision records). No claim here is independently verified.

## Check it yourself
```
git clone <repo> && cd hsi-lab
python3 scripts/verify_all.py          # recomputes all 24 receipts, stdlib only, ~1 minute
```
`docs/REPORT.md` is the one-page summary; `docs/assurance.md` lists every experiment's assurance level;
`docs/decisions/` holds the design and pre-registration for each study; `docs/spot-check.md` is the by-hand
arithmetic check. Cost: about $35 of remote GPU time and several days of laptop compute.
