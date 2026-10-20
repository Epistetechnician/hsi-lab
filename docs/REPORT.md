# hsi-lab: what was tested, what held, what did not

One page for a reviewer. Every number below is in a committed receipt (`docs/experiments/*-receipt.json`),
recomputed from counts by a stdlib-only validator (`validator/recompute.py`). Every claim level is in
`docs/assurance.md`. Nothing here has been checked by an independent party (A4).

## Method (why the numbers can be trusted more than the usual)
Packets (hypothesis, arms, threshold, guardrails) are frozen by digest before any run; held-out cohorts come
from a random lock salt kept outside the repo; an isolated evaluator process produces a receipt; a second
program recomputes the statistics; a clean-environment rerun checks the aggregates (A3). Decisions are
PROMOTE / NO_CANDIDATE / INCONCLUSIVE by a fixed rule (all baselines, all guardrails). Negative results are
kept. The cost of this rigor is that most packets did not promote.

## What held
| Claim | Evidence | Strength |
| --- | --- | --- |
| Typed-event messages beat text at equal bytes (E1) | +0.640 [0.614, 0.667] | A3, 4 families |
| A fixed agent graph beats single / independent agents (E3); adaptive beats fixed under a cap (E5) | +0.352; +0.421 | A3, toy tasks |
| Self-training on **verified** synthetic data beats unverified replay (E6-verified) | +0.238 [0.205, 0.270], 7B | A3; reproduced on PyTorch/CUDA at +0.109 and +0.1675 (E6-torch, A2) |
| ...but about half that gain is just a bigger data pool (E6-poolctl) | verified minus pool-matched +0.141 [0.106, 0.177] | A3 |
| A 10%-noisy verifier still helps; precision is what pays (E6-noisyver) | primary +0.0675 [0.043, 0.092] (below the +0.05 bar: INCONCLUSIVE); tested (98% precise) beats plain by +0.105 | A2 |
| Model-confidence-only filtering does not help (E6-noisyver) | +0.024 [-0.010, 0.057] | A2 |

## What did not hold, or is unresolved
- Verified synthetic data on **new rule types** (E6-ood): effect +0.091 clears the bar but a diversity guardrail
  fails, so NO_CANDIDATE. The gain does not clearly transfer out of distribution.
- Unverified recursion (E6, E6-llm, E6-llmgen): no promotion; E2 verified-vs-unverified only +0.023.
- **Code (E8 v1, MBPP+, 1.5B): NO_CANDIDATE.** Test strength gave +0.049 [-0.009, 0.107] (below +0.03 bar with
  CI through zero) and a guardrail failed. Two chains with the same filter differed by 0.07, and nearly all
  self-trained models scored below the untouched base (0.502). v2 (replicated chains, gentler recipe) is the
  pre-registered follow-up; its power to promote at a true +0.05 is only about 23% (260 paired tasks).
- E7 (relations): INCONCLUSIVE. E6-llm7b: INCONCLUSIVE.

## Corrections of our own claims
- We described the visible-evidence verifier as an "exact oracle". It is not: only 19-43% of its kept samples are
  truly correct (probe metric). The earlier "grounded" metric was circular with it; the probe metric is not.
- A3 for the CUDA run failed (96/175 aggregates match, GPU nondeterminism); the conclusion replicated, the digests
  do not. Recorded, not hidden (ADR 0002).

## What this is not
Not a general result about intelligence or self-improvement. Synthetic worlds plus one small-model code study.
The strongest finding (verifier precision matters) is demonstrated in synthetic worlds and not yet in code.

## How to check it in about 30 minutes
1. `uv run pytest -q` (192 tests).
2. For any row: `python -I validator/recompute.py docs/experiments/<X>-receipt.json <packet.frozen.json>`
   (or `hsi bundle <receipt> <outdir>` to get a self-contained folder).
3. Hand-recompute one receipt with `docs/spot-check.md` and tick the box.
4. For A4 proper: `hsi reproduce` on your own machine at the receipt's `source_commit` (needs the held-out lock
   secret from the owner, by design never in the repo).
