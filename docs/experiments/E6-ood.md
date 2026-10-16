# E6 (7B) v10 — verified vs unverified synthetic data on held-out rule types

Packet `packets/e6-v10.frozen.json` (E6 version 10). Same adapters as the verified-synthetic run,
evaluated on worlds whose *rule types* never appeared in training (mix and swap+copy), through
a fresh lock; no training was needed. Pre-registered primary: **d2 verified − d2 replay**, threshold
+0.05. Guardrails: output diversity at every verified depth ≥ 0.5 × the depth-1 no-replay value, and
d2 verified not below d1 verified by more than 2 points. n=400 per family (the pilot asked for 644,
so the cap binds; underpowered flag).

## Result: NO_CANDIDATE (`E6-ood-receipt.json`): the effect clears the threshold but a guardrail fails

- **Primary:** d2 verified − d2 replay = **+0.091**, 95% CI [+0.058, +0.124]: above +0.05.
- **Guardrail fails:** d2 verified 0.249 vs d1 verified 0.299, a 5.0-point drop against a 2-point
  floor. By the pre-registered rule the packet resolves to NO_CANDIDATE.

| arm | grounded | exact | output diversity |
| --- | --- | --- | --- |
| real_only | 0.196 | 0.000 | 0.43 |
| verified, depth 1 / 2 / 3 | 0.299 / 0.249 / 0.260 | 0.000 each | 0.42 / 0.50 / 0.50 |
| unverified replay, depth 1 / 2 / 3 | 0.165 / 0.158 / 0.087 | 0.000 each | 0.41 / 0.39 / 0.44 |
| unverified no-replay, depth 1 / 2 / 3 | 0.130 / 0.081 / 0.081 | 0.000 each | 0.28 / 0.20 / 0.12 |

## Reading it

1. **Part of the advantage carries to unseen rule types, but unevenly.** All of the gain is in
   compositional transfer (0.215 vs 0.043; 75 vs 6 discordant pairs). On rule induction the two arms
   tie (59 vs 55 discordant pairs).
2. **The verified arms beat the supervised reference here** (0.30 vs 0.20 grounded). That is
   consistent with verification teaching "stay consistent with the observed evidence", which
   transfers, and not with learning new rules.
3. **There is no evidence of correctness on these types.** Exact match is 0.000 for every arm, so
   unlike the in-distribution packets there is no independent signal that the hypotheses are right,
   only that they fit the observations. Do not read this as the model solving unseen rule types.
4. The verified chain does dip from depth 1 to 2 here (and then flattens), which is what the
   guardrail caught.

Caveats: one seed; one prompt per world; underpowered by the cap; the rule-type split is the
project's own, and the model's training vocabulary never contained these types.
