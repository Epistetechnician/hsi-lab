# E6 (7B) v8 — verified synthetic data vs unverified, matched real half

Packet `packets/e6-v8.frozen.json` (E6 version 8), pre-registered before any data. Question left
open by every earlier E6 packet: training on the model's own mostly-wrong text hurts generation at
once and replaying original data does not repair it, so **does verifying the synthetic half?**

Design: a new chain of adapters (d1–d3 "verified"). In each generation the previous model writes
hypotheses for unlabeled worlds; a hypothesis is **verified** if it parses and re-executes on the
observed transitions to give every observation. Proposals are drawn from a pool of 600 worlds until
175 are verified (the same synthetic quota as the replay arms), then 175 original oracle-labelled
examples are added (the same real half and the same 350 examples in total as the replay arms; a
real top-up would have filled any shortfall, but none occurred). Yield rose from 8.4% to 18.9% to
23.3% across generations. Primary, fixed in advance: **d2 verified − d2 replay**, threshold
+0.05, on fresh worlds of the trained rule families from a new custody lock, n=400 per family (the
pilot asked for 644, so the cap binds and the packet carries the underpowered flag). The eight
earlier arms are the pinned, reused adapters.

## Result: PROMOTE (`E6-verified-receipt.json`, recomputed by the standalone validator)

d2 verified − d2 replay = **+0.238**, 95% CI [+0.205, +0.270] (discordant pairs 219 vs 29
across both families). Guardrails held: diversity of the model's own outputs 0.42 vs a floor of
0.156, and d2 verified (0.297) not below d1 verified (0.282).

| arm | grounded | exact | output diversity |
| --- | --- | --- | --- |
| real_only (oracle labels, no synthetic) | 0.351 | 0.165 | 0.30 |
| depth 1 verified | 0.282 | 0.081 | 0.41 |
| depth 2 verified | 0.297 | 0.111 | 0.42 |
| depth 3 verified | 0.314 | 0.152 | 0.39 |
| depth 1 / 2 / 3 replay (unverified) | 0.077 / 0.060 / 0.071 | 0.010 / 0.013 / 0.016 | 0.41 / 0.38 / 0.36 |
| depth 1 / 2 / 3 no-replay | 0.072 / 0.035 / 0.031 | 0.005 / 0.009 / 0.000 | 0.31 / 0.24 / 0.17 |

## What it shows

1. Keeping only verified self-generated examples removes the harm seen in every earlier packet.
   The verified chain does not degrade with depth (0.282, 0.297, 0.314) while unverified arms sit
   at 0.03–0.08, and its output diversity stays flat (0.39–0.42).
2. Exact-match (the generated hypothesis is the true rule) rises with depth, 0.081 to 0.152, close
   to the supervised reference (0.165). Exact-match is not what verification checks, so this is
   independent support that verification improves correctness and not just the measured property.
3. This connects back to E2: there, verification beat trusting a poor generator by a few points
   but lost to real data; here, with a competent-enough model and a larger proposal pool,
   verified self-generated data nearly reaches the supervised reference using only 175 real
   examples.

## Caveats (important)

- **Pool-size confound.** The verified arms drew proposals from 600 worlds; the unverified replay
  arms used 175 prompts from 50 worlds. The label quality (verified vs ~95% wrong) is the obvious
  driver, but a control with unverified examples sampled from the same 600-world pool was not
  run. It is the first thing to run before relying on this result.
- **The verifier shares the metric's definition.** "Grounded" is exactly what verification enforces,
  so the primary metric favours the verified arms by construction. The exact-match trend (point 2)
  is the non-circular evidence; it was not pre-registered as primary.
- **In-distribution only.** Evaluation uses unseen worlds of trained rule families. Held-out rule
  types (mix, swap+copy) were not tested for the verified arms; earlier packets showed those are
  out of vocabulary for this model.
- The verifier needs an oracle-like check on evidence (here, re-execution on observed
  transitions); settings without one cannot do this.
- One seed; one prompt per world; the apparent improvement with depth is within noise between
  adjacent depths and should not be read as a trend.
