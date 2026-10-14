# E6 (7B) v7 — confirmatory depth-2 contrast on unseen in-distribution worlds

Packet `packets/e6-v7.frozen.json` (E6 version 7). The previous packet (`E6-ground.md`) suggested,
only as an unpaired exploratory contrast, that original-data replay helps at recursion depth 2
(+0.08 on held-out rule types). This packet pre-registers that contrast before any new data: **d2
replay vs d2 no-replay, threshold +0.05**, on **fresh worlds from the trained rule families**
(rule_induction and compositional_transfer worlds drawn through a new custody lock, with offsets
that could collide with training or pilot seeds shifted out), so the generator has room to be right
(exact match is no longer pinned at zero). Same metric as before (a valid hypothesis consistent
with the observed transitions, one prompt per world), same 7B model, and the same adapters, reused
under their pinned digests. n=328 per family (pilot-derived, within the cap: not underpowered).

## Result: NO_CANDIDATE (`E6-confirm-receipt.json`); the earlier signal did not replicate

| arm | grounded | valid | exact | output diversity |
| --- | --- | --- | --- | --- |
| frozen base | 0.000 | 0.00 | 0.000 | 0.00 |
| **real_only** (oracle labels only) | **0.329** | 0.99 | **0.125** | 0.30 |
| depth 1 no-replay / replay | 0.078 / 0.067 | 0.88 / 0.90 | 0.009 / 0.006 | 0.32 / 0.42 |
| depth 2 no-replay / replay | 0.037 / 0.035 | 0.73 / 0.87 | 0.008 / 0.003 | 0.26 / 0.38 |
| depth 3 no-replay / replay | 0.021 / 0.059 | 0.78 / 0.84 | 0.000 / 0.015 | 0.17 / 0.38 |

- **Primary:** d2 replay − d2 no-replay = **−0.002**, 95% CI [−0.016, +0.013] (discordant pairs 12
  vs 13 across both families). The interval is narrow and excludes +0.05, so it rules out a replay
  advantage of more than about one point here. The +0.08 seen in the previous packet did not
  reproduce; that contrast was unpaired and not pre-registered, and it was measured on held-out
  rule types, so it should be treated as noise or specific to that regime.
- **Guardrail fails:** d2 replay 0.035 vs d1 replay 0.067, a 3.2-point drop against a 2-point floor.
  Output diversity held (min replay diversity 0.377 vs floor 0.162).

## What this does show

1. **The damage from self-generated training data is immediate, not gradual.** The reference
   trained directly on 350 oracle-labelled examples reaches 0.329 grounded and 12.5% exact. Both
   depth-1 arms, trained on text the model itself wrote, are at 0.07–0.08 and ~1% exact.
2. **Replay did not repair it on this metric.** The replay arm is half real data (175 oracle
   examples) and still scores 0.067 at depth 1, about where the no-replay arm is. The synthetic
   half dominates the harm.
3. **Replay preserves diversity, not quality.** Output diversity is consistently higher with replay
   (0.38–0.42 vs 0.17–0.32), and without replay it keeps falling by depth, as before. Grounded
   quality without replay falls monotonically (0.078 → 0.037 → 0.021); with replay it is flat then
   noisy (0.067 → 0.035 → 0.059).
4. The earlier E6 packets' conclusion stands in a narrower form: unconstrained recursive
   self-training degrades the generator and shrinks the diversity of its outputs, replay only
   protects the diversity; we have found no setting where it protects task-relevant quality.

## Caveats

- The synthetic data are the model's own guesses from a two-demonstration prompt, valid and
  evidence-consistent only about 3–8% of the time. This tests what happens when a generator
  recursively trains on mostly-wrong text, not recursion of a competent generator. A cleaner
  mechanism test needs a verified-synthetic arm (keep only consistent outputs) or a stronger
  starting generator; each is a new packet.
- One seed, one prompt per world; the depth-3 replay uptick (0.059) is a single noisy observation.
- Adapters were reused from an earlier run under pinned digests; the clean-environment reproduction
  retrains them and must match those digests.
