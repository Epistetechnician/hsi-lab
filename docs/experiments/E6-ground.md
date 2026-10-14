# E6 (7B) v6 — recursion measured at the generator, not through planning

Packet `packets/e6-v6.frozen.json` (E6 version 6). Previous E6 packets scored success through a
planner, and success sat on a floor for every arm, so collapse could not show. This packet probes the
generator directly: one prompt per held-out world (n=354 per family, 708 per arm; the pilot called
for 354, so this is the first LLM packet that is **not underpowered**). For one observed action the
model writes a hypothesis; **success = grounded = a valid hypothesis that is consistent with the
observed transitions**. No planning, no fallback. The eight arms and their training are exactly the
7B run's (`E6-llm7b.md`): the adapters are reused, pinned by digest in the packet, and the run fails
closed if retraining would not reproduce them.

Pre-registered: treatment d3 replay vs baseline d3 no-replay, threshold +0.05; guardrails: the
diversity of the model's own outputs at every replay depth ≥ 0.5 × the depth-1 no-replay value, and
d3 replay not below d1 replay by more than 2 points.

## Result: NO_CANDIDATE (`E6-ground-receipt.json`)

| arm | grounded | valid | exact | output diversity |
| --- | --- | --- | --- | --- |
| frozen base (zero-shot, run-on text) | 0.000 | 0.00 | 0.000 | 0.00 |
| real_only (oracle labels, no synthetic) | 0.223 | 0.99 | 0.000 | 0.42 |
| depth 1 no-replay / replay | 0.151 / 0.165 | 0.92 / 0.93 | 0.000 / 0.000 | 0.31 / 0.41 |
| depth 2 no-replay / replay | 0.081 / 0.160 | 0.82 / 0.88 | 0.000 / 0.000 | 0.22 / 0.39 |
| depth 3 no-replay / replay | 0.085 / 0.099 | 0.84 / 0.87 | 0.000 / 0.000 | 0.14 / 0.45 |

- **Primary:** d3 replay − d3 no-replay = +0.014, 95% CI [−0.011, +0.039]. The interval excludes
  +0.05, so the effect is NO_CANDIDATE on its own.
- **Guardrail fails:** d3 replay 0.099 vs d1 replay 0.165, a drop of 6.6 points against a 2-point
  floor. Output diversity held (min replay diversity 0.39 vs floor 0.156).

## Reading it

1. **Collapse is visible once the metric has headroom.** Without replay, grounded generation
   halves from depth 1 to depth 2 (0.151 → 0.081) and the diversity of the model's own outputs
   falls steadily (0.31 → 0.22 → 0.14). This is the signature earlier packets could not show.
2. **Replay protects diversity fully and quality partly.** With replay, output diversity does not
   fall (0.41, 0.39, 0.45) and quality is held through depth 2 (0.160 vs 0.081 without replay), but
   it drops at depth 3 (0.099). So replay delays rather than prevents the quality decline, and the
   pre-registered depth-3 contrast sees almost no difference because both arms are low there.
3. **Exploratory, not in the receipt:** at depth 2 the replay advantage is about +0.08
   (unpaired SE ≈ 0.017; roughly [+0.045, +0.113]); that contrast was not the pre-registered one
   and should be confirmed by a new packet before being relied on.
4. **"Grounded" is not "correct".** Exact-match is 0.000 for every arm, including the real-data
   reference: on held-out rule types (mix, swap+copy) the model writes hypotheses that fit the
   few observed transitions but are not the true rule, and it is never trained on those types.
   Compositional-transfer worlds are much harder (grounded 2–8%) than rule induction (14–28%).
5. **Why earlier packets missed this:** the planner turned a modest generator-quality drop into a
   floor-limited success change. Measuring the generator directly removes that floor.

Caveats: one seed; one prompt per world (one action); the synthetic training texts are the model's
own, valid and consistent only ~3–8% of the time, so this is recursion on a poor generator; the
depth-3 dip for the replay arm is a single observation; held-out rule types are out of the
model's training vocabulary. Next packets: a confirmatory depth-2 contrast, a generator with in-distribution
held-out types so exact match has headroom, and a verified-synthetic arm.
