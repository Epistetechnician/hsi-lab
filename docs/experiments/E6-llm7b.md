# E6 (LLM, 7B) v5 — free-generation evaluation with a stronger generator

Packet `packets/e6-v5.frozen.json` (E6 version 5). Same design as `E6-llmgen.md` (v4) with three
changes pre-registered in the packet: the backbone is **Qwen2.5-7B-Instruct 4-bit** (Apache-2.0,
revision c26a38f6, weights digest-pinned) instead of the 1.5B; the original labelled data (used
for replay and the `real_only` reference) includes tune-type worlds as well as fit-type worlds, so
the model can learn `copy`; and n=100 per family with 300 adapter steps. Success is the model's
own free-text hypothesis per action, no consistency filter. Held-out families: rule_induction
(mix) and compositional_transfer (swap+copy), neither seen in training.

Why 7B: a development check (fit/tune-type worlds only) showed supervised 7B generation is
100% valid and 37–43% evidence-consistent versus about 15% for the 1.5B.

## Result: INCONCLUSIVE (`E6-llm7b-receipt.json`; recomputed by the validator)

| arm | pooled success | valid | consistent with evidence |
| --- | --- | --- | --- |
| frozen base (nothing parseable → fallback) | 0.165 | 0% | 0% |
| real_only (oracle labels only) | 0.140 | 100% | 20% |
| d1 replay / no replay | 0.110 / 0.105 | 93% / 92% | 18% / 14% |
| d2 replay / no replay | 0.100 / 0.155 | 89% / 83% | 16% / 9% |
| d3 replay / no replay | 0.120 / 0.120 | 88% / 85% | 8% / 8% |

d3 replay − d3 no-replay = **0.000**, 95% CI [−0.057, +0.057], straddling the +0.05 threshold:
INCONCLUSIVE. Discordant pairs are identical in both families (12 vs 12 and 5 vs 5). Guardrails
held (d3 replay 0.12 vs d1 replay 0.11; the diversity floor was set to 0 in this packet, so that
check is a report, not a gate).

## Reading it

- The 7B is a materially better generator than the 1.5B on the evidence it can reason about, but
  on held-out rule types it was never trained on, only about 20% of even the real-data-only
  model's hypotheses are consistent with the evidence. Success therefore stays near the
  fallback floor for every arm (0.10–0.165), and a confident wrong hypothesis is no better than
  none: the untouched base is again the best arm.
- A mild collapse signal exists at the generator level and not in task success: diversity of the
  texts used to train each generation falls without replay (0.40 → 0.27 → 0.23) and falls less
  with replay (0.37 → 0.27 → 0.28); validity of those synthetic texts is only 3–8% throughout
  (they are the model's own guesses from two demonstrations).
- So recursion degrades diversity somewhat, replay cushions it slightly, and neither moves the
  held-out metric because the metric is floor-limited by out-of-distribution rule types.

Caveats: one seed; pilot flagged `underpowered` (n capped at 100); the metric cannot distinguish
arms near the floor. A sharper test needs held-out worlds the model can partly solve (so the
metric has headroom), or a metric on the generator itself (validity/diversity of its own
outputs under recursion), or a verified-synthetic arm. Each would be a new packet.
