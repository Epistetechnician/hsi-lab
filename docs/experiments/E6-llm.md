# E6 (LLM) — recursive self-training with an unconstrained free-text generator

Packet: `packets/e6-v2.frozen.json` (E6 version 2, `backbone=llm`). Backbone: Qwen2.5-1.5B-Instruct
4-bit (Apache-2.0, digest-pinned). Generation g trains a fresh rank-8 LoRA (200 steps) on the
previous generation's *own free-text hypotheses* for 350 unlabeled tune-world samples (the
generator is greedy, two-demonstration few-shot, and **not** filtered for validity or consistency —
the earlier pure-Python E6 had restricted its pseudo-labels to consistent hypotheses). Generation 1
is produced by the frozen base. Replay arms swap half of the synthetic set (175/350) for original
oracle-labelled fit-world data. Evaluation: hypothesis selection by LLM scoring among those
consistent with 8 observed transitions, then planning, on held-out rule_induction and
compositional_transfer worlds (n=100 per family).

## Result: NO_CANDIDATE (`E6-llm-receipt.json`; guardrail fails)

| Held-out success (pooled over 2 families, n=200 each) | depth 1 | depth 2 | depth 3 |
| --- | --- | --- | --- |
| with original replay | 0.425 | 0.430 | 0.460 |
| no replay | 0.480 | 0.495 | 0.475 |

d3 replay − d3 no-replay = −0.015, 95% CI [−0.079, +0.049]: the interval's upper end sits just
below the +0.05 threshold, so the packet resolves to NO_CANDIDATE on the effect as well. The
diversity guardrail fails outright (see below), which independently forces NO_CANDIDATE.

**The generator collapsed; the task metric did not.** Generator quality by depth (texts used to
train that depth; validity = parseable *and* consistent with the evidence):

| depth | no-replay validity / signature diversity | replay validity / diversity |
| --- | --- | --- |
| 1 | 4.0% / 0.17 | 6.9% / 0.18 |
| 2 | 3.4% / 0.19 | 6.3% / 0.36 |
| 3 | 0.0% / 0.00 | 0.0% / 0.00 |

By depth 3 every generation, with or without replay, is unparseable repetitive output
(e.g. "mix 4 4 4 4 4 4 4 4 4"), so the signature-diversity metric reads 0.0 against a floor of
0.087 (0.5 × depth-1 no-replay). Replay did not prevent the generator from degenerating. Yet
held-out task success stayed in a 0.43–0.50 band at every depth in both arms: the evaluation
picks hypotheses only among candidates already consistent with the evidence, so a degraded prior
over text has little room to hurt. Success differences between arms are within noise (family-level
swings of ±10 points in opposite directions: rule_induction favours no-replay, compositional
transfer favours replay).

Caveats: the base generator is mostly wrong from the start (4–7% valid), so this tests recursion on
a poor hallucinating generator, not a competent one; "diversity" is signature entropy where all
invalid outputs share one bucket; one seed; pilot flagged `underpowered` (n capped at 100 for
cost). The result supports "unconstrained self-training degrades the generator" and does not
support "that degradation reaches evaluation success when evaluation is evidence-constrained".

Next-packet ideas: score by free generation instead of constrained selection (so generator
quality reaches the metric), a stronger generator (7B Apache model, already licensed), or a
verified-synthetic arm to connect back to E2.
