# E4 (LLM) — LoRA candidate on a frozen Qwen2.5-1.5B-Instruct

Packet: `packets/e4-v2.frozen.json` (E4 version 2, `backbone=llm`). Base: `mlx-community/
Qwen2.5-1.5B-Instruct-4bit` @ 8b40312 (Apache-2.0), weights digest-pinned and never written.
Candidate: rank-8 LoRA on 8 layers, 300 steps (lr 1e-4, batch 8), supervised on oracle-labelled
samples: 421 tune-split (copy, add+copy worlds) + 421 fit-split replay samples (50% replay, the
fix suggested by the first E4). The model picks, per action, the best-scoring hypothesis among
those consistent with 8 observed transitions; the planner then solves the task. Held-out: unseen
compositional-transfer worlds (swap+copy) never touched by training.

## Result: NO_CANDIDATE (`E4-llm-receipt.json`, recomputed and reproduced in a fresh env, 41/41)

| | success |
| --- | --- |
| frozen base | 0.275 (n=200) |
| LoRA candidate | 0.275 (n=200) |

LoRA − base = 0.000, 95% CI [−0.046, +0.046] (discordant pairs 11 vs 11): excludes the +0.05
threshold. Guardrail held, and then some: on fresh fit worlds the candidate scored 92/120 vs
75/120 for the base. The pilot (fit/tune-type pilot seeds) had shown +0.25, which did not carry
over to held-out rule families (n was capped at 200; the pilot flagged `underpowered`).

Diagnostic on development-type worlds only (fresh seeds, not held-out; not in the receipt):
hypothesis-selection accuracy base → LoRA: fit-type 0.72 → 0.91, tune-type 0.41 → 0.57. So the
adapter genuinely learned the training distribution (including the copy primitive), and replay
removed the forgetting seen in the pure-Python E4 (that run lost 9.5 points on fit worlds; this one
gains 14). What it did not do is turn that into success on the held-out swap+copy composition.
I deliberately did not probe held-out-family worlds to explain why: that would leak held-out
information into any follow-up packet.

Ideas for a next packet (each a new packet, not a retry): more adapter training or more tune
families (the adapter has seen only copy and add+copy as new primitives), a scoring prompt that
conveys composition, or evaluating with more support transitions.

Caveats: one seed, one adapter recipe, 1.5B model; a null at n=200 cannot rule out effects of a
few points below the +5 threshold. Receipt and run artifacts: model digest binds checkpoint
weights + adapter weights.
