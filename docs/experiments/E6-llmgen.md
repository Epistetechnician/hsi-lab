# E6 (LLM) v4 — free-generation evaluation, recursive self-training

Packet `packets/e6-v4.frozen.json` (E6 version 4). Same generator, depths, and replay design as the
LLM E6 in `E6-llm.md`, with two changes: (1) **evaluation uses the model's own free-text
hypothesis per action** (greedy, zero-shot prompt) with no consistency filter, so generator
quality reaches the success metric; (2) training targets end with a newline so generation stops
(v3 omitted this and was invalid, see `E6-llmgen-v3.md`). Reference arms: `real_only` (adapter
trained on 350 oracle-labelled fit samples) and `frozen_base` (no adapter). 300 adapter steps,
n=100 per family on held-out rule_induction and compositional_transfer.

## Result: NO_CANDIDATE (`E6-llmgen-receipt.json`); the guardrail fails and the effect is negative

| arm | pooled success | valid hypotheses | consistent with evidence |
| --- | --- | --- | --- |
| frozen base (nothing parseable → exact-transition fallback) | 0.145 | 0% | 0% |
| real_only | 0.080 | 100% | 5% |
| d1 replay / no replay | 0.090 / 0.110 | 95% / 86% | 5% / 5% |
| d2 replay / no replay | 0.105 / 0.110 | 99% / 78% | 9% / 4% |
| d3 replay / no replay | 0.060 / 0.120 | 99% / 80% | 6% / 4% |

d3 replay − d3 no-replay = **−0.060**, 95% CI [−0.109, −0.011]: replay is significantly *worse*
here, the opposite of the hypothesis. The non-collapse guardrail also fails (d3 replay 0.06 vs d1
replay 0.09, a 3-point drop against a 2-point floor).

## How to read it

1. **The generator is too weak for this metric to measure collapse.** Even trained directly on 350
   oracle labels, the model's hypotheses are consistent with the evidence only ~5% of the time.
   A confident wrong hypothesis is worse than none: the untouched base, whose text never parses,
   falls back to exact-transition planning and scores highest (0.145). So the arms differ mainly in
   how often they commit to a wrong rule, not in how well they learned.
2. **Why replay "hurts" is a commitment effect.** Replay adapters emit valid text 99% of the time;
   no-replay adapters, trained on their own invalid text, fail to parse ~20% of the time and so
   fall back more often. More commitment to wrong rules means lower success. This is not evidence
   that self-training without replay is safer.
3. **Correction to the earlier LLM E6 (`E6-llm.md`).** There, the generator "collapsed" by depth 3
   (validity 0%, signature diversity 0.0, repetitive output like "mix 4 4 4 4…"). With terminated
   targets (this run) the training texts stay varied and parseable (diversity 0.17–0.29, 26–31
   distinct strings per set, validity 4–11%) with no such collapse. That earlier collapse was
   largely the missing end-of-hypothesis marker compounding run-on output across generations,
   not recursion per se. Treat that doc's collapse finding as confounded; its null on task success
   still stands (it measured constrained selection).
4. **What survives:** (a) recursion over this unconstrained generator does not visibly collapse
   within 3 generations once targets are terminated; (b) a 1.5B model cannot infer these rules
   well enough, even with supervision, for free generation to be a useful agent policy; (c)
   diversity of the generated set is dominated by two or three repeated hypotheses ("mix 3 4",
   "mix 4 0", "swap 2 4"), a mode-heavy distribution from the start.

Caveats: one seed, small training budget (300 steps), diversity metric treats all invalid output
as one bucket, pilot flagged `underpowered` (n capped at 100). A fair test of collapse needs a
generator that is at least moderately competent: a stronger model (7B Apache-licensed checkpoint),
more supervised training, or a verified-synthetic arm. Any of those is a new packet.
