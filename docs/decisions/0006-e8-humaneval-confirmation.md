# ADR 0006: HumanEval+ confirmation of the E8-rep adapters (E8 packet v3)

Status: accepted, pre-registered before any v3 evaluation.

## Why
E8-rep (INCONCLUSIVE, plus-ex +0.019 [-0.030, 0.068]) could not detect a small effect on 261 MBPP+ tasks. The
cheapest way to add evidence is to score the **same 20 adapters** on a different benchmark whose tasks no chain
trained on and no recipe was tuned against: HumanEval+ (EvalPlus, 163 usable tasks; HumanEval/32 excluded because
its own reference fails the test script). No training happens; GPU cost is about one hour, local.

## Design
- Adapters: the 20 v2 adapters, copied from run `d55611f54cc09b94` only if every digest equals the pin in the
  packet (`pinned_adapters`); a mismatch, or any inability to reuse, aborts (never retrains silently).
- Held-out: 3 families x 54 HumanEval+ tasks (hash-assigned), each scored once; replicated arms stay task-level
  mixtures. The pilot/run stages score MBPP+ tune tasks only (the held-out pool is never touched there).
- Prompt: stub plus an instruction to reply with the full function; program = stub + reply + EvalPlus script.
- Arms and contrasts as in v2. **Primary:** d2_plus minus d2_ex, threshold +0.03, same guardrail (d2_plus not
  below d1_plus beyond 0.02). Secondaries as in v2 (exploratory, Bonferroni over 4).

## Pre-registered pooled analysis (exploratory, not a receipt)
After v3, pool the per-task paired differences of v2 (261 MBPP+) and v3 (163 HumanEval+) into one estimate of
d2 plus-ex and plus-none over 424 tasks, with the same paired-difference CI, reported next to the two receipts.

## Expectations (stated in advance)
Power on 163 tasks alone is low (about 15-20% at a true +0.05); the pooled estimate is the useful number. Transfer
from MBPP-style prompts to HumanEval-style prompts is itself uncertain: a null here is not evidence against v2.
