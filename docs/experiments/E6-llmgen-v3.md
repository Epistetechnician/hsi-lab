# E6 v3 (free-generation evaluation) — INVALID: harness defect, do not interpret

Packet `packets/e6-v3.frozen.json`, receipt `E6-llmgen-v3-receipt.json` (INCONCLUSIVE, recomputed by
the validator; **not** reproduced in a fresh environment, since the defect makes it moot).

Intended: evaluate E6 by the model's own free-text hypotheses so generator quality reaches the
success metric. What happened: **every arm, including the real-data-only reference, produced 0%
valid hypotheses at evaluation time** (receipt guardrail values `eval_valid_* = 0.0`), and success
for all eight arms sat at the ~0.12–0.17 floor that the planner reaches with no usable
hypotheses (frozen base 0.13).

Root cause (diagnosed on fit-type development worlds, not held-out): the adapters were trained
on targets without an end-of-hypothesis marker, so generation begins with a plausible first op and
then never stops (e.g. truth "add 4 5" → "add 4 5 ; swap 0 1 ; add 2 4 ; swap"), and the strict
parser rejects run-on text. The metric therefore measured termination, not generation quality.

Decision recorded in the receipt (d3 replay − d3 no replay = +0.020, CI [−0.040, +0.080]) carries
no information about recursive collapse. The retry is a new packet (E6 v4) that terminates
training targets with a newline.
