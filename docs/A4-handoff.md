# A4 handoff: what we are asking an independent reviewer to do

**Goal.** Move the claims from "carefully built by the authors" to "reproduced by someone else". About two hours
for the full list, 30 minutes for the first two items.

1. **Recompute (10 min, no install).** Clone the repository, run `python3 scripts/verify_all.py`. Expect
   `all receipts recompute`. This uses only the Python standard library and the committed packets and receipts.
2. **Hand-check one receipt (20 min).** Pick any row of `docs/spot-check.md`; recompute the paired difference and
   confidence interval from the printed counts by the formula at the top of that file; compare with the receipt.
3. **Read the pre-registration against the outcome (30 min).** For any experiment, open `packets/<x>.frozen.json`
   and its ADR in `docs/decisions/`, and confirm the receipt's decision follows the stated rule (threshold,
   guardrails, all baselines). E8 (`0004`-`0007`) is the cleanest example.
4. **Full test suite (10 min).** `uv sync && uv run pytest -q` (200 passed, 7 skipped).
5. **Optional, needs the owner (hours).** An independent rerun of a packet needs the held-out lock secret, which
   is deliberately not in the repository. For the code studies the held-out tasks are public (MBPP+, HumanEval+)
   and a rerun with a fresh lock is possible without any secret: `scripts/run_packet.sh packets/e8-v2.frozen.json`
   on an Apple-silicon Mac (about 3 hours) reproduces the E8-rep protocol with new chains.

**Known weak points to look at first.**
- Power was low in several packets (stated in the decision records); several results are inconclusive by design.
- The code studies' held-out tasks are public and fully used; the lock only fixes order there.
- CUDA runs are replications, not digest-level reproductions (GPU nondeterminism, ADR 0002).
- Pooled analyses (E8-pooled.txt) are post hoc by design, though announced in ADR 0006.
- The Linux sandbox used on the GPU node has no network block.

**What to send back.** A pass/fail per item, plus anything that does not recompute, with the receipt name.
