# E6 (7B) v9 — pool-matched control for the verified-synthetic result

Packet `packets/e6-v9.frozen.json` (E6 version 9). The verified-synthetic result (`E6-verified.md`)
had a confound: the verified arms drew proposals from a 600-world pool while the unverified
replay arms used 175 prompts from 50 worlds. This packet adds the control: **unverified**
proposals taken from the *same* 600-world pool (the first 175 proposals, no filter), plus the same
175 original-data examples, so the example count and the real half match the verified arms. Only
verification differs. Pre-registered primary: **d2 verified − d2 pool-matched unverified**,
threshold +0.05, on fresh worlds of the trained rule families from a new lock (n=400 per family;
the pilot asked for 726, so the cap binds and the packet carries the underpowered flag). The
verified and older adapters are reused under pinned digests; only the three control adapters were
trained. Only 7–16% of the control's synthetic examples were evidence-consistent (verified: 100%).

## Result: PROMOTE (`E6-poolctl-receipt.json`, recomputed by the standalone validator)

d2 verified − d2 pool-matched = **+0.141**, 95% CI [+0.106, +0.177] (discordant pairs 172 vs 59
across both families; rule induction 132 vs 39, compositional transfer 40 vs 20). Guardrails held
(output diversity 0.408 vs a floor of 0.204; d2 verified 0.304 not below d1 verified 0.284).

| arm | grounded | exact | output diversity |
| --- | --- | --- | --- |
| real_only (oracle labels, no synthetic) | 0.324 | 0.150 | 0.33 |
| unverified replay, depth 2 (50-world pool) | 0.060 | 0.004 | 0.38 |
| pool-matched unverified, depth 1 / 2 / 3 | 0.126 / 0.163 / 0.163 | 0.036 / 0.050 / 0.068 | 0.45 / 0.44 / 0.49 |
| verified, depth 1 / 2 / 3 | 0.284 / 0.304 / 0.309 | 0.079 / 0.113 / 0.135 | 0.41 / 0.41 / 0.39 |

## Decomposing the earlier gap

The headline comparison (verified vs replay, +0.24) mixed two things. At depth 2:
replay 0.060 → pool-matched unverified 0.163 (**about +0.10 from drawing proposals from far more
worlds**) → verified 0.304 (**+0.14 from verification**). Both parts are real; verification is the
larger single contribution but not the whole story, and the earlier +0.24 overstated it.

## Reading it

- Verification helps beyond pool diversity. Exact match, which verification does not directly
  optimise, also rises (0.113 vs 0.050), so this is not only the metric favouring its own verifier.
- The pool-matched control is itself better than replay and its quality also holds with depth
  (0.126 to 0.163), so world diversity of the synthetic set matters on its own.
- The verified arms approach the supervised reference (0.304–0.309 vs 0.324 grounded; exact
  0.113–0.135 vs 0.150) using only 175 real examples.

## Caveats

In-distribution worlds only; one seed; one prompt per world; the control's examples were mostly
wrong (7–16% consistent), so this isolates "filtered vs unfiltered", not a comparison against a
stronger unverified source. The first attempt at this evaluation crashed at the very end because
the diversity guardrail referenced an arm this packet does not evaluate; it was fixed before any
result existed (the floor now falls back to the guarded chain's own depth-1 arm) and the evaluation
was rerun from scratch, which is why the receipt's source commit is later than the packet's.
