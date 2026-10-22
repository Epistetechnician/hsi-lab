# ADR 0008: close the code line (E8); next study is a controlled precision dose-response (E9)

Status: proposed. Nothing built, frozen or run for E9. Owner go/no-go needed on the compute below.

## Decision 1: E8 is closed
Five packets (v1, rep, he, v4 / 7b) asked whether execution-test strength changes code self-training.
Final answer, on a competent 7B generator over 405 tasks: plus minus ex -0.0025 [-0.025, 0.020], bar +0.03;
every arm within ~0.04 of the untouched base (0.674). Smaller-model results were inconclusive because of
chain-to-chain noise, not because an effect was hiding. No further E8 variants unless a *weak* base model
and a *precise* verifier are both present (see E9). Finding to cite: with a competent generator, this
self-training recipe has no headroom, filtered or not.

## Why E9
The one finding that held up and is useful is "verifier precision is what pays" (E6-noisyver: tested,
~98% probe-correct, beats the exact-visible verifier by +0.105). But its dose-response was never cleanly
identified: precision and kept-set size moved together (noisy30 d2 kept 0 synthetic; tested d1 had a 99-example
real top-up). A claim about precision needs precision to be the only thing that varies.

## Design (to be frozen as packet e9-v1, not yet written)
- **Question.** At fixed kept-set size, fixed real top-up and fixed total examples, how does the probe-metric
  gain over the pool-matched control change with the probe-correct share p of the synthetic part?
- **Arms.** p in {0, 0.25, 0.5, 0.75, 1.0}. The kept set is K proposals drawn from the generator's own
  consistent proposals (visible-verifier passers): probe-correct ones (those that also pass the `tested`
  check) and consistent-but-incorrect ones, mixed at ratio p. Remaining examples are the same real
  top-up for every arm. K = the smallest correct-proposal supply across depths (the d1 `tested` chain kept 76),
  so p=1.0 is feasible at every depth. Anchors: `d2_poolctl`, `d2_verified`, `d2_tested` (pinned from v12).
- **Replication.** Two chains per level (different train seeds, different draw salts): chain variance in
  v12 was ~0.02, which is the resolution the trend must beat.
- **Primary (one).** Slope of d2 probe success on p, pooled over replicates, threshold: slope >= +0.10 per unit
  p (i.e. p=1 vs p=0 differ by at least 0.10) with the 95% CI excluding 0. Monotonicity across the five levels
  is secondary and exploratory.
- **Guardrails.** As v12 (diversity floor, d3 replay not below d1 replay). Add: realised p per chain must be
  within 0.05 of target or the chain is reported invalid, not silently kept.
- **Ground truth use.** Selection by probe-correctness is an *intervention* to set p, not a deployable verifier.
  It must be labelled that way in the packet and the receipt; it never enters evaluation.
- **Cost.** Ten new d1..d2 chains on local MLX 7B. v12's five chains took ~12 h per full set, so expect
  ~20-24 h wall, no cloud spend. Remote ceiling (USD 50, ~35 spent) is untouched.

## Build list (before any freeze)
1. A verifier kind or kept-set builder that takes (K, p, salt) and returns a deterministic kept set; pure in its
   inputs like the existing verdicts; unit tests for realised p, K, determinism, and no overlap with eval worlds.
2. Packet template v13 (extends the v12 template), validator support for a trend-slope primary.
3. Audit run on the audit worlds (1600-1900) to confirm the correct-proposal supply >= K at d1 and d2
   *before* freezing; if supply < K, lower K, do not change p.
4. Pilot to estimate chain variance, as for every packet.

## Non-goals
No new model sizes, no code-domain work, no new base data. A positive result stays inside synthetic rule
worlds (claim ceiling unchanged). A null (flat slope) is a valid, publishable outcome.

## Outstanding, independent of E9
A4 (an external party recomputes from the held-out lock) is still open and is the only step that upgrades
any existing claim. Hand `docs/A4-handoff.md` to a reviewer in parallel.
