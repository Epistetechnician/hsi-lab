# Human spot-check worksheet (A4, pending owner)

For one receipt per experiment, recompute by hand from the counts below. For each family f with n paired tasks, only_t and only_b discordant counts:

- d_f = (only_t − only_b) / n;  var_f = (only_t + only_b − (only_t − only_b)²/n) / n²

- D = mean(d_f);  SE = sqrt(Σ var_f) / F;  CI = D ± 1.96·SE

- Decision: NO_CANDIDATE if a guardrail fails or CI_high < threshold; PROMOTE if CI_low ≥ threshold for every comparison; else INCONCLUSIVE.

Then tick the box. Nothing below has been checked by a human yet.


## E1 — receipt `abfd2649bda9`, threshold 0.05, decision PROMOTE


text → typed_event (receipt: diff 0.6404, CI [0.6141, 0.6668])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 324 | 91 | 233 | 0 | 0 |
| compositional_transfer | 324 | 24 | 151 | 8 | 141 |
| counterevidence | 324 | 77 | 223 | 3 | 21 |
| relational_return | 324 | 87 | 234 | 0 | 3 |

Guardrails: {'byte_budget': 200.0, 'max_payload_bytes': 198.0, 'recon_exact_fraction': 1.0}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E2 — receipt `99adb201d45f`, threshold 0.05, decision NO_CANDIDATE


real_only → verified_synthetic (receipt: diff -0.1902, CI [-0.2167, -0.1638])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 297 | 145 | 9 | 65 | 78 |
| compositional_transfer | 297 | 54 | 7 | 60 | 176 |
| counterevidence | 297 | 132 | 11 | 68 | 86 |
| relational_return | 297 | 153 | 10 | 70 | 64 |

unverified_synthetic → verified_synthetic (receipt: diff 0.0227, CI [0.0091, 0.0364])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 297 | 137 | 17 | 5 | 138 |
| compositional_transfer | 297 | 55 | 6 | 4 | 232 |
| counterevidence | 297 | 133 | 10 | 4 | 150 |
| relational_return | 297 | 148 | 15 | 8 | 126 |

Guardrails: {'example_count_equal_fraction': 1.0, 'family_regression_floor': 0.02, 'lineage_fraction': 1.0, 'min_family_diff_vs_real_only': -0.20202020202020202}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E3 — receipt `fa447f30b6c8`, threshold 0.05, decision PROMOTE


single → fixed_graph (receipt: diff 0.3525, CI [0.2968, 0.4082])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| collective_discovery | 400 | 103 | 160 | 19 | 118 |

independent → fixed_graph (receipt: diff 0.2375, CI [0.1889, 0.2861])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| collective_discovery | 400 | 155 | 108 | 13 | 124 |

Guardrails: {'compute_matched_fraction': 1.0, 'max_message_bytes_used': 24.0, 'message_byte_cap': 24.0}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E4 — receipt `cd616fd7992f`, threshold 0.05, decision NO_CANDIDATE


frozen_base → lora_candidate (receipt: diff -0.0075, CI [-0.0438, 0.0288])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| compositional_transfer | 400 | 89 | 26 | 29 | 256 |

Guardrails: {'compute_matched_fraction': 1.0, 'fit_n': 200.0, 'fit_regression_floor': 0.02, 'fit_success_base': 158.0, 'fit_success_candidate': 139.0}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E5 — receipt `f0c00ae17b53`, threshold 0.05, decision PROMOTE


fixed_graph → adaptive_graph (receipt: diff 0.4213, CI [0.3854, 0.4571])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 400 | 124 | 138 | 6 | 132 |
| compositional_transfer | 400 | 48 | 212 | 7 | 133 |

Guardrails: {'bytes_total_adaptive_graph': 21004.0, 'bytes_total_fixed_graph': 53876.0, 'compute_matched_fraction': 1.0, 'link_churn_mean_adaptive_graph': 4.5, 'link_churn_mean_fixed_graph': 0.0, 'max_message_bytes_used': 70.0, 'message_byte_cap': 72.0, 'success_per_kb_adaptive_graph': 25.44886688249857, 'success_per_kb_fixed_graph': 3.5162224367065114}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6 — receipt `7a7d1247e9be`, threshold 0.05, decision NO_CANDIDATE


d3_no_replay → d3_replay (receipt: diff -0.0154, CI [-0.0391, 0.0084])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 130 | 70 | 0 | 3 | 57 |
| compositional_transfer | 130 | 25 | 3 | 4 | 98 |

Guardrails: {'diversity_floor': 0.2684704263206556, 'min_replay_diversity': 0.35193747488317756, 'non_collapse_floor': 0.02, 'rate_d1_replay': 0.38076923076923075, 'rate_d3_replay': 0.3769230769230769}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E7 — receipt `e6c50d23df7f`, threshold 0.05, decision INCONCLUSIVE


no_relations → adaptive_relations (receipt: diff 0.0387, CI [0.0152, 0.0623])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| relational_return | 400 | 144 | 49 | 17 | 190 |
| counterevidence | 400 | 135 | 14 | 15 | 236 |

static_relations → adaptive_relations (receipt: diff 0.1425, CI [0.1102, 0.1748])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| relational_return | 400 | 137 | 56 | 6 | 201 |
| counterevidence | 400 | 53 | 96 | 32 | 219 |

Guardrails: {'compute_matched_fraction': 1.0, 'duplicate_strength_updates': 0.0}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E4-llm — receipt `153878642bdb`, threshold 0.05, decision NO_CANDIDATE

frozen_base → lora_candidate (receipt: diff 0.0000, CI [-0.0460, 0.0460])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| compositional_transfer | 200 | 44 | 11 | 11 | 134 |

Guardrails: {'compute_matched_fraction': 1.0, 'fit_n': 120.0, 'fit_regression_floor': 0.02, 'fit_success_base': 75.0, 'fit_success_candidate': 92.0}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-llm — receipt `8bd0f0588fd9`, threshold 0.05, decision NO_CANDIDATE

d3_no_replay → d3_replay (receipt: diff -0.0150, CI [-0.0791, 0.0491])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 100 | 55 | 3 | 15 | 27 |
| compositional_transfer | 100 | 16 | 18 | 9 | 57 |

Guardrails: {'diversity_d3_no_replay': -0.0, 'diversity_d3_replay': -0.0, 'diversity_floor': 0.08675854327372498, 'min_replay_diversity': -0.0, 'non_collapse_floor': 0.02, 'rate_d1_replay': 0.425, 'rate_d3_replay': 0.46, 'validity_d3_no_replay': 0.0, 'validity_d3_replay': 0.0}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-llmgen (v4) — receipt `43dc7a01776f`, threshold 0.05, decision NO_CANDIDATE

d3_no_replay → d3_replay (receipt: diff -0.0600, CI [-0.1093, -0.0107])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 100 | 3 | 4 | 9 | 84 |
| compositional_transfer | 100 | 2 | 3 | 10 | 85 |

Guardrails: {'diversity_d3_no_replay': 0.2643844043993107, 'diversity_d3_replay': 0.2852077682752576, 'diversity_floor': 0.0, 'min_replay_diversity': 0.18070647482209062, 'non_collapse_floor': 0.02, 'rate_d1_replay': 0.09, 'rate_d3_replay': 0.06, 'validity_d3_no_replay': 0.04285714285714286, 'validity_d3_replay': 0.11428571428571428}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-llm7b — receipt `dfbcbc239dfd`, threshold 0.05, decision INCONCLUSIVE

d3_no_replay → d3_replay (receipt: diff 0.0000, CI [-0.0571, 0.0571])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 100 | 6 | 12 | 12 | 70 |
| compositional_transfer | 100 | 1 | 5 | 5 | 89 |

Guardrails: {'diversity_floor': 0.0, 'min_replay_diversity': 0.26678812266154167, 'non_collapse_floor': 0.02, 'rate_d1_replay': 0.11, 'rate_d3_replay': 0.12}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-ground — receipt `da550f69c90a`, threshold 0.05, decision NO_CANDIDATE

d3_no_replay → d3_replay (receipt: diff 0.0141, CI [-0.0106, 0.0388])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 354 | 21 | 31 | 31 | 271 |
| compositional_transfer | 354 | 4 | 14 | 4 | 332 |

Guardrails: {'diversity_floor': 0.15602001205706564, 'min_replay_diversity': 0.39310467378251435, 'non_collapse_floor': 0.02, 'rate_d1_replay': 0.1652542372881356, 'rate_d3_replay': 0.09887005649717515}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-confirm — receipt `8dd868227d8d`, threshold 0.05, decision NO_CANDIDATE

d2_no_replay → d2_replay (receipt: diff -0.0015, CI [-0.0165, 0.0134])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction@tune | 328 | 8 | 9 | 9 | 302 |
| compositional_transfer@tune | 328 | 3 | 3 | 4 | 318 |

Guardrails: {'diversity_floor': 0.16225727465384066, 'min_replay_diversity': 0.37726109285765164, 'non_collapse_floor': 0.02, 'rate_d1_replay': 0.06707317073170732, 'rate_d2_replay': 0.0350609756097561}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-verified — receipt `eec822901bc3`, threshold 0.05, decision PROMOTE

d2_replay → d2_verified (receipt: diff 0.2375, CI [0.2046, 0.2704])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction@tune | 400 | 10 | 177 | 15 | 198 |
| compositional_transfer@tune | 400 | 9 | 42 | 14 | 335 |

Guardrails: {'diversity_floor': 0.15648112195519226, 'min_replay_diversity': 0.4084324488631278, 'non_collapse_floor': 0.02, 'rate_d1_verified': 0.2825, 'rate_d2_verified': 0.2975}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-poolctl — receipt `6e0b33050454`, threshold 0.05, decision PROMOTE

d2_poolctl → d2_verified (receipt: diff 0.1413, CI [0.1059, 0.1766])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction@tune | 400 | 59 | 132 | 39 | 170 |
| compositional_transfer@tune | 400 | 12 | 40 | 20 | 328 |

Guardrails: {'diversity_floor': 0.2038678601747587, 'min_replay_diversity': 0.4077357203495174, 'non_collapse_floor': 0.02, 'rate_d1_verified': 0.28375, 'rate_d2_verified': 0.30375}

- [ ] hand-recomputed and matches (owner initials/date: ______)

## E6-ood — receipt `3217e5294f49`, threshold 0.05, decision NO_CANDIDATE

d2_replay → d2_verified (receipt: diff 0.0912, CI [0.0581, 0.1244])

| family | n | both | only_t | only_b | neither |
| --- | --- | --- | --- | --- | --- |
| rule_induction | 400 | 54 | 59 | 55 | 232 |
| compositional_transfer | 400 | 11 | 75 | 6 | 308 |

Guardrails: {'diversity_floor': 0.13990508670301466, 'min_replay_diversity': 0.42241139594162136, 'non_collapse_floor': 0.02, 'rate_d1_verified': 0.29875, 'rate_d2_verified': 0.24875}

- [ ] hand-recomputed and matches (owner initials/date: ______)
