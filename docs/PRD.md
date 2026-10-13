# HSI Lab PRD

Oct 12, 2026 · @Shaan

## Problem, goal and non-goals

Build `hsi-lab`, a new local-first repository that tests whether verified synthetic experience, compact experience codecs, small agent collectives, and an adaptive relationship layer improve held-out adaptation at matched cost. Each mechanism is a separate, falsifiable experiment with a reproducible evidence receipt.

**Problem.** Claims about self-improving or recursive AI rarely separate what was verified from what was generated, or compare against matched-budget baselines. We need a harness where every training example has lineage, every result has a locked evaluator, and a negative result is a first-class output.

**Goal for v1 (about 20 weeks).** One end-to-end synthetic experiment loop on a single Apple Silicon workstation: procedural environment, verified synthesis, three codec arms, baseline and adaptive collectives, a relationship ledger, one reversible adapter update, and an independently recomputed receipt for each experiment E1–E7.

**Non-goals for v1.**

- No real-user, licensed, or sensor data.
- No hosted model endpoints, paid compute, Linux/CUDA clusters, or telemetry.
- No base-model updates, automatic promotion, deployment, or publication.
- No single "intelligence score" and no claims of general intelligence, understanding, or novelty.
- No NeMo RL, Curator, or DataDesigner; they stay on a deferred list with explicit adoption triggers.

## Users and core workflows

Three users share one CLI, `hsi`, and one experiment-packet format.

| User | Needs | Primary commands |
| --- | --- | --- |
| Research lead (owner) | Approve scope, thresholds, claim ceiling; decide promotion. | `hsi packet freeze`, `hsi decide` |
| Implementation agent | Build modules, run pilots and confirmatory runs within a packet. | `hsi run`, `hsi pilot` |
| Evaluation engineer | Lock held-out cohorts, recompute receipts, audit lineage. | `hsi lock`, `hsi verify-receipt` |

Core workflow:

1. Write and freeze an experiment packet: hypothesis, null, arms, primary metric, threshold, guardrails, budget, seeds, stop rules, claim ceiling.
2. Run a pilot on fit and tune families to estimate variance; compute sample size.
3. Lock the held-out cohort under a separate custody root the producer cannot read.
4. Run all arms at matched budget; the run writes episodes, relationship events, and resource records.
5. The locked evaluator scores held-out families and emits a receipt.
6. An independent validator recomputes the receipt from aggregates.
7. The owner records PROMOTE, NO\_CANDIDATE, or INCONCLUSIVE. Any change after step 1 is a new packet.

## System design

Eleven modules in one Python package, `src/hsi/`, with write permissions enforced by process and directory boundaries rather than convention.

| Module | Responsibility | Writes | Must never read |
| --- | --- | --- | --- |
| `contracts` | JSON Schemas and typed models for episodes, messages, relations, packets, receipts. | Schemas only | — |
| `environment` | Seeded procedural rule worlds, task families, solvability oracle. | Episodes | Held-out answer keys |
| `experience` | Append-only, digest-chained episode ledger with lineage and eligibility. | Ledger | — |
| `codec` | Text/trajectory, typed-event, learned-latent encoders and decoders. | Encoded payloads | Held-out data |
| `agents` | Model wrappers (MLX/PyTorch) and scripted baseline policies. | Actions, messages | Evaluator source |
| `collective` | Independent, centralized, fixed-graph, adaptive-graph coordination with byte accounting. | Messages | Held-out data |
| `relations` | Typed relationship records; strengthen, decouple, dormant, reactivate, retire. | Relation events | Held-out data |
| `synthesis` | Task and trajectory proposals. | Quarantine queue | Held-out seeds, verifier, promotion |
| `verification` | Environment oracle and rule checks; moves proposals to eligible or rejected. | Eligibility decisions | Generator state |
| `learning` | Isolated LoRA or external-memory candidates on a frozen base. | Candidate snapshots | Held-out data, base weights (write) |
| `evaluation` + `assurance` | Locked scoring, power analysis, receipts, independent recomputation. | Receipts | Producer implementation |

Data flow: environment → experience ledger → codec → agents/collective → synthesis → verification → learning → locked evaluation → receipt → owner decision. `relations` reads only verified ledger events and feeds routing and retrieval in `collective` and `agents`.

Storage: Parquet/Arrow tables and SQLite index under a run root outside Git; model weights and held-out keys in a separate owner-only custody root.

## Functional requirements

P0 is required for v1; P1 is in v1 if time allows; P2 is deferred.

| ID | Module | Requirement | Priority |
| --- | --- | --- | --- |
| F1 | environment | Grid/symbolic world with hidden rule combinations, seeded reset/step, exact transitions, solvability oracle. | P0 |
| F2 | environment | Six task families: rule induction, compositional transfer, counterevidence, partial observation, collective discovery, relational return. | P0 |
| F3 | environment | Splits by rule family, generator version, and seed family; disjointness test. | P0 |
| F4 | contracts | Versioned schemas; strict ingress validation rejecting unknown, duplicate, or missing fields. | P0 |
| F5 | experience | Append-only digest-chained ledger; each record has ID, lineage, origin (observed/inferred/generated), governance, eligibility. | P0 |
| F6 | codec | Text and typed-event arms behind one interface; byte count includes schema and provenance. | P0 |
| F7 | codec | Learned discrete or continuous latent arm, same interface. | P1 |
| F8 | synthesis | Proposal generator writes only to quarantine; records generation depth and parent IDs. | P0 |
| F9 | verification | Oracle re-executes each proposal; rejects invalid, ambiguous, or provenance-free items with a reason. | P0 |
| F10 | collective | Independent and centralized baselines; fixed graph; message schema with evidence ref, confidence, expiry, bytes. | P0 |
| F11 | collective | Adaptive graph that forms and drops links under a byte budget. | P1 |
| F12 | relations | Records with type, strength, context, evidence, uncertainty, temporal validity, activation state, cost. | P0 |
| F13 | relations | Strengthen on held-out-free utility gain; dormant over delete; retire only on counterevidence; shared-source agreement counted once. | P0 |
| F14 | relations | Structural ops: create, merge, prune with logged justification. | P1 |
| F15 | learning | LoRA adapter or external-memory candidate on frozen base; candidate bound to manifest, seeds, runtime. | P0 |
| F16 | evaluation | Locked evaluator in separate process; regenerates held-out cohort from committed seed; computes paired family-level intervals. | P0 |
| F17 | evaluation | Power analysis from pilot variance before held-out lock. | P0 |
| F18 | assurance | Receipt binding source, lock, model, data, seeds, budget, thresholds, outcomes, nonclaims; validator recomputes without producer imports. | P0 |
| F19 | all | Resource accounting: tokens, calls, wall time, peak memory, bytes, retries, failed proposals. | P0 |
| F20 | evaluation | Experiments E1–E7 runnable from packets. | P0 (E1–E4, E6), P1 (E5, E7 structural arm) |
| F21 | environment | Adapters for Procgen, Crafter/Craftax, Melting Pot. | P2 |

## Experiment specifications

Each experiment is one frozen packet. All arms share seeds, task families, and a matched budget (tokens, calls, wall time, bytes). Primary metric is held-out success rate per task family. The comparison is paired and made at family level with a 95% bootstrap interval. The practical threshold is frozen after the pilot.

| ID | Hypothesis | Arms | Primary metric | Key guardrail |
| --- | --- | --- | --- | --- |
| E1 | Typed-event or latent codecs beat text at equal bytes | text, typed-event, latent (P1) | held-out success at a fixed byte budget | no arm wins on reconstruction loss alone |
| E2 | Verified synthetic experience beats unverified and real-only | real-only, unverified synthetic, verified synthetic | held-out success after equal example count | rare-family accuracy does not drop |
| E3 | A small collective beats one agent at matched compute | single, independent ×N, centralized, fixed graph | collective-discovery family success | message bytes counted in budget |
| E4 | One reversible adapter update improves held-out adaptation | frozen base, LoRA candidate, external-memory candidate | compositional-transfer success | no regression on fit families beyond 2 points |
| E5 | An adaptive graph beats a fixed graph under a byte cap | fixed graph, adaptive graph | success per kilobyte | link churn reported |
| E6 | Recursive synthesis does not collapse over generations | depth 1, 2, 3 with and without original replay | success at each depth | diversity of generated tasks stays above pilot floor |
| E7 | Relationship dynamics help on return and counterevidence | no relations, static relations, adaptive relations (structural ops P1) | relational-return and counterevidence success | shared-source agreement counted once |

Every experiment has three possible outcomes. PROMOTE means the interval clears the threshold and every guardrail holds. NO\_CANDIDATE means the interval excludes the threshold or a guardrail fails. INCONCLUSIVE means the interval straddles the threshold at the planned sample size; a retry needs a new packet.

## Data contracts

All records are versioned JSON Schema with Pydantic models. Ingress rejects any record with unknown, duplicate, or missing fields.

- **Episode:** `episode_id`, `env_version`, `family`, `split`, `seed`, `steps[]`, `outcome`, `origin` (observed/inferred/generated), `parent_ids[]`, `generation_depth`, `eligibility`, `prev_digest`, `digest`.
- **Message:** `msg_id`, `sender`, `receivers[]`, `payload_codec`, `bytes`, `evidence_refs[]`, `confidence`, `expires_at_step`.
- **Relation:** `rel_id`, `type`, `endpoints`, `strength`, `context`, `evidence_refs[]`, `uncertainty`, `valid_from`, `valid_to`, `state` (active/dormant/retired), `cost`.
- **Relation event:** `event_id`, `rel_id`, `op` (strengthen/decouple/dormant/reactivate/retire/create/merge/prune), `justification`, `source_event_ids[]`.
- **Packet:** hypothesis, null, arms, primary metric, threshold, guardrails, budget, seeds, stop rules, claim ceiling, `packet_digest`.
- **Receipt:** `packet_digest`, source commit, lock digest, model and data digests, seeds, budget actually used, per-family outcomes, intervals, decision, nonclaims, `receipt_digest`.

## CLI specification

| Command | Effect | Fails if |
| --- | --- | --- |
| `hsi packet new <exp>` | Writes a draft packet from a template | experiment ID unknown |
| `hsi packet freeze <file>` | Validates and stamps a packet digest; packet becomes read-only | schema invalid or threshold missing |
| `hsi pilot <packet>` | Runs fit and tune families; writes variance estimates and a sample size | packet not frozen |
| `hsi lock <packet>` | Writes the held-out seed commitment into the custody root | pilot missing or caller can read custody |
| `hsi run <packet>` | Runs all arms at matched budget | budget, digest, or split mismatch |
| `hsi eval <run>` | Locked evaluator scores held-out families; emits a receipt | lock digest mismatch |
| `hsi verify-receipt <receipt>` | Recomputes aggregates without importing producer code | any recomputed value differs |
| `hsi decide <receipt> <outcome>` | Records the owner's decision | receipt not verified |

All commands exit non-zero with a typed error code. They never fall back to a degraded mode.

## Repository layout

```
hsi-lab/
  pyproject.toml, uv.lock
  src/hsi/{contracts,environment,experience,codec,agents,collective,relations,synthesis,verification,learning,evaluation,assurance,cli}/
  validator/          # standalone recomputation, no src/hsi imports
  packets/            # frozen packets, one per experiment version
  schemas/            # exported JSON Schemas
  tests/{contract,property,integration}/
  docs/{decisions,experiments}/
```

Run roots and custody roots live outside the repo and are git-ignored. Paths come from one config file, and the CLI refuses any path inside the repo.

## Testing and acceptance

- **Contract tests:** every schema round-trips; malformed records are rejected with the expected error.
- **Property tests (Hypothesis):** environment determinism, split disjointness, digest-chain integrity, and budget conservation across arms.
- **Isolation tests:** the generator process cannot open custody-root paths; the validator imports nothing from `src/hsi`.
- **Golden receipts:** one frozen tiny run per experiment whose receipt digest must not change without a recorded decision.
- **Fault injection:** a tampered digest, swapped split, or over-budget arm each stops the run with the right error.
- **Acceptance per milestone:** the gate in the milestone table, plus a green model-free suite in under 60 s.

## Non-functional requirements and stack

- **Determinism:** identical seed and config produce byte-identical episodes and receipts on the same machine; known nondeterministic ops are listed per run.
- **Isolation:** producer, verifier, trainer, and evaluator run as separate processes with separate writable roots.
- **Local-only:** no network during runs; no telemetry; model weights pre-materialized.
- **Performance:** model-free contract suite under 60 s; one pilot cell under 1 h on an M4 Max with 36 GB.
- **Auditability:** every training example and relationship change traceable to a verified source event.
- **Fail-closed:** any digest, schema, split, or budget mismatch stops the run with a typed error.

| Layer | Choice |
| --- | --- |
| Language | Python 3.12, `uv` lockfile |
| Environment API | Gymnasium-style reset/step |
| Models | MLX on Apple Silicon (primary); PyTorch for portability; small pinned open-weight checkpoint (open question) |
| Adapters | MLX LoRA or PEFT/LoRA |
| Contracts | JSON Schema + Pydantic |
| Storage | Parquet/Arrow, SQLite |
| Tracking | MLflow local, prompt capture off |
| Quality | Ruff, mypy strict, pytest, hypothesis for property tests |
| Deferred | NeMo RL (if E4 outgrows one machine), Curator (when real corpora arrive), DataDesigner (text task proposals only) |

## Milestones and exit gates

| Weeks | Milestone | Exit gate |
| --- | --- | --- |
| 0–2 | M0 Repo and packet | Repo scaffold, CI-equivalent local gates, packet schema, owner-approved scope. |
| 2–5 | M1 Environment + ledger | F1–F5, F19 pass; deterministic replay; split disjointness proven; scripted baselines recorded. |
| 5–8 | M2 Codecs (E1) | F6 passes; E1 run at matched bytes; no arm advanced on reconstruction alone. |
| 8–11 | M3 Verified synthesis (E2) | F8–F9; every eligible example traceable; generator provably blind to held-out data. |
| 11–14 | M4 Collective + relations (E3, E7) | F10, F12–F13; reactivation and counterevidence tasks scored at matched budget. |
| 14–18 | M5 Candidate update (E4, E6) | F15–F17; threshold and guardrails met on held-out, else NO\_CANDIDATE recorded. |
| 18–20 | M6 Reproduction | F18; clean-environment rerun matches receipts; assurance level reported. |

Gates are evidence, not dates: a failed gate ends that packet; a retry is a new packet.

## Success metrics, risks and open questions

v1 succeeds when all seven experiments produce a validated receipt, positive or negative, not when any mechanism wins.

| Metric | Target |
| --- | --- |
| Experiments with independently recomputed receipts | 7 of 7 |
| Training examples with full lineage | 100% |
| Held-out leakage incidents | 0 |
| Clean-rerun receipt match | 100% of aggregates |
| Practical-effect threshold (per experiment, frozen after pilot) | default +5 points success or −15% cost at matched quality |

| Risk | Mitigation |
| --- | --- |
| Recursive quality collapse | Lineage depth caps, original-replay arm, rare-family guardrails. |
| Verifier shares generator blind spots | Executable oracle only; separate process and code. |
| Correlated agent errors amplified | Evidence refs; shared-source agreement counted once; peer-loss tests. |
| Premature pruning of dormant links | Dormant state; retire only on counterevidence; relational-return family. |
| Toy results overclaimed | Claim ceiling and nonclaims in every receipt. |
| Scope creep into clusters or hosted APIs | Deferred-tool list with explicit triggers. |

Open questions:

- [x] Which pinned open-weight checkpoint (size, license) runs under MLX within 36 GB?
- [x] Is a model-free scripted agent enough for E1–E3, reserving the LLM for E4?
- [x] Who acts as the independent validator for M6?
- [x] Repository name and hosting: private GitHub repo `hsi-lab`?

**Proposed resolutions (owner to confirm):**

- **Checkpoint:** a 1–4B instruct model with an Apache-2.0 or MIT license in 4-bit MLX, for example Qwen2.5-3B-Instruct. It fits well within 36 GB and leaves room for LoRA training. Freeze the choice in M0 by digest.
- **Scripted agent for E1–E3:** yes. Scripted policies isolate codec, synthesis, and coordination effects from model variance, keep the contract suite model-free, and make the M1–M4 gates cheap. The LLM enters at E4, with an optional LLM replication arm for E3 at P1.
- **Independent validator:** a separate agent session that has only the `validator/` package and the receipts, with no access to producer source. A human spot-checks one receipt per experiment.
- **Repository:** private GitHub repo `hsi-lab`, local-first, no CI secrets.

---

## Implementation status (updated 2026-10-12)

| Milestone | Status | Notes |
| --- | --- | --- |
| M0 Repo and packet | done | scaffold, packet schema, `packet new/freeze/check`, ADR 0001 |
| M1 Environment + ledger | done (pending owner review) | F1–F5, F19; `hsi baselines`; oracle 100%, random/greedy well below |
| M2 Codecs (E1) | done (pending owner review) | F6, F16–F18 for E1; full `pilot→lock→run→eval→verify-receipt→decide` loop; see docs/experiments/E1.md |
| M3 Verified synthesis (E2) | done (pending owner review) | F8–F9; quarantine → oracle verifier → eligible/rejected; E2 result is NO_CANDIDATE, see docs/experiments/E2.md |
| M4 Collective + relations (E3, E7) | done (pending owner review) | F10, F12–F13; E3 PROMOTE, E7 INCONCLUSIVE; see docs/experiments/E3.md, E7.md |
| M5 Candidate update (E4, E6) | done (pending owner review) | F15–F17; E4 NO_CANDIDATE, E6 NO_CANDIDATE; see docs/experiments/E4.md, E6.md; MLX/LLM backbone not used |
| M6 | not started | |

Schema deviations from the PRD text: `Episode` adds `rule_family` (split is defined on it) and
`governance` (F5). `Receipt` carries `family_results` (successes/n per family and arm) so the
validator can recompute intervals from aggregates. Seed families: fit `[0,1e6)`, tune `[1e6,2e6)`,
held-out `[2e6,…)`; `RuleWorld` rejects a seed outside its split's range.

M2 deviations: (1) the interval is an analytic paired (McNemar-style) normal CI over family means,
not a bootstrap, so the validator recomputes it from aggregate paired counts; (2) the effect
threshold lives in the packet and is frozen before the pilot (the pilot only sets sample size);
(3) `run` exercises all arms on the tune split and enforces budgets; the locked evaluator
(`python -m hsi.evaluation.locked`, separate process) is the only reader of custody and scores
held-out; (4) custody isolation is owner-only file modes plus process separation, not an OS sandbox.

M3 deviations: receipts now carry `comparisons[]` (one per baseline) so multi-baseline experiments
share one format; the E1 receipt was regenerated. An "example" is one transition episode; synthetic
proposals are one-step trajectories from generator-chosen start states, with the claimed outcome
produced by the generator's own induced model. The verifier re-executes via the environment
oracle. If verification cannot fill the synthetic quota within `max_proposals`, the arm is topped up
with real data so example counts stay equal (reported as `topped_up`).

M4 deviations: (1) E3 uses two agents with complementary action rights; with N=2 the centralized
and fixed-graph topologies coincide, so those arms differ in message payload (raw typed log vs
learned-rule summary), not topology; true topology comparisons need N>=3 (E5). (2) `centralized`
runs but is exploratory: the decision compares fixed_graph with single and independent only.
(3) E7 relations are context->stored-transitions with a capacity of 2 active relations; "return"
tasks reuse hidden rules with a fresh start/goal (`RuleWorld(..., variant=1)`). (4) Pilot sample
sizes hit `max_n=400` (flagged `underpowered` in pilot.json) for E3 and E7.

M5 deviations: (1) **No LLM yet.** MLX and a checkpoint are not installed and downloading weights
needs owner approval (open question: checkpoint). The "frozen base + LoRA adapter" contract
(freeze, low-rank update, reversible, digest-bound, trained only on non-held-out data) is
implemented on a pure-Python hypothesis prior (`hsi.learning`, runtime tag `hsi-prior-v1`), so a
MLX backbone can later replace it behind the same interface. (2) Candidates are trained during
`hsi pilot` (producer side) on tune/fit seeds in disjoint ranges, then digest-verified by `run`
and `eval` (fail closed). (3) The pilot/run split offsets (`tune_offset`) keep pilot seeds out of
the training seeds. (4) E6 is framed as treatment d3_replay vs baseline d3_no_replay plus a
non-inferiority guardrail (d3_replay >= d1_replay - 2 points), because the decision rule is
"treatment beats baseline".
