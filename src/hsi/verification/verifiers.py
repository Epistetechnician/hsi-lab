"""Imperfect verifiers for self-generated hypotheses (ADR 0003).

A verifier decides which proposals of generation g enter the next training set. Three kinds:

- ``visible``    keep a proposal that parses and re-executes on every observed transition
                 (the verifier of packets v8-v10).
- ``noisy``      the visible check, then flip its REJECT verdict to accept with probability
                 ``fpr`` for proposals that parse but contradict the evidence (a false accept).
                 Unparseable text is always rejected (a parser does not make mistakes).
- ``tested``     the visible check AND agreement with the true rule on ``extra_probes`` fresh
                 oracle probe states (a test suite strengthened beyond the visible evidence, as
                 EvalPlus strengthens HumanEval).
- ``dose``       NOT a deployable verifier (ADR 0008, E9). Sets the probe-correct
                 share ``p`` of a fixed-size kept set of ``k`` visible-consistent proposals, using
                 ground truth to choose WHICH consistent proposals enter. It exists to identify the
                 precision dose-response with kept-set size held constant.
- ``confidence`` no check against the world at all: keep a parseable proposal whose mean token
                 log-probability under the generating model is at least ``tau`` (self-certainty).

Every verdict is a pure function of (spec, generation, sample key, proposal text, score): no
shared mutable RNG, so reruns, resumed runs and replicate chains are exactly reproducible.

Ground truth is read ONLY to audit a kept set (precision) and to score evaluation; it never
decides a verdict. Probes are fresh random states on which a hypothesis is compared with the true
rule; a proposal that agrees on all of them is behaviourally correct ("probe-correct").
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hsi.agents.learner import _run
from hsi.contracts.records import Split
from hsi.environment import RuleWorld
from hsi.environment.world import MOD, N_CELLS, Prim

KINDS = ("visible", "noisy", "tested", "confidence", "dose")
N_PROBES = 16


@dataclass(frozen=True)
class VerifierSpec:
    name: str
    kind: str
    fpr: float = 0.0
    tau: float | None = None
    extra_probes: int = 0  # tested: oracle probe states a proposal must also pass
    p: float | None = None  # dose: target probe-correct share of the kept set
    k: int = 0  # dose: kept-set size (the synthetic quota)
    n_synth: int | None = None  # visible: synthetic quota override (E10); None = half the examples
    salt: str = ""  # decorrelates the noise of replicate chains
    train_seed_offset: int = 0  # adapter-training seed = packet seed + offset + generation

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown verifier kind {self.kind!r}")
        if not 0.0 <= self.fpr <= 1.0:
            raise ValueError("fpr must be in [0, 1]")
        if self.kind == "confidence" and self.tau is None:
            raise ValueError("a confidence verifier needs tau")
        if self.kind != "noisy" and self.fpr:
            raise ValueError("fpr applies to noisy verifiers only")
        if self.kind == "dose":
            if self.p is None or not 0.0 <= self.p <= 1.0 or self.k <= 0:
                raise ValueError("a dose verifier needs p in [0, 1] and k > 0")
        elif self.p is not None or self.k:
            raise ValueError("p and k apply to dose verifiers only")
        if self.n_synth is not None and (self.kind != "visible" or self.n_synth < 0):
            raise ValueError("n_synth >= 0 applies to visible verifiers only")
        if (self.kind == "tested") != (self.extra_probes > 0):
            raise ValueError("extra_probes > 0 exactly for tested verifiers")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerifierSpec:
        return cls(
            name=str(d["name"]),
            kind=str(d["kind"]),
            fpr=float(d.get("fpr", 0.0)),
            tau=None if d.get("tau") is None else float(d["tau"]),
            extra_probes=int(d.get("extra_probes", 0)),
            p=None if d.get("p") is None else float(d["p"]),
            k=int(d.get("k", 0)),
            n_synth=None if d.get("n_synth") is None else int(d["n_synth"]),
            salt=str(d.get("salt", "")),
            train_seed_offset=int(d["train_seed_offset"]),
        )


def specs_of(params: dict[str, Any]) -> list[VerifierSpec]:
    """The packet's verifier chains, validated: unique names and unique training seeds."""
    specs = [VerifierSpec.from_dict(d) for d in params.get("verifier_chains", [])]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("verifier chain names must be unique")
    if len({s.train_seed_offset for s in specs}) != len(specs):
        raise ValueError("verifier chains must use distinct training seeds")
    return specs


def noise_draw(spec: VerifierSpec, seed: int, gen: int, key: str, text: str) -> float:
    """Uniform [0,1) draw tied to this exact (chain, generation, sample, text) and nothing else."""
    return random.Random(f"verifier/{seed}/{spec.name}/{spec.salt}/{gen}/{key}/{text}").random()


def accepts(
    spec: VerifierSpec,
    seed: int,
    gen: int,
    key: str,
    text: str,
    *,
    parses: bool,
    consistent: bool,
    mean_logprob: float | None = None,
    extra_ok: bool | None = None,
) -> bool:
    if not parses:
        return False
    if spec.kind in ("visible", "dose"):  # dose: bucket quotas are applied by DoseSelector
        return consistent
    if spec.kind == "tested":
        return consistent and bool(extra_ok)  # extra_ok is only computed for consistent text
    if spec.kind == "noisy":
        return consistent or noise_draw(spec, seed, gen, key, text) < spec.fpr
    assert mean_logprob is not None and spec.tau is not None
    return mean_logprob >= spec.tau


# -- ground truth (audit and evaluation only) ----------------------------------------------------


def world_of(key: str) -> tuple[str, Split, int, int]:
    """(family, split, seed, action) from a sample key ``<family>-<split>-<seed>/a<action>``."""
    head, act = key.rsplit("/a", 1)
    fam, split, seed = head.rsplit("-", 2)
    return fam, Split(split), int(seed), int(act)


def true_rule(key: str) -> tuple[Prim, ...]:
    fam, split, seed, a = world_of(key)
    return RuleWorld(fam, split, seed).task.actions[a]


def probe_states(tag: str, k: int = N_PROBES) -> list[tuple[int, ...]]:
    rng = random.Random(f"probe/{tag}")
    return [tuple(rng.randrange(MOD) for _ in range(N_CELLS)) for _ in range(k)]


def probes_agree(h: Sequence[Prim] | None, truth: Sequence[Prim], tag: str, k: int) -> bool:
    """Agreement with the true rule on the first k states of probe set ``tag``."""
    if h is None:
        return False
    return all(_run(tuple(h), s) == _run(tuple(truth), s) for s in probe_states(tag)[:k])


def probe_correct(h: Sequence[Prim] | None, truth: Sequence[Prim], tag: str) -> bool:
    """True iff hypothesis ``h`` maps every probe state exactly as the true rule does."""
    if h is None:
        return False
    return all(_run(tuple(h), s) == _run(tuple(truth), s) for s in probe_states(tag))


# -- dose-response selection (E9) ------------------------------------------------------------------


def dose_plan(k: int, p: float) -> tuple[int, int]:
    """(n probe-correct, n consistent-but-wrong) for a kept set of k at target share p."""
    n_correct = round(p * k)
    return n_correct, k - n_correct


class DoseSelector:
    """Fills two quotas from a stream of visible-consistent proposals, in stream order.

    ``offer(correct)`` returns True if the proposal enters the kept set. Deterministic in the
    order of offers; the caller supplies a seeded order. Stops being needed once ``full``.
    """

    def __init__(self, spec: VerifierSpec) -> None:
        assert spec.kind == "dose" and spec.p is not None
        self.want_correct, self.want_wrong = dose_plan(spec.k, spec.p)
        self.n_correct = 0
        self.n_wrong = 0

    def offer(self, correct: bool) -> bool:
        if correct and self.n_correct < self.want_correct:
            self.n_correct += 1
            return True
        if not correct and self.n_wrong < self.want_wrong:
            self.n_wrong += 1
            return True
        return False

    @property
    def size(self) -> int:
        return self.n_correct + self.n_wrong

    @property
    def full(self) -> bool:
        return self.n_correct == self.want_correct and self.n_wrong == self.want_wrong

    @property
    def realised_p(self) -> float:
        return self.n_correct / max(1, self.size)
