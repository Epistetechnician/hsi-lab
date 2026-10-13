"""Episode, message, relation, relation-event, and receipt schemas (strict, versioned)."""
from __future__ import annotations

from enum import Enum

from pydantic import Field

from hsi.contracts.packet import Outcome, Strict

EPISODE_V = "hsi-episode-v1"
RECEIPT_V = "hsi-receipt-v1"


class Origin(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    GENERATED = "generated"


class Eligibility(str, Enum):
    QUARANTINED = "quarantined"
    ELIGIBLE = "eligible"
    REJECTED = "rejected"


class Split(str, Enum):
    FIT = "fit"
    TUNE = "tune"
    HELDOUT = "heldout"


class Step(Strict):
    action: int
    agent: int = 0
    observation: list[int]
    reward: float


class EpisodeOutcome(Strict):
    success: bool
    steps: int = Field(ge=0)
    oracle_steps: int = Field(ge=0)


class Episode(Strict):
    schema_version: str = EPISODE_V
    episode_id: str
    env_version: str
    family: str
    rule_family: str
    split: Split
    seed: int
    steps: list[Step]
    outcome: EpisodeOutcome
    origin: Origin
    parent_ids: list[str]
    generation_depth: int = Field(ge=0)
    governance: str
    eligibility: Eligibility
    prev_digest: str
    digest: str


class Message(Strict):
    msg_id: str
    sender: int
    receivers: list[int]
    payload_codec: str
    bytes: int = Field(ge=0)
    evidence_refs: list[str]
    confidence: float = Field(ge=0, le=1)
    expires_at_step: int = Field(ge=0)


class RelState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    RETIRED = "retired"


class Relation(Strict):
    rel_id: str
    type: str
    endpoints: list[str] = Field(min_length=2, max_length=2)
    strength: float = Field(ge=0, le=1)
    context: str
    evidence_refs: list[str]
    uncertainty: float = Field(ge=0, le=1)
    valid_from: int
    valid_to: int | None
    state: RelState
    cost: float = Field(ge=0)


class RelOp(str, Enum):
    STRENGTHEN = "strengthen"
    DECOUPLE = "decouple"
    DORMANT = "dormant"
    REACTIVATE = "reactivate"
    RETIRE = "retire"
    CREATE = "create"
    MERGE = "merge"
    PRUNE = "prune"


class RelationEvent(Strict):
    event_id: str
    rel_id: str
    op: RelOp
    justification: str = Field(min_length=1)
    source_event_ids: list[str] = Field(min_length=1)


class FamilyResult(Strict):
    family: str
    arm: str
    n: int = Field(ge=1)
    successes: int = Field(ge=0)


class Receipt(Strict):
    schema_version: str = RECEIPT_V
    packet_digest: str
    source_commit: str
    lock_digest: str
    model_digest: str
    data_digest: str
    seeds: list[int]
    budget_used: dict[str, float]
    family_results: list[FamilyResult]
    paired_diff: float
    ci_low: float
    ci_high: float
    threshold: float
    guardrails_held: bool
    decision: Outcome
    nonclaims: list[str]
    receipt_digest: str
