"""Experiment packet schema. Frozen packets are immutable; changes require a new packet."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "hsi-packet-v1"
EXPERIMENTS = {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Outcome(StrEnum):
    PROMOTE = "PROMOTE"
    NO_CANDIDATE = "NO_CANDIDATE"
    INCONCLUSIVE = "INCONCLUSIVE"


class Budget(Strict):
    max_tokens: int = Field(ge=0)
    max_calls: int = Field(ge=0)
    max_wall_seconds: int = Field(gt=0)
    max_message_bytes: int = Field(ge=0)


class Packet(Strict):
    schema_version: str
    experiment: str
    version: int = Field(ge=1)
    hypothesis: str = Field(min_length=1)
    null_hypothesis: str = Field(min_length=1)
    arms: list[str] = Field(min_length=2)
    primary_metric: str = Field(min_length=1)
    threshold: float
    guardrails: list[str] = Field(min_length=1)
    budget: Budget
    seeds: list[int] = Field(min_length=1)
    stop_rules: list[str] = Field(min_length=1)
    claim_ceiling: str = Field(min_length=1)
    nonclaims: list[str] = Field(min_length=1)
    params: dict[str, Any]

    @field_validator("schema_version")
    @classmethod
    def _schema(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        return v

    @field_validator("experiment")
    @classmethod
    def _exp(cls, v: str) -> str:
        if v not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {v}")
        return v

    @field_validator("arms", "seeds")
    @classmethod
    def _unique(cls, v: list[Any]) -> list[Any]:
        if len(set(v)) != len(v):
            raise ValueError("entries must be unique")
        return v


def canonical_digest(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
