"""Typed errors with stable exit codes. The CLI never degrades silently."""

from __future__ import annotations


class HsiError(Exception):
    code = 1


class InvalidError(HsiError):
    code = 2


class DigestError(HsiError):
    code = 3


class ExistsError(HsiError):
    code = 4


class BudgetError(HsiError):
    code = 6


class LockError(HsiError):
    code = 7


class NotVerifiedError(HsiError):
    code = 8


class PreconditionError(HsiError):
    code = 9


class RecomputeError(HsiError):
    code = 10
