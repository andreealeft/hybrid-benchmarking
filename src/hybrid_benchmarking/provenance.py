"""What kind of number a cost is.

Every count this library produces carries three things besides its value: the
unit it is measured in, how tight it is, and how it was arrived at.  Without
them it is possible -- and in this field, routine -- to place a lower bound
next to an extrapolation and read the pair as a like-for-like comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple


class Unit(Enum):
    """What is being counted.

    ``ITERATIONS`` and ``REPETITIONS`` are multipliers: they carry no cost of
    their own, but multiply the cost of whatever is being repeated.  That is
    what lets ``iterations x gates-per-iteration`` come out in gates.
    """

    GATES = "gates"
    QUERIES = "oracle queries"
    CYCLES = "cycles"
    ITERATIONS = "iterations"
    REPETITIONS = "repetitions"

    @property
    def is_multiplier(self) -> bool:
        return self in (Unit.ITERATIONS, Unit.REPETITIONS)

    def __str__(self) -> str:
        return self.value


class Bound(Enum):
    """How tight the number is."""

    EXACT = "exact"
    LOWER = "lower bound"
    UPPER = "upper bound"
    ESTIMATE = "estimate"

    def __str__(self) -> str:
        return self.value


class Derivation(Enum):
    """How the number was arrived at, ordered by how much it is hedged."""

    ANALYTIC = "analytic"
    LOGGED = "logged from a classical run"
    SIMULATED = "simulated"
    EXTRAPOLATED = "extrapolated"
    ASSUMED = "assumed"

    def __str__(self) -> str:
        return self.value


# Increasing order of caveat.  Combining two derivations keeps the weaker one:
# a closed-form expression evaluated on extrapolated inputs is extrapolated.
_CAVEAT_ORDER = (
    Derivation.ANALYTIC,
    Derivation.LOGGED,
    Derivation.SIMULATED,
    Derivation.EXTRAPOLATED,
    Derivation.ASSUMED,
)


def combine_bound(a: Bound, b: Bound) -> Bound:
    """Bound direction of a sum or product of two non-negative costs.

    Adding or multiplying two lower bounds gives a lower bound; mixing a lower
    with an upper gives neither, only an estimate.
    """
    if a is b:
        return a
    pair = {a, b}
    if Bound.ESTIMATE in pair:
        return Bound.ESTIMATE
    if pair == {Bound.EXACT, Bound.LOWER}:
        return Bound.LOWER
    if pair == {Bound.EXACT, Bound.UPPER}:
        return Bound.UPPER
    return Bound.ESTIMATE  # {LOWER, UPPER}


def combine_derivation(a: Derivation, b: Derivation) -> Derivation:
    return max(a, b, key=_CAVEAT_ORDER.index)


def _merge(a: Tuple[str, ...], b: Tuple[str, ...]) -> Tuple[str, ...]:
    """Concatenate preserving order, dropping duplicates."""
    seen = []
    for item in tuple(a) + tuple(b):
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


@dataclass(frozen=True)
class Provenance:
    """Where a number came from and what it is worth."""

    bound: Bound = Bound.EXACT
    derivation: Derivation = Derivation.ANALYTIC
    sources: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        bound: Bound = Bound.EXACT,
        derivation: Derivation = Derivation.ANALYTIC,
        source: str = "",
        assumptions: Iterable[str] = (),
    ) -> "Provenance":
        """Convenience constructor taking a single source string."""
        return cls(
            bound=bound,
            derivation=derivation,
            sources=(source,) if source else (),
            assumptions=tuple(assumptions),
        )

    def combine(self, other: "Provenance") -> "Provenance":
        return Provenance(
            bound=combine_bound(self.bound, other.bound),
            derivation=combine_derivation(self.derivation, other.derivation),
            sources=_merge(self.sources, other.sources),
            assumptions=_merge(self.assumptions, other.assumptions),
        )

    def assuming(self, *assumptions: str) -> "Provenance":
        """Return a copy carrying additional assumptions."""
        return Provenance(
            bound=self.bound,
            derivation=self.derivation,
            sources=self.sources,
            assumptions=_merge(self.assumptions, tuple(assumptions)),
        )

    def describe(self) -> str:
        parts = ["{}, {}".format(self.bound, self.derivation)]
        if self.sources:
            parts.append("after " + "; ".join(self.sources))
        if self.assumptions:
            parts.append("assuming " + "; ".join(self.assumptions))
        return " -- ".join(parts)
