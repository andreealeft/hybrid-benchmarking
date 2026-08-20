"""Cade-style instrumented counts, kept deliberately apart.

QUBRABENCH and its successor annotate a *classical* algorithm and count how
often it calls a subroutine that a quantum version would accelerate.  That
count is then converted into an expected quantum cost.  It is a genuine and
useful measurement, and it is not the same quantity as anything else in this
library.

The hazard is entirely one of naming.  Both traditions say "query".  Here a
query is a call to a sparse-access oracle, derived in closed form from a
lemma; there it is a call to a subroutine the caller supplied, observed by
running the classical algorithm.  The two are not comparable, cannot be added,
and would be indistinguishable in a table if they shared a word -- which is a
worse kind of duplication than repeated code, because nothing would ever
surface it.

So they get their own unit.  :class:`~hybrid_benchmarking.provenance.Unit`
``SUBROUTINE_CALLS`` will not add to ``QUERIES``, and the cost algebra raises
rather than silently producing a number.

These entries used to carry no formulas at all, on the stated grounds that
supplying them would mean transcribing results this library had not read.  That
reason expired when the results were read: the four constructions below are
reimplemented from QUBRABENCH's ``algorithms/`` modules and, through them, from
the papers those cite.  As everywhere else here, the repository was a fixture to
check against rather than a source to copy.

**All four count in ``SUBROUTINE_CALLS``, including the linear solver**, and
that last one is the decision worth stating.  ``Cade-linalg`` rests on Dalzell's
Theorem 1, which is a closed-form count of queries to a *block encoding* of the
matrix.  Our four functional solvers count queries to *sparse-access* oracles.
Both are honestly called queries; they are different access models, and a table
putting Dalzell's number beside ``QLS-QSVT``'s would be comparing costs
denominated in different things.  Landing it in the other unit is what stops
that, which is exactly what the unit is for.

One consequence worth knowing: the inputs here are not problem sizes but
observations.  ``X`` and ``t`` are the search space and the marked count *as a
particular classical run encountered them*, so a cost from this family is
:class:`~hybrid_benchmarking.provenance.Derivation.LOGGED` in spirit even when
the formula around them is exact.
"""

from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import single
from ..validity import Validity, definition

_CADE = ("Cade, Folkertsma, Niesen and Weggemans, quantifying Grover speed-ups "
         "beyond asymptotic analysis (arXiv:2203.04975); reimplemented from "
         "QUBRABENCH's algorithms/search.py and algorithms/max.py")
_BHMT = ("Brassard, Hoyer, Mosca and Tapp, quantum amplitude amplification and "
         "estimation (arXiv:quant-ph/0005055), Theorem 12; reimplemented from "
         "QUBRABENCH's algorithms/amplitude.py")
_DALZELL = ("Dalzell, a shortcut to an optimal quantum linear system solver "
            "(arXiv:2305.11352), Theorem 1; reimplemented from QUBRABENCH's "
            "algorithms/linalg.py")

_INSTRUMENTED = (
    "counted by instrumenting a classical run: the search space and the marked "
    "count are what one execution encountered, not a property of the problem",
    "a call here is to a subroutine the caller supplied, never to a "
    "sparse-access oracle, and the two must not be compared",
)

#: Two queries to the supplied oracle build one phase oracle, which is what the
#: amplified routine actually calls.  QUBRABENCH applies this at the call site
#: rather than inside its formulas; it is applied here, so the number a reader
#: sees is the number of calls that happen.
PHASE_ORACLE_CALLS = 2

#: QUBRABENCH clamps the condition number up to this before costing a solve,
#: because Dalzell's expression is not meaningful below it.
_LEAST_KAPPA = math.sqrt(12.0)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cade_F(space: float, marked: float) -> float:
    """``F`` of equation (3): the expected Grover cost of one search attempt.

    Constant outside ``1 <= t < X/4``, because beyond a quarter of the space
    being marked there is nothing left for amplification to buy.
    """
    space, marked = float(space), float(marked)
    if 1 <= marked < space / 4.0:
        term = space / (2.0 * math.sqrt((space - marked) * marked))
        return 4.5 * term + math.ceil(math.log(term) / math.log(6.0 / 5.0)) - 3.0
    return 2.0344


def search_quantum_calls(space: float, marked: float, failure: float,
                         budget: float) -> float:
    """Expected quantum calls made by Cade's search, after ``budget`` classical ones.

    With nothing marked the routine cannot stop early and pays the full
    schedule -- the same shape as the terminating search this library's own
    ``FindRow`` pays at ``r = 0``, reached independently.
    """
    space, marked = float(space), float(marked)
    if marked <= 0:
        return 9.2 * math.ceil(math.log(1.0 / failure) / math.log(3.0)) \
            * math.sqrt(space)
    inner = cade_F(space, marked)
    return ((1.0 - marked / space) ** budget * inner
            * (1.0 + 1.0 / (1.0 - inner / (9.2 * math.sqrt(space)))))


def search_classical_calls(space: float, marked: float, budget: float) -> float:
    """Expected classical calls made before the quantum part begins."""
    space, marked = float(space), float(marked)
    if marked <= 0:
        return float(budget)
    return (space / marked) * (1.0 - (1.0 - marked / space) ** budget)


def search_worst_case_calls(space: float, failure: float) -> float:
    """Lemma 8: the worst case, by the Zalka variant rather than the expectation."""
    rounds = math.ceil(math.log(1.0 / failure) / (2.0 * math.log(4.0 / 3.0)))
    return 5.0 * rounds + math.pi * math.sqrt(float(space) * rounds)


# ---------------------------------------------------------------------------
# maximum finding
# ---------------------------------------------------------------------------

def max_quantum_calls(space: float, failure: float) -> float:
    """Corollary 1: maximum finding, summed over the rank of each new maximum.

    The sum runs from one, which is Cade's own indexing.  Note that this
    library's quantum simplex sums the same shape from zero, following (A.25)
    and Lemma 25 -- a different lemma with a different terminating term, and the
    two are not in conflict.  The cost is linear in the search space because
    every rank contributes, so a large instance takes a noticeable moment.
    """
    count = int(space)
    total = sum(cade_F(count, rank) / (rank + 1.0) for rank in range(1, count))
    return math.ceil(math.log(1.0 / failure) / math.log(3.0)) * 3.0 * total


# ---------------------------------------------------------------------------
# amplitude estimation
# ---------------------------------------------------------------------------

def amplitude_rounds(value: float, precision: float, failure: float) -> float:
    """Theorem 12: rounds of amplitude estimation to resolve ``a`` to ``eps``.

    The failure probability enters as a repetition count and stops mattering
    once it is loose enough for a single round, which is the ``8/pi^2``
    threshold below.
    """
    if failure >= 1.0 - 8.0 / math.pi ** 2:
        repeats = 1
    else:
        repeats = 1 + int(math.ceil(0.5 / failure))
    spread = math.sqrt(value * (1.0 - value))
    return math.ceil(repeats * math.pi
                     / (math.sqrt(precision + value * (1.0 - value)) - spread))


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------

def dalzell_queries(alpha: float, kappa: float, epsilon: float) -> float:
    """Theorem 1: queries to a block encoding of ``A`` for one solve.

    Three terms, none of them dominant across the whole range, which is why it
    is carried whole rather than reduced to its leading behaviour.
    """
    kappa = max(float(kappa), _LEAST_KAPPA)
    first = ((1741.0 * alpha * math.e / 500.0) * math.sqrt(kappa ** 2 + 1.0)
             * ((133.0 / 125.0 + 4.0 / (25.0 * kappa ** (1.0 / 3.0)))
                * math.pi * math.log(2.0 * kappa + 3.0) + 1.0))
    second = ((351.0 / 50.0) * math.log(2.0 * kappa + 3.0) ** 2
              * (math.log(451.0 * math.log(2.0 * kappa + 3.0) ** 2 / epsilon)
                 + 1.0))
    third = alpha * kappa * math.log(32.0 / epsilon)
    return first + second + third


def dalzell_queries_with_failure(alpha: float, kappa: float, epsilon: float,
                                 failure: float) -> float:
    """The same, repeated until the failure probability is met.

    One run succeeds with probability at least ``0.39 - 0.201 eps``; asking for
    better than that costs repetitions, and asking for worse costs nothing.
    """
    once = dalzell_queries(alpha, kappa, epsilon)
    succeeds = 0.39 - 0.201 * epsilon
    if 1.0 - failure <= succeeds:
        return once
    return once * math.log(failure) / math.log(1.0 - succeeds)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _instrumented(expr: sp.Expr, kernel, source: str, *notes: str,
                  validity: Validity = None) -> Cost:
    return Cost(
        expr=expr,
        unit=Unit.SUBROUTINE_CALLS,
        provenance=Provenance.of(
            Bound.UPPER, Derivation.LOGGED, source,
            assumptions=_INSTRUMENTED + notes,
        ),
        validity=validity if validity is not None else Validity(),
        kernel=kernel,
    )


def _searchable() -> Validity:
    return Validity((
        definition(sp.Ge(S.X, 1), "a search space holds at least one element"),
        definition(sp.Le(S.t, S.X),
                   "cannot mark more elements than the space holds"),
        definition(sp.Lt(S.failure, 1), "a failure probability is below one"),
    ))


CadeSearch = single(
    name="Cade-search",
    summary="Search a classically-defined predicate over a collection, with "
            "the classical and would-be-quantum call counts tracked side by "
            "side. Counted by instrumenting a run, in subroutine calls -- "
            "which are not sparse-access oracle queries and must never be "
            "compared with them.",
    citation=_CADE,
    costs={
        Unit.SUBROUTINE_CALLS: _instrumented(
            sp.Function("Q_search")(S.X, S.t, S.failure, S.classical_budget),
            lambda v: PHASE_ORACLE_CALLS * search_quantum_calls(
                v["X"], v["t"], v["eta"], v["K"]),
            _CADE,
            "the expected count of equation (3), not the worst case of Lemma 8",
            "two supplied-oracle calls per phase oracle",
            validity=_searchable(),
        ),
    },
)

CadeMax = single(
    name="Cade-max",
    summary="Maximum finding over a classically-defined key function. Counted "
            "by instrumenting a run, in subroutine calls.",
    citation=_CADE,
    built_from=("Cade-search",),
    costs={
        Unit.SUBROUTINE_CALLS: _instrumented(
            sp.ceiling(sp.log(1 / S.failure) / sp.log(3)) * 3
            * sp.Sum(sp.Function("F")(S.X, S.t) / (S.t + 1), (S.t, 1, S.X - 1)),
            lambda v: PHASE_ORACLE_CALLS * max_quantum_calls(v["X"], v["eta"]),
            _CADE,
            "the rank sum runs from one, as Corollary 1 indexes it",
            "two supplied-oracle calls per phase oracle",
            validity=Validity((
                definition(sp.Ge(S.X, 2), "there is nothing to maximise over"),
                definition(sp.Lt(S.failure, 1),
                           "a failure probability is below one"),
            )),
        ),
    },
)

CadeAmplitude = single(
    name="Cade-amplitude",
    summary="Amplitude estimation of a classically-defined event. Counted by "
            "instrumenting a run, in subroutine calls.",
    citation=_BHMT,
    costs={
        Unit.SUBROUTINE_CALLS: _instrumented(
            sp.ceiling(sp.Function("k")(S.failure) * sp.pi
                       / (sp.sqrt(S.epsilon + S.amplitude * (1 - S.amplitude))
                          - sp.sqrt(S.amplitude * (1 - S.amplitude)))),
            lambda v: PHASE_ORACLE_CALLS * amplitude_rounds(
                v["a"], v["epsilon"], v["eta"]),
            _BHMT,
            "two supplied-oracle calls per phase oracle",
            validity=Validity((
                definition(sp.Le(S.amplitude, 1), "an amplitude is a probability"),
                definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
                definition(sp.Lt(S.failure, 1),
                           "a failure probability is below one"),
            )),
        ),
    },
)

CadeLinalg = single(
    name="Cade-linalg",
    summary="A linear solve instrumented in the same style, costed in queries "
            "to a block encoding. Deliberately not in the same unit as the "
            "functional solvers, which count sparse-access oracle queries -- a "
            "different access model wearing the same word.",
    citation=_DALZELL,
    costs={
        Unit.SUBROUTINE_CALLS: _instrumented(
            sp.Function("Q_star")(S.subnormalisation, S.kappa, S.epsilon)
            * sp.Function("n_repeat")(S.failure, S.epsilon),
            lambda v: dalzell_queries_with_failure(
                v["alpha"], v["kappa"], v["epsilon"], v["eta"]),
            _DALZELL,
            "queries to a block encoding of A, which are not the sparse-access "
            "queries the four functional solvers count",
            "the condition number is raised to sqrt(12) where it falls below "
            "it, as QUBRABENCH does, since the expression has no meaning there",
            validity=Validity((
                definition(sp.Ge(S.kappa, 1), "condition number is at least 1"),
                definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
                definition(sp.Lt(S.failure, 1),
                           "a failure probability is below one"),
            )),
        ),
    },
)
