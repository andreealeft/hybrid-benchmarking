"""Amplitude amplification, and everything that is secretly amplitude amplification.

Quantum search, quantum maximum finding, the post-selection in HHL, the
amplification step in the Fourier and Chebyshev linear solvers, knapsack search
and the search inside quantum breadth-first search are all the same shape:
prepare a state, mark the good subspace, amplify.

This module implements that once.  Everything else here is a wrapper that fixes
the success probability and its known lower bound.

Two published statements of the expected round count differ in one factor.  The
quantum-search form (Lemma 6 of the thesis) counts expected Grover *iterations*
and carries a factor of one half; the amplification form (Lemma 13) counts
applications of the amplified algorithm and does not.  The product term is
identical: ``1 - <p>`` in one is ``1/2 + sin(...)/(...)`` in the other.  Both
are derived here from :func:`rounds`, so the relationship stays visible rather
than being reimplemented twice with a discrepancy nobody notices.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import Routine, register
from ..validity import Validity, definition

#: Schedule growth factor.  Any 1 < c < 2 works; the literature uses 6/5.
LAMBDA = sp.Rational(6, 5)
_LAMBDA = float(LAMBDA)


# --------------------------------------------------------------------------
# the one kernel
# --------------------------------------------------------------------------

def rounds(
    p: float,
    p0: float,
    c: float = _LAMBDA,
    k_max: Optional[int] = None,
    tol: float = 1e-15,
    hard_cap: int = 100_000,
) -> float:
    """Expected number of applications of the amplified algorithm.

    ``p`` is the true success probability; ``p0`` is the lower bound the
    algorithm is actually given.  The gap between them is the whole cost of not
    knowing how well you are doing: the schedule has to grow blindly, capped at
    ``sqrt(1/p0)``.

    With ``k_max`` the sum is truncated after that many rounds, which is what
    the quantum-search statement does; without it, the sum runs until the
    remaining mass is below ``tol``.
    """
    if not 0 < p <= 1:
        raise ValueError("success probability p must lie in (0, 1], got {}".format(p))
    if not 0 < p0 <= 1:
        raise ValueError("lower bound p_0 must lie in (0, 1], got {}".format(p0))
    if p0 > p:
        raise ValueError(
            "p_0 = {} is not a lower bound on p = {}".format(p0, p)
        )
    if p >= 1 - 1e-12:
        return 1.0

    theta = math.asin(math.sqrt(p))
    sin_2theta = math.sin(2 * theta)
    cap = math.sqrt(1.0 / p0)

    total = 0.0
    remaining = 1.0  # product over previous rounds having all failed
    k = 1
    while True:
        m = math.floor(min(c ** k, cap))
        total += m * remaining

        # Probability that round k also fails, carried into the next product.
        success_k = 0.5 - math.sin(4 * (m + 1) * theta) / (
            4 * (m + 1) * sin_2theta
        )
        failure_k = min(1.0, max(0.0, 1.0 - success_k))
        remaining *= failure_k

        if k_max is not None:
            if k >= k_max:
                break
        elif remaining < tol or k >= hard_cap:
            break
        k += 1

    return total


def qsearch_k_max(list_length: float) -> int:
    """Round budget used by the quantum-search statement."""
    length = float(list_length)
    if length <= 1:
        return 1
    denominator = 2.0 * math.sqrt(length - 1.0)
    ratio = length / denominator
    if ratio <= 1.0:
        return 4
    return int(math.ceil(math.log(ratio) / math.log(_LAMBDA))) + 4


def qsearch_iterations(list_length: float, marked: float) -> float:
    """Expected Grover iterations to find one of ``marked`` elements."""
    return 0.5 * rounds(
        p=float(marked) / float(list_length),
        p0=1.0 / float(list_length),
        k_max=qsearch_k_max(list_length),
    )


def qsearch_all_iterations(list_length: float, marked: float) -> float:
    """Expected iterations to find *all* marked elements.

    Each success removes one element from the list, so the cost is the sum over
    successively smaller instances.  Used by quantum breadth-first search, where
    every vertex in a layer has to be found -- missing one yields an incomplete
    layered graph and the wrong maximum flow.
    """
    length, count = float(list_length), int(marked)
    return sum(
        qsearch_iterations(length - i, count - i) for i in range(count)
    )


def qaa_overhead(p: float, p0: float) -> float:
    """Expected applications needed to boost success probability to O(1)."""
    return rounds(p=p, p0=p0)


# --------------------------------------------------------------------------
# symbolic forms, for display
# --------------------------------------------------------------------------

_k, _l = sp.symbols("k l", integer=True, positive=True)


def _round_term(m_index: sp.Expr, theta: sp.Expr, cap: sp.Expr) -> sp.Expr:
    m_l = sp.floor(sp.Min(LAMBDA ** m_index, cap))
    return sp.Rational(1, 2) + sp.sin(4 * (m_l + 1) * theta) / (
        4 * (m_l + 1) * sp.sin(2 * theta)
    )


def _sum_form(theta: sp.Expr, cap: sp.Expr, upper: sp.Expr,
              half: bool) -> sp.Expr:
    m_k = sp.floor(sp.Min(LAMBDA ** _k, cap))
    body = m_k * sp.Product(_round_term(_l, theta, cap), (_l, 1, _k - 1))
    if half:
        body = body / 2
    return sp.Sum(body, (_k, 1, upper))


_theta_search = sp.asin(sp.sqrt(S.t / S.X))
_k_max_search = sp.ceiling(
    sp.log(S.X / (2 * sp.sqrt(S.X - 1))) / sp.log(LAMBDA)
) + 4

QSEARCH_EXPR = _sum_form(_theta_search, sp.sqrt(S.X), _k_max_search, half=True)

_theta_qaa = sp.asin(sp.sqrt(S.p))
QAA_EXPR = _sum_form(_theta_qaa, sp.sqrt(1 / S.p0), sp.oo, half=False)


# --------------------------------------------------------------------------
# registry entries
# --------------------------------------------------------------------------

_SEARCH_SOURCE = "Boyer-Brassard-Hoyer-Tapp schedule; Lemma 6 of the thesis"
_QAA_SOURCE = "Lemma 13 of the thesis"

QSearch = register(Routine(
    name="QSearch",
    summary="Unstructured search for one of t marked elements in a list of X; "
            "returns expected Grover iterations.",
    citation=_SEARCH_SOURCE,
    costs={
        Unit.ITERATIONS: Cost(
            expr=QSEARCH_EXPR,
            unit=Unit.ITERATIONS,
            provenance=Provenance.of(
                Bound.EXACT, Derivation.ANALYTIC, _SEARCH_SOURCE,
                assumptions=("the number of marked elements t is known",),
            ),
            validity=Validity((
                definition(sp.Ge(S.X, 2), "list must hold at least 2 elements"),
                definition(sp.Ge(S.t, 1), "at least one marked element"),
                definition(sp.Le(S.t, S.X),
                           "cannot mark more elements than the list holds"),
            )),
            kernel=lambda v: qsearch_iterations(v["X"], v["t"]),
        ),
    },
))

QSearchAll = register(Routine(
    name="QSearchAll",
    summary="Find every one of the t marked elements, by repeated search over a "
            "shrinking list.",
    citation="Lemma 2 of the quantum breadth-first search study",
    built_from=("QSearch",),
    costs={
        Unit.ITERATIONS: Cost(
            expr=sp.Sum(
                QSEARCH_EXPR.subs({S.X: S.X - _k, S.t: S.t - _k}),
                (_k, 0, S.t - 1),
            ),
            unit=Unit.ITERATIONS,
            provenance=Provenance.of(
                Bound.EXACT, Derivation.ANALYTIC,
                "Lemma 2 of the quantum breadth-first search study",
                assumptions=("t is known, having been computed classically",),
            ),
            validity=Validity((
                definition(sp.Ge(S.X, 2), "list must hold at least 2 elements"),
                definition(sp.Le(S.t, S.X),
                           "cannot mark more elements than the list holds"),
            )),
            kernel=lambda v: qsearch_all_iterations(v["X"], v["t"]),
        ),
    },
))

QAA = register(Routine(
    name="QAA",
    summary="Amplitude amplification: expected applications of the amplified "
            "algorithm needed to reach constant success probability.",
    citation=_QAA_SOURCE,
    costs={
        Unit.ITERATIONS: Cost(
            expr=QAA_EXPR,
            unit=Unit.ITERATIONS,
            provenance=Provenance.of(
                Bound.EXACT, Derivation.ANALYTIC, _QAA_SOURCE,
                assumptions=(
                    "only the lower bound p_0 is available to the algorithm",
                ),
            ),
            validity=Validity((
                definition(sp.Le(S.p0, S.p), "p_0 must lower-bound the true p"),
                definition(sp.Le(S.p, 1), "p is a probability"),
            )),
            kernel=lambda v: qaa_overhead(v["p"], v["p_0"]),
        ),
    },
))


def amplify(inner: Cost, p: sp.Expr, p0: sp.Expr) -> Cost:
    """Wrap a cost in amplitude amplification.

    ``p`` and ``p0`` are expressions in the caller's parameters -- for a linear
    solver, typically the solution norm over the condition number, and the
    bound the algorithm is handed in its place.
    """
    overhead = QAA.cost(Unit.ITERATIONS)
    overhead = Cost(
        expr=overhead.expr.subs({S.p: p, S.p0: p0}),
        unit=overhead.unit,
        provenance=overhead.provenance,
        validity=overhead.validity,
        kernel=_substituted_kernel(p, p0),
    )
    return overhead * inner


def _substituted_kernel(p: sp.Expr, p0: sp.Expr):
    p_fn = sp.lambdify(sorted(p.free_symbols, key=lambda s: s.name), p, "math")
    p0_fn = sp.lambdify(sorted(p0.free_symbols, key=lambda s: s.name), p0, "math")
    p_names = [s.name for s in sorted(p.free_symbols, key=lambda s: s.name)]
    p0_names = [s.name for s in sorted(p0.free_symbols, key=lambda s: s.name)]

    def kernel(values: Dict[str, float]) -> float:
        return qaa_overhead(
            p_fn(*[values[n] for n in p_names]),
            p0_fn(*[values[n] for n in p0_names]),
        )

    return kernel
