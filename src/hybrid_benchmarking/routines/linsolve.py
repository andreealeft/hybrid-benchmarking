"""The four functional quantum linear solvers.

Each prepares a state proportional to the solution of ``Ax = b`` for a d-sparse
Hermitian A, and each is costed in calls to the same pair of sparse-access
oracles.  That shared denominator is the whole point of the comparison: it is
hardware-free, and it compares quantum against quantum without a clock.

They do not all reach the matrix the same way:

- **HHL** and **Fourier** evolve it, so they pay for Hamiltonian simulation --
  and pay it through qubitization, never through Berry's construction.
- **Chebyshev** reaches it through a quantum walk, ``Q[U] = 2(4 j0) Q[P_A]``,
  and never simulates anything.
- **QSVT** reaches it through a block encoding.

The dominant parameter is the condition number, which enters the simulation
time directly.  All four hold only while the precision sits far below the
inverse condition number; that is declared, and stepping outside it warns.

Two readings in the sources are not self-consistent and were settled by the
author:

*j0* is the truncation degree of the Chebyshev series approximating 1/x.
Equation (B.74) of the thesis writes the same quantity ``s`` as
``d k log2(d k / eps)`` in one slot and ``d^2 k^2 log2(d k / eps)`` in the
other.  The quadratic form is correct -- it agrees with Lemma 16, with the
Childs-Kothari-Somma truncation it derives from, and with Lemma 1 of the
quantum interior point paper, which reuses this result verbatim.

*n_1/x* is defined in Lemma 17 with three arguments and called with two.  The
``d / 2k`` precision rescaling in the definition is kept: it encodes the
block-encoding sub-normalisation used when block-encoding ``A / d``.
"""

from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import Implementation, Routine, register, single
from ..validity import Validity, definition, much_less_than
from .amplification import qaa_overhead

# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------

#: Constant in HHL's simulation time, from the error analysis of Lemma 14.
C_F = 2 ** 7 + 22 * math.pi ** 2 + (64 + 14 * math.pi ** 2) ** 2 / math.pi ** 2


def _segments(t_prime: float, epsilon: float) -> int:
    """Qubitization segments; the numeric twin of :func:`hamsim.segments`."""
    crossover = math.log(1.0 / epsilon) / math.e
    if t_prime >= crossover:
        return math.ceil(math.e * t_prime)
    return math.ceil(
        4.0 * math.log(1.0 / epsilon)
        / math.log(math.e + math.log(1.0 / epsilon) / t_prime)
    )


def simulation_queries(d: float, norm_max: float, t: float,
                       epsilon: float) -> float:
    """Oracle calls to simulate exp(-iAt) by qubitization."""
    return 48.0 * _segments(d * norm_max * t, epsilon)


def _validity() -> Validity:
    """The regime every functional solver's query count was derived in."""
    return Validity((
        definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
        definition(sp.Ge(S.d, 1), "sparsity is at least 1"),
        definition(sp.Ge(S.kappa, 1), "condition number is at least 1"),
        definition(sp.Gt(S.norm_x, 0), "the solution has a non-zero norm"),
        much_less_than(S.epsilon, 1 / S.kappa),
    ))


def _provenance(source: str, *extra: str) -> Provenance:
    return Provenance.of(
        Bound.EXACT, Derivation.ANALYTIC, source,
        assumptions=(
            "queries to the sparse-access oracles dominate the cost",
            "preparing the right-hand side is not counted",
            "no quantum error correction overhead",
        ) + extra,
    )


# ---------------------------------------------------------------------------
# HHL
# ---------------------------------------------------------------------------

_HHL = "Harrow-Hassidim-Lloyd; Lemma 14 of the thesis (Appendix B.3)"


def hhl_time(kappa: float, epsilon: float) -> float:
    return math.sqrt(C_F) * kappa / epsilon


def hhl_queries(d: float, kappa: float, epsilon: float, norm_x: float,
                norm_max: float = 1.0) -> float:
    """``Q[HHL] = 2 n_QAA Q[e^{-iAt}]``.

    The factor of two is the uncomputation: Algorithm 22 undoes the phase
    estimation after the conditioned rotation, so each application evolves the
    matrix twice.
    """
    inner = simulation_queries(d, norm_max, hhl_time(kappa, epsilon), epsilon)
    p0 = 1.0 / (4.0 * kappa ** 2)
    p = norm_x ** 2 / (4.0 * kappa ** 2)
    return 2.0 * qaa_overhead(p=min(p, 1.0), p0=min(p0, p)) * inner


# ---------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------

_FOURIER = "Childs-Kothari-Somma Fourier approach; Lemma 15 of the thesis " \
           "(Appendix B.4)"


def fourier_time(kappa: float, epsilon: float) -> float:
    return 2.0 * math.sqrt(2.0) * kappa * math.log(1.0 + 8.0 * kappa / epsilon)


def fourier_alpha(kappa: float, epsilon: float) -> float:
    """The linear-combination weight, which sets the success probability.

    Summed directly rather than approximated: the summand peaks at
    ``l dz ~ 1`` and the number of terms stays modest even for large condition
    numbers.
    """
    log_term = math.log(1.0 + 8.0 * kappa / epsilon)
    delta_z = (2.0 * math.pi / (kappa + 1.0)) / math.sqrt(log_term)
    L = int(math.floor((kappa + 1.0) / math.pi * log_term))
    total = 0.0
    for l in range(1, L + 1):
        u = l * delta_z
        total += u * math.exp(-(u ** 2) / 2.0)
    return 4.0 * math.sqrt(math.pi) * kappa / (kappa + 1.0) * total


def fourier_queries(d: float, kappa: float, epsilon: float, norm_x: float,
                    norm_max: float = 1.0) -> float:
    inner = simulation_queries(d, norm_max, fourier_time(kappa, epsilon), epsilon)
    alpha = fourier_alpha(kappa, epsilon)
    p0 = 1.0 / alpha ** 2
    p = norm_x ** 2 / alpha ** 2
    return qaa_overhead(p=min(p, 1.0), p0=min(p0, p)) * inner


def oaa_rounds(alpha: float) -> float:
    """Applications of the inner routine under oblivious amplitude amplification.

    ``2m + 1`` with ``m = floor(pi / 4 theta)`` and ``sin theta = 1 / alpha``.
    This is the fixed schedule of Lemma 33 -- the success probability is known
    in advance here, so no geometric search is needed.
    """
    if alpha < 1:
        raise ValueError(
            "alpha = {} is below one, so 1/alpha is not a sine".format(alpha)
        )
    return math.pi / (2.0 * math.asin(1.0 / alpha)) + 1.0


def fourier_gates(d: float, kappa: float, epsilon: float, norm_1: float,
                  norm_max: float) -> float:
    """``G[QLS_F]`` of Lemma 7 -- the gate count the quantum simplex is built on.

    Composes exactly: oblivious amplification rounds times the gate cost of one
    Hamiltonian simulation, taken from **Berry's** construction rather than
    qubitization.  That substitution is the whole reason routines carry named
    implementations.
    """
    from .hamsim import berry_gates

    t = fourier_time(kappa, epsilon)
    alpha = fourier_alpha(kappa, epsilon)
    return oaa_rounds(alpha) * berry_gates(
        epsilon=epsilon, d=d, t=t, norm_1=norm_1, norm_max=norm_max
    )


# ---------------------------------------------------------------------------
# Chebyshev
# ---------------------------------------------------------------------------

_CHEBYSHEV = "Childs-Kothari-Somma Chebyshev approach; Lemma 16 of the thesis " \
             "(Appendix B.5)"

#: Above this many Chebyshev terms the binomial tail is summed in closed form
#: rather than term by term.  The two agree to better than a part in 10^3 well
#: below the crossover; see the tests.
_EXACT_TAIL_LIMIT = 2000


def chebyshev_degree_parameter(y: float, epsilon: float) -> int:
    """``s = ceil((d kappa)^2 log2(d kappa / eps))``, with ``y = d kappa``."""
    return math.ceil(y ** 2 * math.log2(y / epsilon))


def chebyshev_terms(y: float, epsilon: float) -> int:
    """``j0 = ceil(sqrt(s log2(4 s / eps)))`` -- the truncation degree.

    Quadratic in ``y`` inside ``s``, per Lemma 16 rather than (B.74); see the
    module docstring.
    """
    s = chebyshev_degree_parameter(y, epsilon)
    return math.ceil(math.sqrt(s * math.log2(4.0 * s / epsilon)))


def _binomial_upper_tail(s: int, j: int) -> float:
    """``P(X >= s + j + 1)`` for ``X ~ Bin(2s, 1/2)``."""
    if j >= s:
        return 0.0
    if s <= _EXACT_TAIL_LIMIT:
        log_denominator = 2 * s * math.log(2.0)
        log_top = math.lgamma(2 * s + 1)
        return sum(
            math.exp(log_top - math.lgamma(k + 1) - math.lgamma(2 * s - k + 1)
                     - log_denominator)
            for k in range(s + j + 1, 2 * s + 1)
        )
    return 0.5 * math.erfc((j + 0.5) / math.sqrt(s))


def chebyshev_alpha(d: float, y: float, epsilon: float) -> float:
    """``alpha = (4 / d 2^{2s}) sum_j sum_i binom(2s, s+i)``.

    The inner double sum is the expected positive deviation of a fair binomial,
    ``E[max(X - s, 0)]``.  For the values of ``s`` that arise here -- routinely
    above 10^4 -- that is far beyond exact summation of binomials, so beyond
    :data:`_EXACT_TAIL_LIMIT` it is evaluated as ``sqrt(s / pi) / 2``, the
    normal limit.  The truncation at ``j0`` never binds: ``j0`` grows like
    ``sqrt(s log s)`` while the deviation has scale ``sqrt(s)``.
    """
    s = chebyshev_degree_parameter(y, epsilon)
    j0 = chebyshev_terms(y, epsilon)
    if s <= _EXACT_TAIL_LIMIT:
        upper = min(j0, s - 1)
        total = sum(_binomial_upper_tail(s, j) for j in range(upper + 1))
    else:
        total = 0.5 * math.sqrt(s / math.pi)
    return 4.0 * total / d


def chebyshev_queries(d: float, kappa: float, epsilon: float,
                      norm_x: float) -> float:
    """``Q[QLS_C] = 8 j0 n_QAA`` -- no Hamiltonian simulation anywhere."""
    y = d * kappa
    j0 = chebyshev_terms(y, epsilon)
    alpha = chebyshev_alpha(d, y, epsilon)
    p0 = 1.0 / alpha ** 2
    p = norm_x ** 2 / alpha ** 2
    return 8.0 * j0 * qaa_overhead(p=min(p, 1.0), p0=min(p0, p))


def binkowski_chebyshev_queries(sparsity: float, kappa: float, epsilon: float,
                                n_qaa: float = 1.0) -> float:
    """Lemma 1 of the quantum interior point paper, stated in its own terms.

    Written out independently so the two papers can be checked against each
    other rather than assumed to agree.  His ``s`` is the sparsity, this
    library's ``d``; his ``kappa^2`` is squared inside the same bracket.
    """
    inner = math.ceil(sparsity ** 2 * kappa ** 2
                      * math.log2(sparsity * kappa / epsilon))
    return 8.0 * math.ceil(
        math.sqrt(inner * math.log2(4.0 / epsilon * inner))
    ) * n_qaa


# ---------------------------------------------------------------------------
# QSVT
# ---------------------------------------------------------------------------

_QSVT = "Gilyen et al. quantum singular value transform; Lemma 17 of the " \
        "thesis (Appendix B.6)"


def _n_exp(beta: float, epsilon: float) -> int:
    return math.ceil(math.sqrt(
        2.0 * math.log(4.0 / epsilon)
        * math.ceil(max(beta * math.e ** 2, math.log(2.0 / epsilon)))
    ))


def _n_rect(k: float, epsilon: float) -> int:
    return 2 * _n_exp(2.0 * k ** 2, math.sqrt(math.pi) * epsilon / (16.0 * k)) + 1


def _n_inverse(y: float, epsilon: float, d: float, kappa: float) -> int:
    """``n_1/x``, keeping the ``d / 2 kappa`` precision rescaling.

    That factor encodes the block-encoding sub-normalisation used when the
    matrix is block-encoded as ``A / d``.  Lemma 17 states the definition with
    three arguments and calls it with two; this is the reading the author
    confirmed.
    """
    return 2 * chebyshev_terms(y, (d / (2.0 * kappa)) * epsilon) + 1


def qsvt_queries(d: float, kappa: float, epsilon: float,
                 norm_x: float) -> float:
    y = d * kappa
    inner_precision = epsilon / (4.0 * kappa)
    rect_precision = min(
        inner_precision,
        y / (2.0 * chebyshev_terms(y, inner_precision)),
    )
    per_application = (
        4.0 * _n_rect(1.0 / y, rect_precision)
        + 4.0 * _n_inverse(y, inner_precision, d, kappa)
    )
    p = norm_x ** 2 / (4.0 * kappa ** 2)
    p0 = ((1.0 - epsilon / 2.0) / (2.0 * kappa)) ** 2
    return qaa_overhead(p=min(p, 1.0), p0=min(p0, p)) * per_application


# ---------------------------------------------------------------------------
# display forms
# ---------------------------------------------------------------------------

_n_qaa = sp.Function("n_QAA")(S.norm_x, S.kappa, S.d, S.epsilon)
_sim = sp.Function("Q_sim")(S.d, S.norm_A_max, S.kappa, S.epsilon)
_j0 = sp.Function("j_0")(S.d, S.kappa, S.epsilon)

HHL_EXPR = 2 * _n_qaa * _sim
FOURIER_EXPR = _n_qaa * _sim
CHEBYSHEV_EXPR = 8 * _j0 * _n_qaa
QSVT_EXPR = _n_qaa * (
    4 * sp.Function("n_rect")(S.d, S.kappa, S.epsilon)
    + 4 * sp.Function("n_inv")(S.d, S.kappa, S.epsilon)
)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

HHL = single(
    name="HHL",
    summary="Phase-estimation linear solver with a sine-window clock state and "
            "a rotation conditioned on the inverse eigenvalue.",
    citation=_HHL,
    built_from=("P_b", "QPE", "HamSim/qubitization", "QAA"),
    costs={
        Unit.QUERIES: Cost(
            expr=HHL_EXPR,
            unit=Unit.QUERIES,
            provenance=_provenance(
                _HHL,
                "the simulation time carries HHL's 1/eps dependence, not log(1/eps)",
            ),
            validity=_validity(),
            kernel=lambda v: hhl_queries(
                d=v["d"], kappa=v["kappa"], epsilon=v["epsilon"],
                norm_x=v["x_norm"], norm_max=v["A_max"],
            ),
        ),
    },
)

_FOURIER_GATES = "Lemma 7 of the thesis, used throughout the quantum simplex " \
                 "gate counts (Appendix A.3.6)"

QLS_Fourier = register(Routine(
    name="QLS-Fourier",
    summary="Inverse approximated by a Fourier sum of evolution unitaries, "
            "applied by linear combination of unitaries.",
    implementations=(
        Implementation(
            name="via-qubitization",
            summary="Evolution by Low-Chuang qubitization. Costed in oracle "
                    "queries, for the four-way comparison of solvers.",
            citation=_FOURIER,
            built_from=("P_b", "LCU", "HamSim/qubitization", "OAA"),
            costs={
                Unit.QUERIES: Cost(
                    expr=FOURIER_EXPR,
                    unit=Unit.QUERIES,
                    provenance=_provenance(_FOURIER),
                    validity=_validity(),
                    kernel=lambda v: fourier_queries(
                        d=v["d"], kappa=v["kappa"], epsilon=v["epsilon"],
                        norm_x=v["x_norm"], norm_max=v["A_max"],
                    ),
                ),
            },
        ),
        Implementation(
            name="via-berry",
            summary="Evolution by Berry et al.'s fractional-query reduction. "
                    "Costed in gates, because Nannicini's setting fixes the "
                    "oracle implementation. This is the solver inside every "
                    "quantum simplex subroutine.",
            citation=_FOURIER_GATES,
            built_from=("P_b", "LCU", "HamSim/berry", "OAA"),
            costs={
                Unit.GATES: Cost(
                    expr=sp.Function("m_OAA")(S.kappa, S.epsilon)
                    * sp.Function("G_sim")(S.d, S.kappa, S.epsilon,
                                           S.norm_A_1, S.norm_A_max),
                    unit=Unit.GATES,
                    provenance=Provenance.of(
                        Bound.LOWER, Derivation.ANALYTIC, _FOURIER_GATES,
                        assumptions=(
                            "one Toffoli costs one gate",
                            "state preparation is not counted",
                            "non-leading terms dropped, in favour of the "
                            "quantum side",
                        ),
                    ),
                    validity=_validity(),
                    kernel=lambda v: fourier_gates(
                        d=v["d"], kappa=v["kappa"], epsilon=v["epsilon"],
                        norm_1=v["A_1"], norm_max=v["A_max"],
                    ),
                ),
            },
        ),
    ),
))

QLS_Chebyshev = single(
    name="QLS-Chebyshev",
    summary="Inverse approximated by Chebyshev polynomials realised as a "
            "quantum walk. Queries the oracles directly -- no simulation.",
    citation=_CHEBYSHEV,
    built_from=("P_b", "LCU", "QAA"),
    costs={
        Unit.QUERIES: Cost(
            expr=CHEBYSHEV_EXPR,
            unit=Unit.QUERIES,
            provenance=_provenance(
                _CHEBYSHEV,
                "j0 quadratic in d kappa, per Lemma 16; (B.74) drops the square",
            ),
            validity=_validity(),
            kernel=lambda v: chebyshev_queries(
                d=v["d"], kappa=v["kappa"], epsilon=v["epsilon"],
                norm_x=v["x_norm"],
            ),
        ),
    },
)

QLS_QSVT = single(
    name="QLS-QSVT",
    summary="Inverse via singular value transformation with a matrix-inversion "
            "polynomial applied to a block encoding.",
    citation=_QSVT,
    built_from=("P_b", "QAA"),
    costs={
        Unit.QUERIES: Cost(
            expr=QSVT_EXPR,
            unit=Unit.QUERIES,
            provenance=_provenance(
                _QSVT,
                "n_1/x keeps the d/2kappa rescaling from its definition",
            ),
            validity=_validity(),
            kernel=lambda v: qsvt_queries(
                d=v["d"], kappa=v["kappa"], epsilon=v["epsilon"],
                norm_x=v["x_norm"],
            ),
        ),
    },
)
