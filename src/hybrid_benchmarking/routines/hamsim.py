"""Hamiltonian simulation, in both of the constructions this work uses.

These are two different algorithms for the same task, and they are costed in
different units because they were analysed for different purposes.  Nothing
below lets you mix them.

**berry** -- Berry et al.'s reduction to the fractional-query model
(Section 2.10, Appendix A.3.9, Lemma 34).  The d-sparse Hamiltonian is
decomposed into 1-sparse pieces by edge colouring, those into Hamiltonians with
eigenvalues 0 and pi, the evolution approximated by a Lie product formula, and
the result reduced to a fractional-query algorithm finished with oblivious
amplitude amplification.  This is what the **quantum simplex gate counts** are
built on -- the linear solver inside Nannicini's subroutines reaches the matrix
this way.

**qubitization** -- Low and Chuang's approach via block encoding and the
Jacobi-Anger expansion (Section 2.11, Appendix B.1, Lemma 12).  This is what
the **linear solver query counts** are built on.

Not every solver uses either.  HHL and the Fourier solver evolve the matrix and
so pay for simulation; the Chebyshev and QSVT solvers do not -- they reach the
matrix through a quantum walk and a block encoding respectively, querying the
sparse-access oracles directly.
"""

from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import Implementation, Routine, register
from ..validity import Validity, definition

# ---------------------------------------------------------------------------
# Low-Chuang qubitization -- query counts
# ---------------------------------------------------------------------------

_QUBITIZATION = "Low-Chuang qubitization; Lemma 12 of the thesis (Appendix B.1)"

#: Rescaled evolution time.  The oracles see the matrix through its sparsity and
#: largest entry, so those enter only in this combination.
t_rescaled = S.d * S.norm_A_max * S.t_sim


def segments(t_prime: sp.Expr, precision: sp.Expr) -> sp.Expr:
    """Simulation segments for a rescaled time and precision.

    Long evolutions cost linearly in time; short ones are dominated by the
    precision and the cost flattens into a logarithmic regime.
    """
    crossover = sp.log(1 / precision) / sp.E
    return sp.Piecewise(
        (sp.ceiling(sp.E * t_prime), t_prime >= crossover),
        (
            sp.ceiling(
                4 * sp.log(1 / precision)
                / sp.log(sp.E + sp.log(1 / precision) / t_prime)
            ),
            True,
        ),
    )


QUBITIZATION_QUERIES = 48 * segments(t_rescaled, S.epsilon)


# ---------------------------------------------------------------------------
# Berry et al. fractional-query -- gate counts
# ---------------------------------------------------------------------------

_BERRY = "Berry et al. fractional-query simulation; Lemma 34 of the thesis " \
         "(Appendix A.3.9)"


def precision_parameter(epsilon: float, d: float, t: float) -> float:
    """The discretisation scale gamma = eps / (sqrt(2) d^3 t)."""
    return epsilon / (math.sqrt(2.0) * d ** 3 * t)


def segment_precision(epsilon: float, d: float, t: float,
                      norm_max: float) -> float:
    """Precision demanded of each fractional-query segment.

    The factor of one third is there to absorb the oblivious amplitude
    amplification step.
    """
    gamma = precision_parameter(epsilon, d, t)
    return (1.0 / 3.0) * (1.0 / (6.0 * d ** 2 * math.ceil(norm_max / gamma))) \
        * (epsilon / (5.0 * gamma * t))


def queries_per_segment(eps_seg: float, cap: int = 10 ** 7) -> int:
    """Smallest integer w with e^w / w^w <= eps_seg^2 / 2.

    Equivalently the smallest w for which ``w log w - w >= log(2 / eps_seg^2)``.
    Solved by search rather than in closed form; w grows like log(1/eps_seg), so
    the search is short.
    """
    if not 0 < eps_seg < 1:
        raise ValueError(
            "segment precision must lie in (0, 1), got {}".format(eps_seg)
        )
    target = math.log(2.0 / eps_seg ** 2)
    w = 2
    while w < cap:
        if w * math.log(w) - w >= target:
            return w
        w += 1
    raise ValueError("no feasible segment count below {}".format(cap))


def berry_gates(epsilon: float, d: float, t: float, norm_1: float,
                norm_max: float) -> float:
    """Minimum gates to simulate exp(-iHt), following Lemma 34.

    ``G[e^{-iHt}] >= 5 t (||H||_1 - d^2 gamma) w G[F_Q]`` with the oracle itself
    lower bounded by ``G[F_Q] >= 2 (ceil(log2(||H||_1 / gamma - d^2)) - 1)``,
    taking one Toffoli as one gate.
    """
    gamma = precision_parameter(epsilon, d, t)
    eps_seg = segment_precision(epsilon, d, t, norm_max)
    w = queries_per_segment(eps_seg)

    bulk = norm_1 - d ** 2 * gamma
    inner = norm_1 / gamma - d ** 2
    if bulk <= 0 or inner <= 1:
        raise ValueError(
            "the 1-norm is too small for this sparsity and precision: the "
            "bound in Lemma 34 has gone non-positive"
        )
    oracle_gates = 2.0 * (math.ceil(math.log2(inner)) - 1)
    return 5.0 * t * bulk * w * oracle_gates


# The displayed formula.  ``w`` is determined by a search rather than a closed
# form, so it appears as an applied function of the parameters it depends on --
# which keeps it out of the free symbols the caller has to supply.
_gamma = S.epsilon / (sp.sqrt(2) * S.d ** 3 * S.t_sim)
_w = sp.Function("w")(S.epsilon, S.d, S.t_sim, S.norm_A_max)

BERRY_GATES = (
    10 * S.t_sim * _w * (S.norm_A_1 - S.d ** 2 * _gamma)
    * (sp.ceiling(sp.log(S.norm_A_1 / _gamma - S.d ** 2, 2)) - 1)
)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _berry_kernel(v: Dict[str, float]) -> float:
    return berry_gates(
        epsilon=v["epsilon"], d=v["d"], t=v["t_sim"],
        norm_1=v["A_1"], norm_max=v["A_max"],
    )


HamSim = register(Routine(
    name="HamSim",
    summary="Simulate exp(-iAt) for a d-sparse Hermitian A to precision epsilon.",
    implementations=(
        Implementation(
            name="qubitization",
            summary="Low-Chuang block encoding with the Jacobi-Anger expansion. "
                    "Costed in oracle queries; used by HHL and the Fourier solver.",
            citation=_QUBITIZATION,
            built_from=("O_F", "O_A"),
            costs={
                Unit.QUERIES: Cost(
                    expr=QUBITIZATION_QUERIES,
                    unit=Unit.QUERIES,
                    provenance=Provenance.of(
                        Bound.EXACT, Derivation.ANALYTIC, _QUBITIZATION,
                        assumptions=(
                            "queries to the sparse-access oracles dominate the cost",
                            "no quantum error correction overhead",
                        ),
                    ),
                    validity=Validity((
                        definition(sp.Lt(S.epsilon, 1),
                                   "precision must be below 1"),
                        definition(sp.Ge(S.d, 1), "sparsity is at least 1"),
                    )),
                ),
            },
        ),
        Implementation(
            name="berry",
            summary="Reduction to the fractional-query model via 1-sparse "
                    "decomposition and a Lie product formula, finished with "
                    "oblivious amplitude amplification. Costed in gates; used by "
                    "the quantum simplex.",
            citation=_BERRY,
            built_from=("OAA",),
            costs={
                Unit.GATES: Cost(
                    expr=BERRY_GATES,
                    unit=Unit.GATES,
                    provenance=Provenance.of(
                        Bound.LOWER, Derivation.ANALYTIC, _BERRY,
                        assumptions=(
                            "one Toffoli costs one gate",
                            "the logarithm counting control qubits is base 2",
                            "non-leading terms dropped, in favour of the quantum side",
                        ),
                    ),
                    validity=Validity((
                        definition(sp.Lt(S.epsilon, 1),
                                   "precision must be below 1"),
                        definition(sp.Ge(S.d, 1), "sparsity is at least 1"),
                        definition(sp.Le(S.norm_A_max, S.norm_A_1),
                                   "the largest entry cannot exceed the 1-norm"),
                    )),
                    kernel=_berry_kernel,
                ),
            },
        ),
    ),
))
