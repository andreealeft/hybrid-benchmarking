"""Hamiltonian simulation by qubitization.

Shared by every linear solver that reaches the matrix through time evolution,
which is why it lives here rather than inside any one of them.

The cost is a closed form, so unlike the amplification schedule it needs no
numeric kernel -- the expression is both what the reader sees and what gets
evaluated.
"""

from __future__ import annotations

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import Routine, register
from ..validity import Validity, definition

_SOURCE = "Low-Chuang qubitization; Lemma 12 of the thesis"

#: Rescaled evolution time.  The oracles see the matrix through its sparsity
#: and largest entry, so those enter the cost only in this combination.
t_rescaled = S.d * S.norm_A_max * S.t_sim


def segments(t_prime: sp.Expr, precision: sp.Expr) -> sp.Expr:
    """Number of simulation segments for a rescaled time and precision.

    Long evolutions cost linearly in time; short ones are dominated by the
    precision, and the cost flattens into a logarithmic regime.
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


QUERIES_EXPR = 48 * segments(t_rescaled, S.epsilon)


HamSim = register(Routine(
    name="HamSim",
    summary="Simulate exp(-iAt) for a d-sparse Hamiltonian to precision "
            "epsilon, by qubitization.",
    citation=_SOURCE,
    built_from=("O_F", "O_A"),
    costs={
        Unit.QUERIES: Cost(
            expr=QUERIES_EXPR,
            unit=Unit.QUERIES,
            provenance=Provenance.of(
                Bound.EXACT, Derivation.ANALYTIC, _SOURCE,
                assumptions=(
                    "queries to the sparse-access oracles dominate the cost",
                    "no error correction overhead",
                ),
            ),
            validity=Validity((
                definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
                definition(sp.Ge(S.d, 1), "sparsity is at least 1"),
            )),
        ),
    },
))
