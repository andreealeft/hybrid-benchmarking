"""Quantum interior point methods, and the readout that dominates them.

Structurally this is the quantum simplex with a different outer algorithm: a
classical linear-programming method that hands its inner linear system to a
quantum solver.  The solver is the Chebyshev one, and it is *the same result* --
Lemma 1 of the interior point paper cites Lefterovici et al. for it, and the
tests in ``test_linsolve.py`` assert that the two statements agree.  So equation
(10) is assembled here by calling that count rather than restating it.

It did restate it, and wrongly: the inner logarithm read ``log2(2/eps)`` where
both (10) and Lemma 1 have ``log2(gamma/eps)``, which understated the dominant
term by about a third at realistic difficulties.  The docstring claiming reuse
was right and the code was not, which is the argument for reuse over
restatement.

What is new is the step the thesis never costed.  A linear solver prepares a
state proportional to the solution; an interior point method needs the solution
as numbers, to form the next Newton system.  Reading a d-dimensional amplitude
vector out to precision epsilon takes on the order of ``d / epsilon^2``
repetitions of the entire solver, and no improvement to the solver touches
that.  It is the reason the negative result here is structural rather than a
matter of constant factors.

Two formulations of the Newton step are carried as separate constructions, since
they produce systems of different dimension and sparsity from the same linear
program, and the comparison between them is the point of the paper.
"""

from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import Implementation, Routine, register, single
from ..validity import Validity, definition
from .linsolve import binkowski_chebyshev_queries

_SOURCE = "Binkowski, practical lower bounds for hybrid quantum interior point " \
          "methods (arXiv:2604.24362)"

_BENEVOLENT = (
    "one oracle call costs one quantum cycle",
    "no amplitude amplification overhead",
    "iterative refinement converges in a single step",
    "the interior point method converges in a single iteration",
    "no quantum error correction overhead",
)


def tomography_repetitions(dimension: float, epsilon: float) -> float:
    """``(d - 1) / eps^2`` uses of the state preparation, in the copy-access model.

    Resolving ``d - 1`` independent real amplitudes to additive error epsilon
    needs at least this many preparations, whatever the tomography protocol.
    Without a reusable state-preparation unitary there is no way around it.

    The factor of eight that (10) carries belongs to the solver, not here: it is
    the constant in front of Lemma 1's query count.  The product is the same
    either way, but attributing it to the readout made the readout look four
    times worse than the source says it is.
    """
    if dimension < 2:
        raise ValueError("tomography of a one-dimensional state is vacuous")
    return (dimension - 1.0) / epsilon ** 2


def difficulty(sparsity: float, kappa: float) -> float:
    """``gamma = s kappa`` -- the paper's effective difficulty of a system."""
    return sparsity * kappa


def newton_system_cycles(dimension: float, sparsity: float, kappa: float,
                         epsilon: float) -> float:
    """Equation (10): cycles to solve one Newton system and read it out.

    The solver cost and the tomography overhead multiply: every repetition
    demanded by the readout runs the solver again.  So this is exactly
    :func:`tomography_repetitions` times the query count of the Chebyshev
    solver at the system's own difficulty -- which is what the module docstring
    has always claimed and what, until the paper was read against the code, it
    did not do.  The inner logarithm was ``log2(2/eps)`` where both (10) and
    Lemma 1 have ``log2(gamma/eps)``, understating the dominant term by around
    a third at realistic difficulties.

    Reusing :func:`~.linsolve.binkowski_chebyshev_queries` rather than restating
    it is the point: the two are the same lemma, and one copy of it cannot drift
    from the other.
    """
    return tomography_repetitions(dimension, epsilon) * \
        binkowski_chebyshev_queries(sparsity, kappa, epsilon)


def _cost(kernel, *notes) -> Cost:
    return Cost(
        expr=sp.Function("C_Newton")(S.N, S.d, S.kappa, S.epsilon),
        unit=Unit.CYCLES,
        provenance=Provenance.of(
            Bound.LOWER, Derivation.ANALYTIC, _SOURCE,
            assumptions=_BENEVOLENT + notes,
        ),
        validity=Validity((
            definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
            definition(sp.Ge(S.kappa, 1), "condition number is at least 1"),
            definition(sp.Ge(S.N, 2), "a system has at least two unknowns"),
        )),
        kernel=kernel,
    )


Tomography = single(
    name="Tomography",
    summary="Read a d-dimensional amplitude vector out as numbers, in the "
            "copy-access model. The structural bottleneck of every quantum "
            "linear solver used inside a classical algorithm.",
    citation=_SOURCE,
    costs={
        Unit.REPETITIONS: Cost(
            expr=(S.N - 1) / S.epsilon ** 2,
            unit=Unit.REPETITIONS,
            provenance=Provenance.of(
                Bound.LOWER, Derivation.ANALYTIC, _SOURCE,
                assumptions=(
                    "copy-access model: no reusable state-preparation unitary",
                    "irreducible by any improvement to the solver",
                ),
            ),
            validity=Validity((
                definition(sp.Ge(S.N, 2),
                           "tomography of a one-dimensional state is vacuous"),
                definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
            )),
        ),
    },
)

IterRefine = single(
    name="IterRefine",
    summary="Solve coarsely, then solve again on the residual. Trades a tight "
            "solver precision for repeated cheap solves -- which matters "
            "because the readout cost scales as one over precision squared.",
    citation=_SOURCE,
    costs={},
)

VTAA = single(
    name="VTAA",
    summary="Variable-time amplitude amplification: amplify each branch of the "
            "approximation domain separately, so cheaper branches pay less. "
            "The interior point analysis sets its overhead to one, "
            "benevolently, rather than costing it.",
    citation=_SOURCE,
    costs={},
)

IPM = register(Routine(
    name="IPM",
    summary="Primal-dual path-following interior point method with the Newton "
            "system outsourced to a quantum linear solver. The formulation of "
            "that system is the construction.",
    implementations=(
        Implementation(
            name="mnes",
            summary="Modified normal equation system: an m-dimensional system "
                    "in the dual variables. Smaller when constraints are far "
                    "fewer than variables, at the price of pre- and "
                    "post-processing to recover the full Newton step.",
            citation=_SOURCE,
            built_from=("QLS-Chebyshev", "Tomography", "IterRefine"),
            costs={Unit.CYCLES: _cost(
                lambda v: newton_system_cycles(
                    dimension=v["N"], sparsity=v["d"], kappa=v["kappa"],
                    epsilon=v["epsilon"],
                ),
                "the basis inverse is generally dense, so the system inherits "
                "sparsity of order m rather than that of the constraint matrix",
            )},
        ),
        Implementation(
            name="oss",
            summary="Orthogonal subspace system: an n-dimensional system whose "
                    "solution splits into row-space and null-space "
                    "coefficients, so every update is feasible by "
                    "construction.",
            citation=_SOURCE,
            built_from=("QLS-Chebyshev", "Tomography", "IterRefine"),
            costs={Unit.CYCLES: _cost(
                lambda v: newton_system_cycles(
                    dimension=v["N"], sparsity=v["d"], kappa=v["kappa"],
                    epsilon=v["epsilon"],
                ),
                "sparsity is set by the densest of the concatenated blocks",
            )},
        ),
    ),
))
