"""Quantum breadth-first search, and Dinic's algorithm around it.

A second host algorithm for the same machinery: where the simplex logs basis
condition numbers per iteration, Dinic logs layer sizes per phase.  If the
abstraction holds, adding this means writing lemmas and reading a different log
-- not new plumbing.  It does: the expected-iteration count is the same one the
simplex subroutines use, summed over a shrinking list by :func:`QSearchAll`.

Classical breadth-first search enumerates a vertex's neighbours explicitly;
qBFS calls quantum search repeatedly instead.  Every vertex in a layer has to
be found -- missing one yields an incomplete layered graph and the wrong
maximum flow -- so the relevant cost is finding *all* marked elements, not one.

Unlike the simplex, this construction is explicit enough to schedule, so Lemma 1
counts *cycles*: a Grover iteration over a one-qubit-per-vertex register takes
``2|L|`` of them.  The paper then assumes one cycle costs one gate and reports a
gate count, which is what its comparison against classical breadth-first search
uses.

So this entry reports **gates**, and only gates.  Offering a cycle count beside
it would imply two independent results where there is one number wearing two
labels -- the conversion is an assumption, not a second derivation, and it is
recorded as one.  :func:`grover_cycles` remains available as what Lemma 1
actually establishes.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import single
from ..validity import Validity, definition
from .amplification import qsearch_all_iterations

_SOURCE = "Lefterovici, Lelakowski and Perk, quantum breadth-first search for " \
          "maximum flow (Lemmas 1 and 2)"

_ASSUMPTIONS = (
    "one qubit per vertex",
    "the phase oracle is implemented in a single gate",
    "gates are obtained from the cycle count of Lemma 1 by assuming one cycle "
    "costs exactly one gate",
)


def grover_cycles(vertices: int) -> int:
    """Cycles in one Grover iteration over a register of one qubit per vertex.

    A phase oracle and two Hadamard layers cost one cycle each; the
    ``(|L| - 1)``-controlled Z decomposes into ``2(|L| - 2)`` CNOTs and one
    controlled Z, which cannot be parallelised further.  Together
    ``1 + 1 + (2|L| - 3) + 1 = 2|L|``.
    """
    return 2 * int(vertices)


def layer_cost(vertices: int, marked: int) -> float:
    """Expected minimum cost of finding every vertex in one layer.

    ``G_Q(|L|, t) = 2 |L| N_Q(|L|, t)`` -- the per-iteration cycle count times
    the expected iterations to find all ``t`` of them.
    """
    return grover_cycles(vertices) * qsearch_all_iterations(vertices, marked)


def qbfs_cost(vertices: int, layers: Sequence[int]) -> float:
    """One breadth-first sweep, summed over its layers.

    ``layers`` is what the classical run logs: how many vertices landed in each
    layer.  The quantum version is assumed to build the same layered graph,
    which is what makes the logged sizes the right inputs.
    """
    return sum(layer_cost(vertices, marked) for marked in layers if marked > 0)


def dinic_cost(vertices: int, phases: Iterable[Sequence[int]]) -> float:
    """Every breadth-first sweep Dinic performs, one per blocking-flow phase."""
    return sum(qbfs_cost(vertices, layers) for layers in phases)


def _cost(unit: Unit, kernel, extra, *notes) -> Cost:
    return Cost(
        expr=sp.Function("G_Q")(S.X),
        unit=unit,
        provenance=Provenance.of(
            Bound.LOWER, Derivation.ANALYTIC, _SOURCE,
            assumptions=_ASSUMPTIONS + notes,
        ),
        validity=Validity((
            definition(sp.Ge(S.X, 2), "a graph needs at least two vertices"),
        )),
        kernel=kernel,
        extra_parameters=extra,
    )


qBFS = single(
    name="qBFS",
    summary="Breadth-first search with neighbours found by repeated quantum "
            "search. Every vertex in a layer must be found, so the cost is "
            "find-all rather than find-one.",
    citation=_SOURCE,
    built_from=("QSearchAll",),
    costs={
        Unit.GATES: _cost(
            Unit.GATES, lambda v: qbfs_cost(v["X"], v["layers"]), ("layers",),
        ),
    },
)

Dinic = single(
    name="Dinic",
    summary="Maximum flow by Dinic's algorithm with the layering step replaced "
            "by quantum breadth-first search.",
    citation=_SOURCE,
    built_from=("qBFS",),
    costs={
        Unit.GATES: _cost(
            Unit.GATES, lambda v: dinic_cost(v["X"], v["phases"]), ("phases",),
        ),
    },
)
