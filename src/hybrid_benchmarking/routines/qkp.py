"""The tree generator for the quadratic and multidimensional knapsack problems.

Wilkening, Lefterovici, Binkowski, Funck, Perk, Karimov, Fekete and Osborne,
*A quantum search method for quadratic and multidimensional knapsack problems*
(arXiv:2503.22325), extend the tree generator of :mod:`.knapsack` to two harder
problems.  These are **separate constructions with separate counts**, not a
parameterisation of the 0-1 entry: the circuits differ in what each layer does,
the instances differ in what they carry, and a number from one is not a number
from the other.  They share the primitives -- transform, adder, comparison --
because those really are the same gates, which is the only thing that should be
shared.

**The quadratic knapsack** keeps one capacity register and changes the profit
update.  Where the 0-1 layer adds one profit, layer ``m`` here adds the linear
profit and then, for every earlier item ``m'`` whose pairwise profit is
non-zero, a *doubly*-controlled addition -- controlled on both items being in
the path::

    U3_m = C_m( C_{m-1}(ADD p_{m,m-1}) ... C_1(ADD p_{m,1}) ADD p_m )

**The multidimensional knapsack** keeps the linear profit update and changes the
capacity.  There are ``d`` capacity registers, one per weight dimension, and a
path is feasible only when every one of them survives::

    U2_m = C_m( SUB w_1m (x) ... (x) SUB w_dm )

Its qubit count the paper states outright, and it is reproduced here as written:
``n + sum_i |c_i| + |P| + max(n, sum_i |c_i| + 1, |P|)``.

**Three constants are derived rather than read, and they are the whole caveat
on these entries.**  The paper gives the circuits and says the quadratic gate
count "is rather straight-forward as the QTG for the QKP effectively arises from
the QTG for the KP plus additional doubly-controlled profit additions"; it does
not print closed forms, and the extended simulator is not published.  So the
following are worked out from Appendix C's own decomposition rules -- one
Toffoli is one gate, disjoint gates share a cycle, multi-controlled gates share
one ancilla register -- and each is named on the cost:

*A doubly-controlled addition* costs a singly-controlled one plus two Toffolis,
which compute the conjunction of the two controls into the shared ancilla and
uncompute it.  That is the cheapest reading of "multi-controlled gates share one
ancilla register", and the standing rule prefers the lower count.

*The dimensions of a multidimensional instance run in parallel.*  Their capacity
registers are disjoint, so by Appendix C's own cycle rule their subtractions and
comparisons occupy the same cycles; the cycle count takes the deepest dimension
where the gate count takes the sum.

*Feasibility across dimensions is a conjunction*, gathered by a balanced tree of
Toffolis over the ``d`` per-dimension flags, costing ``d - 1`` gates and
``2 ceil(log2 d)`` cycles.

What is **not** implemented is the further parallelisation the paper describes
for the quadratic profits -- assigning each pairwise profit to its own ancilla
register so that ``O(log2(n^2))`` layers of pairwise additions replace the
sequence, giving ``O(log2 n)`` depth.  It is stated asymptotically, without
constants, so there is nothing here to transcribe; the sequential form below is
what these entries count, and it is the dearer of the two.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import single
from ..validity import Validity, definition
from .knapsack import (
    _ASSUMPTIONS,
    _ceil_log2,
    add_cycles,
    add_gates,
    compare_ge_cycles,
    compare_ge_gates,
    least_significant_one,
    qft_cycles,
    qft_gates,
    register_size,
    subtract_cycles,
    subtract_gates,
)

_PAPER = ("Wilkening et al., a quantum search method for quadratic and "
          "multidimensional knapsack problems (arXiv:2503.22325)")

_DERIVED = (
    "the circuits are the paper's; the constants below are worked out from "
    "Appendix C's decomposition rules, because the paper states the counts "
    "structurally and its simulator is not published",
    "a doubly-controlled addition costs a singly-controlled one plus two "
    "Toffolis, computing and uncomputing the conjunction of its two controls "
    "in the shared ancilla register",
    "the further parallelisation of the quadratic profits, which the paper "
    "gives asymptotically as O(log2 n) depth using O(log2(n^2)) ancilla "
    "registers, is not implemented; this is the sequential form and the dearer "
    "of the two",
)

#: Toffolis to compute and uncompute the conjunction of a doubly-controlled
#: addition's two controls into the shared ancilla register.
CONJUNCTION_GATES = 2

#: And the cycles they occupy, which cannot overlap the addition they gate.
CONJUNCTION_CYCLES = 2


# ---------------------------------------------------------------------------
# quadratic knapsack
# ---------------------------------------------------------------------------

def _pair_key(key) -> Tuple[int, int]:
    """The two items a pair names, however the pair was written down.

    A log is a file, and a file's keys are strings, so ``(1, 0)`` arrives as
    ``"(1, 0)"`` once it has been through JSON -- and as a tuple when it comes
    straight from Python.  Both are the same pair and both are accepted; a key
    that is neither is refused rather than guessed at.
    """
    if isinstance(key, str):
        cleaned = key.strip().strip("()[] ").replace(";", ",")
        parts = [piece for piece in cleaned.split(",") if piece.strip()]
    else:
        parts = list(key)
    if len(parts) != 2:
        raise ValueError(
            "{!r} does not name a pair of items; write it as (i, j)".format(key)
        )
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        raise ValueError(
            "{!r} does not name a pair of items by index".format(key)
        )


def pairwise_terms(pairs: Mapping) -> Tuple[Tuple[int, int, int], ...]:
    """The pairwise profits that cost anything, as ``(m, m', value)``.

    ``value`` is what the pair earns *in total* when both items are selected.
    The paper states its matrix symmetrically -- ``p_{mm'} + p_{m'm}`` is the
    profit for the pair, and its objective's double sum runs over ordered pairs
    -- while the circuit performs one addition per unordered pair.  So a full
    symmetric matrix passed in here has both of its entries added together, and
    passing only the upper triangle means passing the total.  This is not
    pedantry: the gate count depends on the position of the lowest set bit, so
    a factor of two moves it.

    A zero pairwise profit is not an addition.  The layer has nothing to add, so
    the circuit has no gate for it -- the doubly-controlled addition appears
    "for each item ``m' < m`` such that ``p_{mm'} > 0``".  A *negative* one is
    refused rather than counted: strictly greater than zero is what the paper
    says, and the position of the lowest set bit of a negative number is a
    number this would otherwise happily return.
    """
    gathered: Dict[Tuple[int, int], int] = {}
    for key, value in dict(pairs).items():
        first, second = _pair_key(key)
        if first == second:
            raise ValueError(
                "({0}, {0}) is not a pair; the profit of item {0} on its own "
                "belongs in the linear profits".format(first)
            )
        canonical = (max(first, second), min(first, second))
        gathered[canonical] = gathered.get(canonical, 0) + int(value)
    for (first, second), value in sorted(gathered.items()):
        if value < 0:
            raise ValueError(
                "the pair ({}, {}) is worth {}, and this circuit has no gate "
                "for a pair that costs something when both are chosen. The "
                "layer adds a profit where p > 0 and does nothing otherwise, "
                "so a story about interference or cannibalisation is not one "
                "these counts describe".format(second, first, value)
            )
    return tuple(sorted((m, other, value)
                        for (m, other), value in gathered.items() if value))


def quadratic_addition_gates(value: int, profit_bits: int) -> int:
    """One doubly-controlled addition into the profit register."""
    return (3 * (profit_bits - least_significant_one(value)) + 1
            + CONJUNCTION_GATES)


def quadratic_addition_cycles(value: int, profit_bits: int) -> int:
    """The same, in cycles, inside a transform pair already paid for."""
    return (2 * _ceil_log2(profit_bits - least_significant_one(value)) + 1
            + CONJUNCTION_CYCLES)


def qkp_gates(profits: Sequence[int], pairs: Mapping, weights: Sequence[int],
              capacity: int, profit_bound: int) -> int:
    """``G[QTG_QKP]``: the 0-1 count plus one doubly-controlled addition per pair."""
    from .knapsack import qtg_gates

    profit_bits = register_size(profit_bound)
    return qtg_gates(profits, weights, capacity, profit_bound) + sum(
        quadratic_addition_gates(value, profit_bits)
        for _, _, value in pairwise_terms(pairs)
    )


def qkp_cycles(profits: Sequence[int], pairs: Mapping, weights: Sequence[int],
               capacity: int, profit_bound: int) -> int:
    """``C[QTG_QKP]``.

    The quadratic additions are folded into the layer unitaries rather than
    appended, which is what lets them share the transform pair the linear
    addition already opened -- the paper's own reason for the arrangement.  So
    each pair costs its rotations and its conjunction, and no transform.
    """
    from .knapsack import qtg_cycles

    profit_bits = register_size(profit_bound)
    return qtg_cycles(profits, weights, capacity, profit_bound) + sum(
        quadratic_addition_cycles(value, profit_bits)
        for _, _, value in pairwise_terms(pairs)
    )


# ---------------------------------------------------------------------------
# multidimensional knapsack
# ---------------------------------------------------------------------------

def _dimensions(weights: Sequence[Sequence[int]],
                capacities: Sequence[int]) -> Tuple[int, int]:
    rows, items = len(weights), len(weights[0]) if weights else 0
    if rows != len(capacities):
        raise ValueError(
            "{} weight dimensions but {} capacities".format(rows, len(capacities))
        )
    return rows, items


def mdkp_qubits(weights: Sequence[Sequence[int]], capacities: Sequence[int],
                profit_bound: int) -> int:
    """The paper's own qubit count, reproduced as written.

    ``n + sum_i |c_i| + |P| + max(n, sum_i |c_i| + 1, |P|)`` -- path, capacities,
    profit, and an ancilla register wide enough for whichever of the three the
    comparisons need most.
    """
    _, items = _dimensions(weights, capacities)
    capacity_bits = sum(register_size(c) for c in capacities)
    profit_bits = register_size(profit_bound)
    return (items + capacity_bits + profit_bits
            + max(items, capacity_bits + 1, profit_bits))


def mdkp_gates(profits: Sequence[int], weights: Sequence[Sequence[int]],
               capacities: Sequence[int], profit_bound: int) -> int:
    """``G[QTG_MDKP]``: every dimension pays, and the flags are conjoined."""
    count, items = _dimensions(weights, capacities)
    total = 0
    for row, capacity in zip(weights, capacities):
        bits = register_size(capacity)
        total += sum(compare_ge_gates(w, bits) for w in row)
        total += subtract_gates(row, bits)
    total += add_gates(profits, register_size(profit_bound))
    # One balanced tree of Toffolis per item, conjoining the d feasibility
    # flags into the single control the biasing rotation takes.
    total += items * max(count - 1, 0)
    return total


def mdkp_cycles(profits: Sequence[int], weights: Sequence[Sequence[int]],
                capacities: Sequence[int], profit_bound: int) -> int:
    """``C[QTG_MDKP]``.

    The dimensions occupy disjoint registers, so Appendix C's own rule --
    disjoint gates run in the same cycle -- puts their subtractions and
    comparisons side by side.  The depth is the deepest dimension, not the sum.
    """
    count, items = _dimensions(weights, capacities)
    deepest = 0
    for row, capacity in zip(weights, capacities):
        bits = register_size(capacity)
        depth = sum(compare_ge_cycles(w, bits) for w in row)
        depth += subtract_cycles(row, bits)
        deepest = max(deepest, depth)
    total = deepest + add_cycles(profits, register_size(profit_bound))
    total += items * 2 * _ceil_log2(max(count, 1))
    return total


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _cost(expr, unit: Unit, kernel, extra: Tuple[str, ...],
          *notes: str) -> Cost:
    return Cost(
        expr=expr,
        unit=unit,
        provenance=Provenance.of(
            Bound.EXACT, Derivation.ANALYTIC, _PAPER,
            assumptions=_ASSUMPTIONS + _DERIVED + notes,
        ),
        validity=Validity((
            definition(sp.Ge(S.items, 1), "a knapsack has at least one item"),
        )),
        kernel=kernel,
        extra_parameters=extra,
    )


_QKP_INPUTS = ("profits", "pair_profits", "weights")
_MDKP_INPUTS = ("profits", "weights", "capacities")

QTG_Quadratic = single(
    name="QTG-quadratic",
    summary="Tree generator for the quadratic knapsack problem: the 0-1 "
            "circuit with a doubly-controlled profit addition for every pair "
            "of items that earn something together.",
    citation=_PAPER,
    built_from=("QFT", "QFTAdd", "QFTSub"),
    costs={
        Unit.GATES: _cost(
            sp.Function("G_QKP")(S.capacity, S.profit_bound, S.items),
            Unit.GATES,
            lambda v: qkp_gates(v["profits"], v["pair_profits"], v["weights"],
                                v["capacity"], v["profit_bound"]),
            _QKP_INPUTS,
        ),
        Unit.CYCLES: _cost(
            sp.Function("C_QKP")(S.capacity, S.profit_bound, S.items),
            Unit.CYCLES,
            lambda v: qkp_cycles(v["profits"], v["pair_profits"], v["weights"],
                                 v["capacity"], v["profit_bound"]),
            _QKP_INPUTS,
            "the quadratic additions are folded into the layer unitaries, so "
            "they share the transform pair the linear addition opens",
        ),
    },
)

QTG_Multidimensional = single(
    name="QTG-multidimensional",
    summary="Tree generator for the multidimensional knapsack problem: one "
            "capacity register per weight dimension, and a path is feasible "
            "only where every one of them survives.",
    citation=_PAPER,
    built_from=("QFT", "QFTAdd", "QFTSub"),
    costs={
        Unit.GATES: _cost(
            sp.Function("G_MDKP")(S.profit_bound, S.items, S.dimensions),
            Unit.GATES,
            lambda v: mdkp_gates(v["profits"], v["weights"], v["capacities"],
                                 v["profit_bound"]),
            _MDKP_INPUTS,
        ),
        Unit.CYCLES: _cost(
            sp.Function("C_MDKP")(S.profit_bound, S.items, S.dimensions),
            Unit.CYCLES,
            lambda v: mdkp_cycles(v["profits"], v["weights"], v["capacities"],
                                  v["profit_bound"]),
            _MDKP_INPUTS,
            "the dimensions act on disjoint registers, so their subtractions "
            "and comparisons share cycles; the depth is the deepest dimension "
            "and not the sum",
        ),
    },
)
