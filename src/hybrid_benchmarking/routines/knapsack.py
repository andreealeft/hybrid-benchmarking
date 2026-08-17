"""The quantum tree generator and its circuits, in gates and in cycles.

This is the only family in the library with cycle counts, and the reason is
structural: a cycle is a layer of gates that run in parallel, so counting them
needs an actual schedule.  These circuits have one -- Appendix C works out the
parallelisation explicitly -- whereas the simplex gate counts are analytic
lower bounds with terms dropped, and there is nothing there to schedule.

The quantum tree generator is a **state preparation**, not an algorithm.  It
builds a superposition of feasible knapsack paths biased towards profitable
ones; the search around it is ordinary maximum finding with amplitude
amplification.  Treating it that way is what lets the amplification generic in
:mod:`.amplification` serve here unchanged.

Costs here depend on the instance itself, not on summary statistics: each
profit and weight enters through the position of its least significant one,
because that is where the Fourier adder's rotations start.  So these entries
declare vector parameters alongside their scalar ones.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import single
from ..validity import Validity, definition

_APPENDIX_C = "Appendix C of the thesis"
_CHAPTER_6 = "Chapter 6 of the thesis (equations 6.6 to 6.15)"

_ASSUMPTIONS = (
    "single-qubit gates, two-qubit rotations and Toffolis all cost one gate",
    "disjoint gates run in the same cycle",
    "multi-controlled gates share one ancilla register",
)


# ---------------------------------------------------------------------------
# bit arithmetic
# ---------------------------------------------------------------------------

def register_size(value: int) -> int:
    """Qubits needed to hold ``value``: ``floor(log2 v) + 1``."""
    return int(value).bit_length()


def least_significant_one(value: int) -> int:
    """Position of the lowest set bit, counting from one.

    This is where the Fourier adder's rotation sequence starts: rotations from
    higher bit positions merge into it, so everything below costs nothing.
    """
    value = int(value)
    if value == 0:
        raise ValueError("zero has no least significant one")
    return (value & -value).bit_length()


def _bit(value: int, index: int) -> int:
    """The ``index``-th bit of ``value``, counting from one at the bottom."""
    return (int(value) >> (index - 1)) & 1


def _ceil_log2(value: float) -> int:
    """``ceil(log2 v)``, and zero for v <= 1 -- a register of one needs no tree."""
    return 0 if value <= 1 else int(math.ceil(math.log2(value)))


# ---------------------------------------------------------------------------
# C.1  Fourier transform
# ---------------------------------------------------------------------------

def qft_gates(bits: int) -> int:
    """``s (s + 1) / 2`` -- one Hadamard and a descending ladder per qubit."""
    return bits * (bits + 1) // 2


def qft_cycles(bits: int) -> int:
    """``2 s - 1``.

    Shifting each level's sequence two layers after the one below turns a
    quadratic gate count into a linear depth; see Figure C.1.
    """
    return 2 * bits - 1


# ---------------------------------------------------------------------------
# C.2  addition on the profit register
# ---------------------------------------------------------------------------

def add_gates(profits: Sequence[int], profit_bits: int) -> int:
    """``G[ADD_p]``.

    All the profit additions are consecutive, so every inner inverse transform
    cancels the next transform: only one pair survives for the whole sequence.
    """
    return profit_bits * (profit_bits + 1) + sum(
        3 * (profit_bits - least_significant_one(p)) + 1 for p in profits
    )


def add_cycles(profits: Sequence[int], profit_bits: int) -> int:
    """``C[ADD_p]``.

    The rotations are controlled onto separate ancillas, so all of them run in
    one cycle and the control on the transforms can be dropped entirely.
    """
    return 4 * profit_bits - 2 + sum(
        2 * _ceil_log2(profit_bits - least_significant_one(p)) + 1
        for p in profits
    )


# ---------------------------------------------------------------------------
# C.3  subtraction on the capacity register
# ---------------------------------------------------------------------------

def subtract_gates(weights: Sequence[int], capacity_bits: int) -> int:
    """``G[SUB_w]``.

    Unlike the additions these do not telescope: each subtraction is followed
    by a comparison against the remaining capacity, so the transforms in
    between survive.  The last subtraction is omitted -- after the final item
    nothing checks the capacity again.
    """
    head = list(weights)[:-1]
    n = len(weights)
    return (n - 1) * capacity_bits * (capacity_bits + 1) + sum(
        3 * (capacity_bits - least_significant_one(w)) + 1 for w in head
    )


def subtract_cycles(weights: Sequence[int], capacity_bits: int) -> int:
    head = list(weights)[:-1]
    n = len(weights)
    return 4 * (n - 1) * capacity_bits - 2 * (n - 1) + sum(
        2 * _ceil_log2(capacity_bits - least_significant_one(w)) + 1
        for w in head
    )


# ---------------------------------------------------------------------------
# C.4  comparison
# ---------------------------------------------------------------------------

def _compare_ge(weight: int, capacity_bits: int, cycles: bool) -> int:
    """Is the remaining capacity at least this item's weight.

    Two strategies, and the circuit takes whichever is cheaper.  The first
    applies the rotation only when ``c > w - 1`` and so reads the bits of
    ``w - 1``; the second applies it unconditionally and undoes it when
    ``c < w``, reading the bits of ``w``.  Which wins depends on where the ones
    sit in the weight's binary representation.
    """
    width = register_size(weight)

    def weight_of(index: int) -> int:
        return (2 * _ceil_log2(capacity_bits - index) + 1) if cycles \
            else (2 * (capacity_bits - index) + 1)

    first = sum(
        (1 - _bit(weight - 1, i)) * weight_of(i) for i in range(1, width + 1)
    )
    second = 1 + sum(
        _bit(weight, i) * weight_of(i) for i in range(1, width + 1)
    )
    return min(first, second)


def compare_ge_gates(weight: int, capacity_bits: int) -> int:
    return _compare_ge(weight, capacity_bits, cycles=False)


def compare_ge_cycles(weight: int, capacity_bits: int) -> int:
    return _compare_ge(weight, capacity_bits, cycles=True)


def compare_zero_gates(n_items: int) -> int:
    """``G[= 0]`` -- the reflection about the all-zero path register."""
    return 2 * n_items - 1


def compare_zero_cycles(n_items: int) -> int:
    return 2 * _ceil_log2(n_items - 1) + 1


def _compare_gt(threshold: int, profit_bits: int, cycles: bool) -> int:
    """Does this path's profit exceed the current threshold.

    Note the asymmetry with :func:`_compare_ge`: (C.28) and (C.29) group the
    cycle weight as ``2 (ceil log2 (P - i) + 1)`` where (C.24) and (C.25) use
    ``2 ceil log2 (c - i) + 1``.  Both are transcribed as written.
    """
    width = register_size(threshold) if threshold else 1

    def weight_of(index: int) -> int:
        return 2 * (_ceil_log2(profit_bits - index) + 1) if cycles \
            else 2 * (profit_bits - index) + 1

    first = sum(
        (1 - _bit(threshold, i)) * weight_of(i) for i in range(1, width + 1)
    )
    second = 1 + sum(
        _bit(threshold + 1, i) * weight_of(i) for i in range(1, width + 1)
    )
    return min(first, second)


def compare_gt_gates(threshold: int, profit_bits: int) -> int:
    return _compare_gt(threshold, profit_bits, cycles=False)


def compare_gt_cycles(threshold: int, profit_bits: int) -> int:
    return _compare_gt(threshold, profit_bits, cycles=True)


# ---------------------------------------------------------------------------
# the tree generator itself
# ---------------------------------------------------------------------------

def qtg_gates(profits: Sequence[int], weights: Sequence[int],
              capacity: int, profit_bound: int) -> int:
    """``G[QTG]`` of (6.6): one biased branch, one subtraction, one addition
    per item, with the copying steps merged where they share ancillas."""
    n = len(profits)
    c_bits = register_size(capacity)
    p_bits = register_size(profit_bound)
    lso_p = [least_significant_one(p) for p in profits]
    lso_w = [least_significant_one(w) for w in weights]

    total = sum(compare_ge_gates(w, c_bits) for w in weights)
    total += 2 * qft_gates(p_bits)
    total += 2 * (n - 1) * qft_gates(c_bits)
    total += 2 * sum(
        max(p_bits, c_bits) - min(lso_p[m], lso_w[m]) for m in range(n - 1)
    )
    total += 2 * (p_bits - lso_p[-1])
    total += sum(
        (p_bits - lso_p[m]) + (c_bits - lso_w[m]) + 2 for m in range(n - 1)
    )
    total += p_bits - lso_p[-1] + 1
    return total


def qtg_cycles(profits: Sequence[int], weights: Sequence[int],
               capacity: int, profit_bound: int) -> int:
    """``C[QTG]`` of (6.7) to (6.11), layer unitary by layer unitary.

    The layers cannot overlap: the subtraction and addition in one layer are
    controlled by the qubit the biasing rotation acts on, so nothing commutes
    across a layer boundary.
    """
    n = len(profits)
    c_bits = register_size(capacity)
    p_bits = register_size(profit_bound)
    qft_c, qft_p = qft_cycles(c_bits), qft_cycles(p_bits)

    # First layer: the profit transform can absorb the subtraction only when
    # the profit register is the deeper of the two.
    if p_bits > c_bits:
        total = compare_ge_cycles(weights[0], c_bits) + qft_c + qft_p
    else:
        total = compare_ge_cycles(weights[0], c_bits) + 2 * qft_c + 1

    for m in range(1, n - 1):
        total += compare_ge_cycles(weights[m], c_bits) + 2 * qft_c + 1

    if n > 1:
        last_p = least_significant_one(profits[-1])
        total += (compare_ge_cycles(weights[-1], c_bits)
                  + _ceil_log2(p_bits - last_p) + qft_p + 1)
    return total


Schedule = Iterable[Tuple[int, Sequence[int]]]


def _qtg_search(profits, weights, capacity, profit_bound, schedule: Schedule,
                cycles: bool) -> float:
    """``G[QMF]`` / ``C[QMF]`` of (6.12) to (6.15).

    ``schedule`` is what the runtime calculator observes: for each round of
    maximum finding, the threshold in force and the Grover powers actually
    drawn.  The formulas are here; producing the schedule needs a simulation of
    the instance and is a probe, not a cost.
    """
    generator = qtg_cycles(profits, weights, capacity, profit_bound) if cycles \
        else qtg_gates(profits, weights, capacity, profit_bound)
    p_bits = register_size(profit_bound)
    zero = compare_zero_cycles(len(profits)) if cycles \
        else compare_zero_gates(len(profits))

    total = 0.0
    for threshold, powers in schedule:
        marker = compare_gt_cycles(threshold, p_bits) if cycles \
            else compare_gt_gates(threshold, p_bits)
        for j in powers:
            total += (2 * j + 1) * generator + j * (zero + marker)
    return total


def qtg_search_gates(profits, weights, capacity, profit_bound,
                     schedule: Schedule) -> float:
    return _qtg_search(profits, weights, capacity, profit_bound, schedule,
                       cycles=False)


def qtg_search_cycles(profits, weights, capacity, profit_bound,
                      schedule: Schedule) -> float:
    return _qtg_search(profits, weights, capacity, profit_bound, schedule,
                       cycles=True)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _cost(expr, unit, kernel, source, extra=(), validity=None, *notes) -> Cost:
    return Cost(
        expr=expr,
        unit=unit,
        provenance=Provenance.of(
            Bound.EXACT, Derivation.ANALYTIC, source,
            assumptions=_ASSUMPTIONS + notes,
        ),
        validity=validity if validity is not None else Validity(),
        kernel=kernel,
        extra_parameters=extra,
    )


#: The vectors an instance supplies.  ``capacity`` and
#: ``profit_bound`` are ordinary scalar parameters and appear in the
#: displayed formulas, since the register sizes follow from them.
_INSTANCE = ("profits", "weights")

QFT = single(
    name="QFT",
    summary="Quantum Fourier transform on a register of the given size.",
    citation=_APPENDIX_C + " (C.1)",
    costs={
        Unit.GATES: _cost(
            S.register_bits * (S.register_bits + 1) / 2, Unit.GATES,
            lambda v: qft_gates(int(v["bits"])), _APPENDIX_C,
            validity=Validity((
                definition(sp.Ge(S.register_bits, 1),
                           "a register holds at least one qubit"),
            )),
        ),
        Unit.CYCLES: _cost(
            2 * S.register_bits - 1, Unit.CYCLES,
            lambda v: qft_cycles(int(v["bits"])), _APPENDIX_C,
        ),
    },
)

QFTAdd = single(
    name="QFTAdd",
    summary="Add every item's profit into the profit register, by Fourier "
            "adder. Consecutive additions share one transform pair.",
    citation=_APPENDIX_C + " (C.2)",
    built_from=("QFT",),
    costs={
        Unit.GATES: _cost(
            sp.Function("G_add")(S.profit_bound), Unit.GATES,
            lambda v: add_gates(v["profits"], register_size(v["profit_bound"])),
            _APPENDIX_C, extra=_INSTANCE,
        ),
        Unit.CYCLES: _cost(
            sp.Function("C_add")(S.profit_bound), Unit.CYCLES,
            lambda v: add_cycles(v["profits"], register_size(v["profit_bound"])),
            _APPENDIX_C, extra=_INSTANCE,
        ),
    },
)

QFTSub = single(
    name="QFTSub",
    summary="Subtract every item's weight from the capacity register. These do "
            "not telescope: a comparison sits between each pair.",
    citation=_APPENDIX_C + " (C.3)",
    built_from=("QFT",),
    costs={
        Unit.GATES: _cost(
            sp.Function("G_sub")(S.capacity), Unit.GATES,
            lambda v: subtract_gates(v["weights"], register_size(v["capacity"])),
            _APPENDIX_C, extra=_INSTANCE,
        ),
        Unit.CYCLES: _cost(
            sp.Function("C_sub")(S.capacity), Unit.CYCLES,
            lambda v: subtract_cycles(v["weights"], register_size(v["capacity"])),
            _APPENDIX_C, extra=_INSTANCE,
        ),
    },
)

QTG = single(
    name="QTG",
    summary="Quantum tree generator: prepares a superposition of feasible "
            "knapsack paths, biased towards profitable ones. A state "
            "preparation, not an algorithm.",
    citation=_CHAPTER_6 + "; " + _APPENDIX_C,
    built_from=("QFT", "QFTAdd", "QFTSub"),
    costs={
        Unit.GATES: _cost(
            sp.Function("G_QTG")(S.capacity, S.profit_bound), Unit.GATES,
            lambda v: qtg_gates(v["profits"], v["weights"], v["capacity"],
                                v["profit_bound"]),
            _CHAPTER_6, extra=_INSTANCE,
        ),
        Unit.CYCLES: _cost(
            sp.Function("C_QTG")(S.capacity, S.profit_bound), Unit.CYCLES,
            lambda v: qtg_cycles(v["profits"], v["weights"], v["capacity"],
                                 v["profit_bound"]),
            _CHAPTER_6, extra=_INSTANCE,
        ),
    },
)

QTGSearch = single(
    name="QTGSearch",
    summary="Maximum finding around the tree generator: repeated amplitude "
            "amplification against a rising profit threshold.",
    citation=_CHAPTER_6,
    built_from=("QTG", "QAA"),
    costs={
        Unit.GATES: _cost(
            sp.Function("G_QMF")(S.capacity, S.profit_bound), Unit.GATES,
            lambda v: qtg_search_gates(v["profits"], v["weights"], v["capacity"],
                                       v["profit_bound"], v["schedule"]),
            _CHAPTER_6, extra=_INSTANCE + ("schedule",),
        ),
        Unit.CYCLES: _cost(
            sp.Function("C_QMF")(S.capacity, S.profit_bound), Unit.CYCLES,
            lambda v: qtg_search_cycles(v["profits"], v["weights"], v["capacity"],
                                        v["profit_bound"], v["schedule"]),
            _CHAPTER_6, extra=_INSTANCE + ("schedule",),
        ),
    },
)
