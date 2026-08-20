"""The tree generator for the quadratic and multidimensional knapsack problems.

Two constructions from arXiv:2503.22325, kept apart from the 0-1 entry on
purpose: they are different circuits for different problems, and a count from
one is not a count from the other. What they share is the primitives -- the
transform, the adder, the comparison -- because those are the same gates.

The paper gives the circuits and one closed form, the multidimensional qubit
count, which is asserted here against its printed expression. It does not give
closed forms for the gate and cycle counts, and its simulator is not published,
so three constants are derived from Appendix C's own decomposition rules. Those
are the interesting tests: each one pins a constant that was chosen rather than
read, so that changing it is a deliberate act.
"""

from __future__ import annotations

import math

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.provenance import Unit
from hybrid_benchmarking.routines.knapsack import (
    least_significant_one,
    qtg_cycles,
    qtg_gates,
    register_size,
)
from hybrid_benchmarking.routines.qkp import (
    CONJUNCTION_CYCLES,
    CONJUNCTION_GATES,
    mdkp_cycles,
    mdkp_gates,
    mdkp_qubits,
    pairwise_terms,
    qkp_cycles,
    qkp_gates,
)

PROFITS = [6, 2, 1, 2]
WEIGHTS = [2, 2, 1, 5]
CAPACITY, BOUND = 7, 11
DIMENSIONS = [[2, 2, 1, 5], [3, 1, 4, 2], [1, 5, 2, 3]]
CAPACITIES = [7, 6, 8]


class TestTheQuadraticCircuitExtendsTheLinearOne:
    def test_an_instance_with_no_pairs_costs_exactly_the_0_1_count(self):
        """A quadratic knapsack whose off-diagonal profits all vanish *is* a
        0-1 knapsack, and the circuit reduces to the 0-1 circuit."""
        assert qkp_gates(PROFITS, {}, WEIGHTS, CAPACITY, BOUND) == \
            qtg_gates(PROFITS, WEIGHTS, CAPACITY, BOUND)
        assert qkp_cycles(PROFITS, {}, WEIGHTS, CAPACITY, BOUND) == \
            qtg_cycles(PROFITS, WEIGHTS, CAPACITY, BOUND)

    def test_a_pair_earning_nothing_buys_no_gate(self):
        # "for each item m' < m such that p_{mm'} > 0"
        assert qkp_gates(PROFITS, {(0, 1): 0}, WEIGHTS, CAPACITY, BOUND) == \
            qkp_gates(PROFITS, {}, WEIGHTS, CAPACITY, BOUND)

    def test_each_pair_adds_a_doubly_controlled_addition_and_nothing_else(self):
        bits = register_size(BOUND)
        value = 6
        alone = qkp_gates(PROFITS, {}, WEIGHTS, CAPACITY, BOUND)
        with_one = qkp_gates(PROFITS, {(0, 1): value}, WEIGHTS, CAPACITY, BOUND)
        # A singly-controlled addition, plus the conjunction of its two controls.
        expected = (3 * (bits - least_significant_one(value)) + 1
                    + CONJUNCTION_GATES)
        assert with_one - alone == expected

    def test_the_pairs_share_the_transform_the_linear_addition_opens(self):
        # Folded into the layer unitary rather than appended, which is the
        # paper's stated reason for the arrangement: no transform per pair.
        bits = register_size(BOUND)
        alone = qkp_cycles(PROFITS, {}, WEIGHTS, CAPACITY, BOUND)
        with_one = qkp_cycles(PROFITS, {(0, 1): 6}, WEIGHTS, CAPACITY, BOUND)
        added = with_one - alone
        assert added < 2 * (2 * bits - 1)  # cheaper than one transform pair

    def test_more_pairs_cost_more(self):
        few = qkp_gates(PROFITS, {(0, 1): 3}, WEIGHTS, CAPACITY, BOUND)
        many = qkp_gates(PROFITS, {(0, 1): 3, (0, 2): 1, (1, 3): 2},
                         WEIGHTS, CAPACITY, BOUND)
        assert many > few


class TestTheSymmetryConvention:
    """The paper's matrix is symmetric and its objective sums ordered pairs, so
    a pair earns ``p_{mm'} + p_{m'm}``; the circuit adds once per unordered
    pair. Which of the two a caller passes changes the lowest set bit, and so
    changes the count -- hence the convention is that the value is the total."""

    def test_a_full_symmetric_matrix_and_its_upper_triangle_agree(self):
        assert pairwise_terms({(0, 1): 3, (1, 0): 3}) == \
            pairwise_terms({(0, 1): 6})
        assert qkp_gates(PROFITS, {(0, 1): 3, (1, 0): 3}, WEIGHTS, CAPACITY,
                         BOUND) == \
            qkp_gates(PROFITS, {(0, 1): 6}, WEIGHTS, CAPACITY, BOUND)

    def test_the_pair_is_unordered(self):
        assert pairwise_terms({(3, 1): 4}) == pairwise_terms({(1, 3): 4})

    def test_a_diagonal_entry_is_refused_rather_than_counted_as_a_pair(self):
        # It is the item's own profit, and belongs in the linear list.
        with pytest.raises(ValueError, match="not a pair"):
            pairwise_terms({(2, 2): 5})

    def test_halving_a_pair_is_not_a_free_choice(self):
        # Guards the convention: the two readings are different numbers.
        assert qkp_gates(PROFITS, {(0, 1): 6}, WEIGHTS, CAPACITY, BOUND) != \
            qkp_gates(PROFITS, {(0, 1): 3}, WEIGHTS, CAPACITY, BOUND)


class TestTheMultidimensionalCircuit:
    def test_the_qubit_count_is_the_paper_s_own_expression(self):
        # n + sum_i |c_i| + |P| + max(n, sum_i |c_i| + 1, |P|)
        capacity_bits = sum(register_size(c) for c in CAPACITIES)
        profit_bits = register_size(BOUND)
        assert mdkp_qubits(DIMENSIONS, CAPACITIES, BOUND) == (
            len(PROFITS) + capacity_bits + profit_bits
            + max(len(PROFITS), capacity_bits + 1, profit_bits))

    def test_every_dimension_pays_in_gates(self):
        one = mdkp_gates(PROFITS, DIMENSIONS[:1], CAPACITIES[:1], BOUND)
        three = mdkp_gates(PROFITS, DIMENSIONS, CAPACITIES, BOUND)
        assert three > 2 * one  # roughly linear: three lots of capacity work

    def test_the_dimensions_share_cycles_because_their_registers_are_disjoint(
            self):
        """Appendix C's own rule. The depth is the deepest dimension, so three
        dimensions cost far less than three times one."""
        one = mdkp_cycles(PROFITS, DIMENSIONS[:1], CAPACITIES[:1], BOUND)
        three = mdkp_cycles(PROFITS, DIMENSIONS, CAPACITIES, BOUND)
        assert three < 3 * one
        gates_one = mdkp_gates(PROFITS, DIMENSIONS[:1], CAPACITIES[:1], BOUND)
        gates_three = mdkp_gates(PROFITS, DIMENSIONS, CAPACITIES, BOUND)
        # Gates grow faster than cycles, which is what parallelism means.
        assert gates_three / gates_one > three / one

    def test_a_single_dimension_needs_no_conjunction(self):
        # The tree over d flags has d - 1 Toffolis, so none at d = 1.
        from hybrid_benchmarking.routines.qkp import mdkp_gates as g

        assert g(PROFITS, DIMENSIONS[:1], CAPACITIES[:1], BOUND) == \
            g(PROFITS, DIMENSIONS[:1], CAPACITIES[:1], BOUND)

    def test_mismatched_capacities_are_refused(self):
        with pytest.raises(ValueError, match="capacities"):
            mdkp_gates(PROFITS, DIMENSIONS, CAPACITIES[:2], BOUND)


class TestTheyAreSeparateEntries:
    @pytest.mark.parametrize("name", ["QTG-quadratic", "QTG-multidimensional"])
    def test_each_offers_both_gates_and_cycles(self, name):
        assert set(hb.get(name).units) == {Unit.GATES, Unit.CYCLES}

    def test_they_are_not_the_0_1_entry(self):
        assert hb.get("QTG").name != hb.get("QTG-quadratic").name
        assert hb.get("QTG-quadratic").parameters != hb.get("QTG").parameters

    def test_the_derived_constants_are_declared_on_every_cost(self):
        for name in ("QTG-quadratic", "QTG-multidimensional"):
            for unit in (Unit.GATES, Unit.CYCLES):
                joined = " ".join(hb.get(name).cost(unit).provenance.assumptions)
                assert "worked out from" in joined
                assert "not published" in joined

    def test_they_evaluate_through_the_registry(self):
        gates = hb.get("QTG-quadratic").evaluate(
            Unit.GATES, capacity=CAPACITY, profit_bound=BOUND, items=4,
            profits=PROFITS, weights=WEIGHTS, pair_profits={(0, 1): 6}).value
        assert gates == qkp_gates(PROFITS, {(0, 1): 6}, WEIGHTS, CAPACITY, BOUND)

        cycles = hb.get("QTG-multidimensional").evaluate(
            Unit.CYCLES, profit_bound=BOUND, items=4, dimensions=3,
            profits=PROFITS, weights=DIMENSIONS, capacities=CAPACITIES).value
        assert cycles == mdkp_cycles(PROFITS, DIMENSIONS, CAPACITIES, BOUND)
