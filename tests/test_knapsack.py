"""The quantum tree generator and its circuits.

The one family with cycle counts, so this is where the relationship between
gates and cycles gets pinned down: a cycle is a layer of gates, so cycles can
never exceed gates, and the gap is exactly what the parallelisation bought.
"""

from __future__ import annotations

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.provenance import Unit
from hybrid_benchmarking.routines.knapsack import (
    compare_gt_cycles,
    compare_gt_gates,
    add_cycles,
    add_gates,
    compare_ge_cycles,
    compare_ge_gates,
    compare_zero_cycles,
    compare_zero_gates,
    least_significant_one,
    qft_cycles,
    qft_gates,
    qtg_cycles,
    qtg_gates,
    qtg_search_cycles,
    qtg_search_gates,
    register_size,
    subtract_gates,
)

#: The four-item instance of Figure 6.1.
INSTANCE = dict(profits=[6, 2, 1, 2], weights=[2, 2, 1, 5], capacity=7,
                profit_bound=11)

BOTH_UNITS = ("QFTAdd", "QFTSub", "QTG")


class TestCyclesAreLayersOfGates:
    @pytest.mark.parametrize("name", BOTH_UNITS)
    def test_both_units_are_offered(self, name):
        assert set(hb.get(name).units) == {Unit.GATES, Unit.CYCLES}

    @pytest.mark.parametrize("name", BOTH_UNITS)
    def test_cycles_never_exceed_gates(self, name):
        routine = hb.get(name)
        gates = routine.evaluate(Unit.GATES, **INSTANCE).value
        cycles = routine.evaluate(Unit.CYCLES, **INSTANCE).value
        assert 0 < cycles <= gates

    def test_the_transform_is_where_parallelisation_pays_most(self):
        """Quadratic in gates, linear in depth."""
        for bits in (4, 8, 16, 32):
            assert qft_gates(bits) == bits * (bits + 1) // 2
            assert qft_cycles(bits) == 2 * bits - 1
        assert qft_gates(32) / qft_cycles(32) > qft_gates(4) / qft_cycles(4)


class TestBitArithmetic:
    @pytest.mark.parametrize("value,expected", [
        (1, 1), (2, 2), (3, 1), (4, 3), (6, 2), (8, 4), (12, 3),
    ])
    def test_least_significant_one_counts_from_one(self, value, expected):
        assert least_significant_one(value) == expected

    def test_zero_has_none(self):
        with pytest.raises(ValueError, match="least significant one"):
            least_significant_one(0)

    @pytest.mark.parametrize("value,expected", [(1, 1), (7, 3), (8, 4), (255, 8)])
    def test_register_size(self, value, expected):
        assert register_size(value) == expected


class TestAdditionTelescopes:
    """Consecutive profit additions share a single transform pair."""

    def test_only_one_transform_pair_however_many_items(self):
        bits = 8
        one = add_gates([5], bits)
        ten = add_gates([5] * 10, bits)
        per_item = one - bits * (bits + 1)
        assert ten == bits * (bits + 1) + 10 * per_item

    def test_subtraction_does_not_telescope(self):
        """A comparison sits between each pair, so the transforms survive."""
        bits = 8
        two = subtract_gates([3, 3], bits)
        four = subtract_gates([3, 3, 3, 3], bits)
        assert four > 2 * two - bits * (bits + 1)

    def test_trailing_zeros_make_an_item_cheaper(self):
        """Rotations below the least significant one all merge away."""
        bits = 10
        odd = add_gates([1], bits)      # least significant one at position 1
        even = add_gates([512], bits)   # at position 10
        assert even < odd

    def test_the_cycle_count_is_far_below_the_gate_count(self):
        profits, bits = [3, 5, 9, 17], 12
        assert add_cycles(profits, bits) < add_gates(profits, bits)


class TestComparison:
    def test_it_takes_the_cheaper_of_two_strategies(self):
        """Whether to condition the rotation or undo it afterwards depends on
        where the ones sit in the weight."""
        capacity_bits = 12
        for weight in range(1, 64):
            assert compare_ge_gates(weight, capacity_bits) > 0
            assert compare_ge_cycles(weight, capacity_bits) \
                <= compare_ge_gates(weight, capacity_bits)

    def test_the_reflection_is_linear_in_gates_logarithmic_in_depth(self):
        assert compare_zero_gates(600) == 2 * 600 - 1
        assert compare_zero_cycles(600) < compare_zero_gates(600)


class TestTheGenerator:
    def test_it_matches_its_parts(self):
        gates = qtg_gates(**INSTANCE)
        assert gates > 0
        # Every item contributes a comparison, and the transforms dominate.
        assert gates > sum(
            compare_ge_gates(w, register_size(INSTANCE["capacity"]))
            for w in INSTANCE["weights"]
        )

    def test_more_items_cost_more(self):
        small = qtg_gates(profits=[6, 2], weights=[2, 2], capacity=7,
                          profit_bound=8)
        large = qtg_gates(profits=[6, 2, 1, 2], weights=[2, 2, 1, 5],
                          capacity=7, profit_bound=11)
        assert large > small

    def test_a_larger_capacity_widens_the_register_and_costs_more(self):
        narrow = qtg_gates(profits=[6, 2], weights=[2, 2], capacity=7,
                           profit_bound=8)
        wide = qtg_gates(profits=[6, 2], weights=[2, 2], capacity=10 ** 6,
                         profit_bound=8)
        assert wide > narrow

    def test_it_is_a_state_preparation_not_an_algorithm(self):
        """Which is why the search around it is the ordinary generic."""
        assert "QAA" in hb.get("QTGSearch").implementation().built_from
        assert "QTG" in hb.get("QTGSearch").implementation().built_from


class TestTheSearch:
    SCHEDULE = [(6, [1, 2]), (9, [1, 3])]

    def test_it_costs_more_than_one_generator(self):
        one = qtg_gates(**INSTANCE)
        search = qtg_search_gates(schedule=self.SCHEDULE, **INSTANCE)
        assert search > one

    def test_a_longer_schedule_costs_more(self):
        short = qtg_search_gates(schedule=[(6, [1])], **INSTANCE)
        long = qtg_search_gates(schedule=[(6, [1, 2, 3])], **INSTANCE)
        assert long > short

    def test_higher_grover_powers_cost_more(self):
        low = qtg_search_gates(schedule=[(6, [1])], **INSTANCE)
        high = qtg_search_gates(schedule=[(6, [8])], **INSTANCE)
        assert high > low

    def test_cycles_stay_below_gates(self):
        assert qtg_search_cycles(schedule=self.SCHEDULE, **INSTANCE) < \
            qtg_search_gates(schedule=self.SCHEDULE, **INSTANCE)

    def test_the_schedule_is_an_input_not_a_cost(self):
        """Producing it needs a simulation of the instance; the formulas here
        only say what it costs once you have it."""
        assert "schedule" in hb.get("QTGSearch").parameters
        with pytest.raises(ValueError, match="missing parameters: schedule"):
            hb.get("QTGSearch").evaluate(Unit.GATES, **INSTANCE)


class TestThroughTheRegistry:
    def test_the_instance_vectors_are_declared_parameters(self):
        params = set(hb.get("QTG").parameters)
        assert {"profits", "weights", "capacity", "profit_bound"} <= params

    def test_a_missing_vector_is_named(self):
        with pytest.raises(ValueError, match="missing parameters: weights"):
            hb.get("QTG").evaluate(Unit.GATES, profits=[1], capacity=7,
                                   profit_bound=8)

    def test_the_transform_takes_only_a_register_size(self):
        assert set(hb.get("QFT").parameters) == {"bits"}


class TestTheSearchAgainstTheOriginalSimulator:
    """``QTGSearch`` accumulates exactly what the original's driver accumulates.

    Its ``execute_q_max_search`` adds, once per threshold segment,
    ``(rounds + 2 iter) C_QTG + iter (C_mc(n-1) + C_comp(profit_qubits, P))``,
    where ``rounds`` counts amplification attempts in that segment and ``iter``
    sums the Grover powers drawn. This library writes the same thing per drawn
    power, ``(2j + 1) C_QTG + j (zero + marker)``, and the two are the same sum:
    ``sum_j (2j + 1) = 2 iter + rounds``. Worth pinning, because the two
    groupings look nothing alike and only one of them is obviously the circuit.
    """

    SCHEDULE = [(8, [1, 3, 2]), (9, [1, 2])]

    def _segmented(self, generator, zero, marker_of):
        bits = register_size(INSTANCE["profit_bound"])
        total = 0
        for threshold, powers in self.SCHEDULE:
            rounds, iterations = len(powers), sum(powers)
            total += (rounds + 2 * iterations) * generator
            total += iterations * (zero + marker_of(threshold, bits))
        return total

    def test_the_cycle_accumulation_matches_segment_for_segment(self):
        assert qtg_search_cycles(schedule=self.SCHEDULE, **INSTANCE) == \
            self._segmented(qtg_cycles(**INSTANCE),
                            compare_zero_cycles(len(INSTANCE["profits"])),
                            compare_gt_cycles)

    def test_the_gate_accumulation_matches_too(self):
        assert qtg_search_gates(schedule=self.SCHEDULE, **INSTANCE) == \
            self._segmented(qtg_gates(**INSTANCE),
                            compare_zero_gates(len(INSTANCE["profits"])),
                            compare_gt_gates)

    def test_a_segment_that_draws_nothing_costs_nothing(self):
        assert qtg_search_cycles(schedule=[(8, [])], **INSTANCE) == 0
