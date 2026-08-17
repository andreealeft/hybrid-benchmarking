"""Knapsack, where the instance is the input and nothing is run.

The one route in this package with no classical solve behind it, which makes
two things worth stating that nothing else here has to.

The first is the profit bound.  It is not in the file -- or rather, an optimum
sometimes is and a bound never -- and it fixes the width of the profit register,
so it appears inside every count in Appendix C.  A looser bound is still
correct and costs more, so the tests pin down that what is chosen really is an
upper bound on the optimum, and really is the tighter of the two available.

The second is what kind of number comes out.  Nothing was measured here, so the
count is exact and analytic: it would come out identical on anyone else's
machine.  Marking it as logged, which is what every other route here does and
what the machinery does by default, would hedge a number that is not hedged --
and a caveat that is not true is as much a provenance failure as a missing one.
"""

from __future__ import annotations

from itertools import combinations

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.knapsack import dantzig_bound, solve
from hybrid_benchmarking.instances import Knapsack
from hybrid_benchmarking.instances.knapsack import read as read_knapsack
from hybrid_benchmarking.provenance import Bound, Derivation


def instance(profits, weights, capacity, optimum=None) -> Knapsack:
    return Knapsack(name="k", source="(hand built)", layout="pisinger",
                    profits=tuple(profits), weights=tuple(weights),
                    capacity=capacity, optimum=optimum)


def best_value(item: Knapsack) -> int:
    """The optimum, by trying every subset.  A statement of the problem."""
    best = 0
    indices = range(len(item.profits))
    for size in range(len(item.profits) + 1):
        for chosen in combinations(indices, size):
            if sum(item.weights[i] for i in chosen) <= item.capacity:
                best = max(best, sum(item.profits[i] for i in chosen))
    return best


TOY = instance([6, 2, 1, 2], [2, 2, 1, 5], 7)  # the library's own example


class TestTheProfitBound:
    def test_the_dantzig_bound_is_above_the_true_optimum(self):
        assert dantzig_bound(TOY.profits, TOY.weights, TOY.capacity) \
            >= best_value(TOY)

    @pytest.mark.parametrize("case", [
        instance([10, 6, 5], [5, 4, 3], 8),
        instance([1, 1, 1, 1], [1, 1, 1, 1], 2),
        instance([44, 46, 90, 72, 91, 40], [23, 31, 52, 44, 54, 26], 60),
    ])
    def test_it_stays_above_the_optimum_on_every_instance_we_try(self, case):
        assert dantzig_bound(case.profits, case.weights, case.capacity) \
            >= best_value(case)

    def test_everything_fitting_makes_the_bound_the_total_profit(self):
        case = instance([3, 4, 5], [1, 1, 1], 10)
        assert dantzig_bound(case.profits, case.weights, case.capacity) == 12

    def test_a_stated_optimum_is_preferred_when_it_is_tighter(self):
        # An instance where the relaxation is genuinely loose: one heavy item
        # dominates on value per unit of cost but cannot be taken whole.
        case = instance([44, 46, 90, 72, 91, 40], [23, 31, 52, 44, 54, 26], 60)
        exact = best_value(case)
        assert exact < dantzig_bound(case.profits, case.weights, case.capacity)

        run = solve(instance(case.profits, case.weights, case.capacity, exact),
                    Budget(60))
        assert run.instance["profit_bound"] == exact
        assert any("the optimum the file states" in note
                   for note in run.assumptions)

    def test_the_two_bounds_agreeing_is_not_treated_as_a_disagreement(self):
        # On this instance the relaxation happens to be tight, and the answer
        # is the same number whichever route it came by.
        exact = best_value(TOY)
        assert dantzig_bound(TOY.profits, TOY.weights, TOY.capacity) == exact
        run = solve(instance(TOY.profits, TOY.weights, TOY.capacity, exact),
                    Budget(60))
        assert run.instance["profit_bound"] == exact

    def test_a_loose_stated_optimum_is_not_preferred_over_the_bound(self):
        # A file claiming an optimum above the relaxation is claiming something
        # impossible; the computed bound wins rather than the file.
        run = solve(instance(TOY.profits, TOY.weights, TOY.capacity, 10_000),
                    Budget(60))
        assert run.instance["profit_bound"] == dantzig_bound(
            TOY.profits, TOY.weights, TOY.capacity)

    def test_the_bound_that_was_used_is_recorded_either_way(self):
        for case in (TOY, instance(TOY.profits, TOY.weights, TOY.capacity, 8)):
            run = solve(case, Budget(60))
            assert any("profit register is sized by" in note
                       for note in run.assumptions)


class TestWhatComesOut:
    def test_the_log_carries_the_instance_and_no_records(self):
        run = solve(TOY, Budget(60))
        assert run.status is Status.COMPLETE
        assert run.records == ()
        assert run.instance["profits"] == list(TOY.profits)
        assert run.instance["weights"] == list(TOY.weights)
        assert run.instance["capacity"] == TOY.capacity

    def test_a_run_with_no_records_is_still_usable(self):
        # The default reading of "usable" is "produced records", and this route
        # never produces any.
        assert solve(TOY, Budget(60)).usable

    def test_the_count_is_exact_and_analytic_rather_than_logged(self):
        report = cost(generate(TOY, budget=Budget(60)))
        assert report["derivation"] == str(Derivation.ANALYTIC)
        assert report["bound"] == str(Bound.EXACT)
        assert report["unit"] == "CYCLES"

    def test_it_still_says_where_the_numbers_came_from(self):
        report = cost(generate(TOY, budget=Budget(60)))
        assert "read rather than solved" in report["provenance"]

    def test_the_search_around_the_generator_is_handed_off_not_guessed(self):
        run = solve(TOY, Budget(60))
        assert "QTGSearch" in run.handoff
        assert "simulation" in run.handoff
        assert any("amplification schedule is not produced" in note
                   for note in run.assumptions)


class TestItRefusesRatherThanProducingNonsense:
    def test_an_instance_where_nothing_fits_has_no_register_to_size(self):
        run = solve(instance([5, 6], [100, 200], 3), Budget(60))
        assert run.status is Status.FAILED
        assert "no item fits" in run.reason

    def test_an_instance_with_no_items_is_refused(self):
        run = solve(instance([], [], 10), Budget(60))
        assert run.status is Status.FAILED


class TestFromAFile:
    def test_a_pisinger_instance_goes_from_disk_to_a_cycle_count(self):
        item = read_knapsack("tests/fixtures/knapsack/pisinger_six_items.kp")
        report = cost(generate(item, budget=Budget(60)))
        assert report["total"] > 0
        assert report["unit_label"] == "cycles"

    def test_the_bound_the_file_states_is_the_optimum_of_what_was_read(self):
        item = read_knapsack("tests/fixtures/knapsack/pisinger_six_items.kp")
        assert item.optimum == best_value(item)

    def test_costing_it_twice_gives_the_same_number(self):
        # Nothing here depends on timing or on a solver's tie-breaking, which
        # is the whole reason this route is analytic.
        item = read_knapsack("tests/fixtures/knapsack/pisinger_six_items.kp")
        first = cost(generate(item, budget=Budget(60)))["total"]
        second = cost(generate(item, budget=Budget(60)))["total"]
        assert first == second
