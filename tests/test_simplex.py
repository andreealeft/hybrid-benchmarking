"""The quantum simplex subroutines.

Every one bottoms out in a linear solve, so what these tests mostly check is
that the right solver is reached at the right precision -- the two things that
are easy to get wrong and impossible to notice from a single number.
"""

from __future__ import annotations

import math

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.provenance import Bound, Unit
from hybrid_benchmarking.routines.simplex import (
    can_enter_nfn,
    can_enter_nfp,
    find_column_dantzig,
    find_column_random,
    find_column_steepest_edge,
    find_row,
    is_optimal,
    is_unbounded,
    red_cost,
    simplex_iteration,
)

CORE = ("RedCost", "CanEnterNFN", "CanEnterNFP")
MAIN = ("IsOptimal", "IsUnbounded", "FindRow")
RULES = ("steepest-edge", "dantzig", "random")

INSTANCE = dict(
    d=4, kappa=10.0, epsilon=1e-3, delta=1e-3, A_1=3.0, A_max=1.0,
    n=200, m=50, t=5, c_max=2.0, u_norm=1.5,
)


def _gates(name, **overrides):
    params = dict(INSTANCE, **overrides)
    routine = hb.get(name)
    wanted = set(routine.parameters)
    return routine.evaluate(
        Unit.GATES, **{k: v for k, v in params.items() if k in wanted}
    ).value


class TestEverythingIsThere:
    @pytest.mark.parametrize("name", CORE + MAIN)
    def test_registered_and_costed_in_gates(self, name):
        assert hb.get(name).cost(Unit.GATES).unit is Unit.GATES

    @pytest.mark.parametrize("rule", RULES)
    def test_both_multi_rule_routines_carry_every_rule(self, rule):
        assert rule in {i.name for i in hb.get("FindColumn").implementations}
        assert rule in {i.name for i in hb.get("SimplexIter").implementations}

    @pytest.mark.parametrize("name", CORE + MAIN)
    def test_all_of_them_are_lower_bounds(self, name):
        """Non-leading terms are dropped and the assumptions favour the quantum
        side, so these are minima -- the real cost is higher."""
        assert hb.get(name).cost(Unit.GATES).provenance.bound is Bound.LOWER

    @pytest.mark.parametrize("name", CORE + MAIN)
    def test_positive_and_finite(self, name):
        value = _gates(name)
        assert value > 0 and math.isfinite(value)


class TestTheyReachTheRightSolver:
    """Berry's construction, never qubitization -- the substitution the
    implementation layer exists to prevent."""

    def test_the_solver_underneath_is_the_berry_one(self):
        for name in ("RedCost", "IsUnbounded", "FindRow"):
            assert "QLS-Fourier/via-berry" in \
                hb.get(name).implementation().built_from

    def test_no_simplex_entry_offers_a_query_count(self):
        for name in CORE + MAIN + ("FindColumn", "SimplexIter"):
            assert Unit.QUERIES not in hb.get(name).units


class TestPrecisionPropagation:
    """One tolerance in, several different tolerances handed to the solver."""

    def test_no_false_positives_costs_more_than_no_false_negatives(self):
        """Both call the solver at 0.1 eps / sqrt(2), so the difference is
        entirely the factor of nine between their amplitude estimation
        precisions.  Not exactly nine: both prefactors carry a trailing minus
        one, so the ratio only approaches it as the tolerance tightens.
        """
        loose = _gates("CanEnterNFP", epsilon=1e-2) / _gates("CanEnterNFN",
                                                             epsilon=1e-2)
        tight = _gates("CanEnterNFP", epsilon=1e-6) / _gates("CanEnterNFN",
                                                             epsilon=1e-6)
        assert loose > 9.0
        assert tight == pytest.approx(9.0, rel=1e-5)
        assert tight < loose

    def test_reduced_cost_is_the_cheapest_thing_here(self):
        """It is one solve at the caller's own precision, with no prefactor."""
        assert _gates("RedCost") < min(_gates(n) for n in CORE[1:] + MAIN)

    @pytest.mark.parametrize("name", CORE + MAIN)
    def test_tighter_tolerance_costs_more(self, name):
        loose = _gates(name, epsilon=1e-2, delta=1e-2)
        tight = _gates(name, epsilon=1e-4, delta=1e-4)
        assert tight > loose

    @pytest.mark.parametrize("name", CORE + MAIN)
    def test_worse_conditioning_costs_more(self, name):
        assert _gates(name, kappa=50.0) > _gates(name, kappa=10.0)


class TestProblemShape:
    def test_more_columns_make_the_optimality_test_harder(self):
        assert _gates("IsOptimal", n=800) > _gates("IsOptimal", n=200)

    def test_optimality_scales_as_the_square_root_of_the_columns(self):
        """The 24 sqrt(n - m) prefactor of Lemma 8, visible end to end."""
        small = _gates("IsOptimal", n=450, m=50)   # 400 non-basic
        large = _gates("IsOptimal", n=1650, m=50)  # 1600 non-basic
        assert large / small == pytest.approx(2.0, rel=0.02)

    def test_a_basis_using_every_column_is_refused(self):
        with pytest.raises(ValueError, match="no meaning"):
            _gates("IsOptimal", n=50, m=50)


class TestFindRowSearchesForNothing:
    def test_it_is_costed_at_zero_marked_elements(self):
        """Lemma 11 evaluates the search at r = 0, where it finds nothing and
        runs its schedule to exhaustion.  That is the dominant step, and it is
        why a marked count of zero has to be a meaningful input."""
        from hybrid_benchmarking.routines.amplification import qsearch_iterations

        exhausted = qsearch_iterations(INSTANCE["m"], 0)
        found = qsearch_iterations(INSTANCE["m"], 1)
        assert exhausted > found

    def test_the_pivot_direction_norm_enters_linearly(self):
        doubled = _gates("FindRow", u_norm=3.0)
        single = _gates("FindRow", u_norm=1.5)
        assert doubled / single == pytest.approx(2.0, rel=0.02)


class TestPivotingRules:
    def test_the_two_sophisticated_rules_cost_more_than_random(self):
        """They run minimum finding over all columns rather than one search."""
        cheap = find_column_random(INSTANCE)
        for rule in (find_column_steepest_edge, find_column_dantzig):
            assert rule(INSTANCE) > cheap

    def test_dantzig_and_steepest_edge_differ_only_in_solver_precision(self):
        """Same prefactor, same minimum-finding cost; Dantzig divides the
        solver tolerance by the pivot direction's norm as well."""
        with_unit_norm = dict(INSTANCE, u_norm=1.0)
        assert find_column_dantzig(with_unit_norm) == pytest.approx(
            find_column_steepest_edge(with_unit_norm), rel=1e-9
        )

    def test_a_larger_pivot_norm_makes_dantzig_dearer(self):
        assert find_column_dantzig(dict(INSTANCE, u_norm=4.0)) > \
            find_column_dantzig(dict(INSTANCE, u_norm=1.0))


class TestTheWholeIteration:
    @pytest.mark.parametrize("rule", RULES)
    def test_an_iteration_is_its_four_parts(self, rule):
        parts = {
            "steepest-edge": find_column_steepest_edge,
            "dantzig": find_column_dantzig,
            "random": find_column_random,
        }[rule]
        expected = (is_optimal(INSTANCE) + parts(INSTANCE)
                    + is_unbounded(INSTANCE) + find_row(INSTANCE))
        assert simplex_iteration(INSTANCE, rule) == pytest.approx(expected)

    @pytest.mark.parametrize("rule", RULES)
    def test_it_evaluates_through_the_registry(self, rule):
        assert _gates("SimplexIter/" + rule) > 0

    def test_choosing_the_pivoting_rule_changes_the_iteration(self):
        assert _gates("SimplexIter/steepest-edge") != \
            _gates("SimplexIter/random")

    def test_the_entering_column_dominates_under_steepest_edge(self):
        """Which is why the thesis plots that rule: it is where the cost is."""
        total = _gates("SimplexIter/steepest-edge")
        assert find_column_steepest_edge(INSTANCE) > 0.5 * total
