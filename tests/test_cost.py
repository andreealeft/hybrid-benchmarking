"""The cost algebra: units, bound directions, and how they survive composition."""

from __future__ import annotations

import pytest
import sympy as sp

import hybrid_benchmarking as hb
from hybrid_benchmarking import symbols as S
from hybrid_benchmarking.cost import Cost, UnitMismatch, exact, lower_bound
from hybrid_benchmarking.provenance import (
    Bound,
    Derivation,
    Provenance,
    Unit,
    combine_bound,
    combine_derivation,
)


class TestBoundArithmetic:
    def test_like_with_like(self):
        assert combine_bound(Bound.LOWER, Bound.LOWER) is Bound.LOWER
        assert combine_bound(Bound.EXACT, Bound.EXACT) is Bound.EXACT

    def test_exact_plus_bound_is_a_bound(self):
        assert combine_bound(Bound.EXACT, Bound.LOWER) is Bound.LOWER
        assert combine_bound(Bound.UPPER, Bound.EXACT) is Bound.UPPER

    def test_opposing_bounds_give_neither(self):
        """A lower bound plus an upper bound is not a bound in any direction."""
        assert combine_bound(Bound.LOWER, Bound.UPPER) is Bound.ESTIMATE

    def test_estimates_are_contagious(self):
        assert combine_bound(Bound.EXACT, Bound.ESTIMATE) is Bound.ESTIMATE


class TestDerivationArithmetic:
    def test_the_weaker_derivation_wins(self):
        assert combine_derivation(
            Derivation.ANALYTIC, Derivation.EXTRAPOLATED
        ) is Derivation.EXTRAPOLATED

    def test_analytic_on_analytic_stays_analytic(self):
        assert combine_derivation(
            Derivation.ANALYTIC, Derivation.ANALYTIC
        ) is Derivation.ANALYTIC


class TestUnits:
    def test_cannot_add_different_things(self):
        gates = exact(S.N, Unit.GATES)
        queries = exact(S.N, Unit.QUERIES)
        with pytest.raises(UnitMismatch, match="cannot add"):
            gates + queries

    def test_iterations_multiply_into_the_thing_repeated(self):
        iterations = exact(sp.Integer(7), Unit.ITERATIONS)
        per_iteration = exact(sp.Integer(5), Unit.QUERIES)
        total = iterations * per_iteration
        assert total.unit is Unit.QUERIES
        assert total.evaluate().value == pytest.approx(35)

    def test_two_absolute_counts_cannot_be_multiplied(self):
        gates = exact(sp.Integer(2), Unit.GATES)
        queries = exact(sp.Integer(3), Unit.QUERIES)
        with pytest.raises(UnitMismatch, match="must be a multiplier"):
            gates * queries

    def test_plain_numbers_scale_without_changing_the_unit(self):
        doubled = exact(S.N, Unit.GATES) * 2
        assert doubled.unit is Unit.GATES
        assert doubled.evaluate(N=10).value == pytest.approx(20)


class TestCompositionCarriesProvenance:
    def test_a_sum_is_as_weak_as_its_weakest_part(self):
        tight = exact(sp.Integer(1), Unit.GATES, source="A")
        loose = lower_bound(sp.Integer(1), Unit.GATES, source="B")
        total = tight + loose
        assert total.provenance.bound is Bound.LOWER
        assert set(total.provenance.sources) == {"A", "B"}

    def test_assumptions_accumulate_without_duplicating(self):
        a = Cost(sp.Integer(1), Unit.GATES,
                 Provenance.of(source="X", assumptions=("no error correction",)))
        b = Cost(sp.Integer(1), Unit.GATES,
                 Provenance.of(source="Y", assumptions=("no error correction",)))
        assert (a + b).provenance.assumptions == ("no error correction",)

    def test_validity_domains_union(self):
        left = hb.get("QSearch").cost(Unit.ITERATIONS)
        right = hb.get("QSearchAll").cost(Unit.ITERATIONS)
        combined = left + right
        assert len(combined.validity.conditions) >= len(left.validity.conditions)


class TestPresentation:
    def test_a_symbolic_cost_reports_its_parameters(self):
        cost = hb.get("HamSim").cost(Unit.QUERIES)
        assert set(cost.parameters) == {"d", "A_max", "t_sim", "epsilon"}

    def test_formula_renders(self):
        assert "Sum" in repr(hb.get("QAA").cost(Unit.ITERATIONS).expr)

    def test_partial_substitution_keeps_it_symbolic(self):
        cost = hb.get("HamSim").cost(Unit.QUERIES).subs(d=4, A_max=1)
        assert set(cost.parameters) == {"t_sim", "epsilon"}


class TestTwoConstructionsOfOneRoutine:
    """Hamiltonian simulation is two algorithms, not one seen two ways.

    The simplex gate counts come from Berry et al.'s fractional-query
    reduction; the linear solver query counts come from Low-Chuang
    qubitization. A composition that silently picked the wrong one would be
    wrong in a way no test of either construction alone would catch.
    """

    def test_the_routine_holds_both(self):
        assert {i.name for i in hb.get("HamSim").implementations} == {
            "qubitization", "berry"
        }

    def test_asking_without_disambiguating_refuses(self):
        with pytest.raises(ValueError, match="say which"):
            hb.get("HamSim").cost()

    def test_the_unit_picks_the_construction(self):
        """Only one construction offers each unit, so the unit is enough."""
        assert hb.get("HamSim").cost(Unit.QUERIES).unit is Unit.QUERIES
        assert hb.get("HamSim").cost(Unit.GATES).unit is Unit.GATES

    def test_addressing_by_path(self):
        impl = hb.get("HamSim/qubitization")
        assert impl.path == "HamSim/qubitization"
        assert impl.units == (Unit.QUERIES,)

    def test_gate_counts_are_lower_bounds_query_counts_are_exact(self):
        assert hb.get("HamSim/berry").cost().provenance.bound is Bound.LOWER
        assert hb.get("HamSim/qubitization").cost().provenance.bound is Bound.EXACT

    def test_the_two_take_different_parameters(self):
        """Berry's bound needs the 1-norm; qubitization never sees it."""
        assert "A_1" in hb.get("HamSim/berry").parameters
        assert "A_1" not in hb.get("HamSim/qubitization").parameters


class TestQubitization:
    def test_long_evolutions_cost_linearly_in_time(self):
        short = hb.get("HamSim").evaluate(
            Unit.QUERIES, d=4, A_max=1, t_sim=100, epsilon=1e-8).value
        long = hb.get("HamSim").evaluate(
            Unit.QUERIES, d=4, A_max=1, t_sim=200, epsilon=1e-8).value
        assert long == pytest.approx(2 * short, rel=0.01)

    def test_tighter_precision_costs_more(self):
        loose = hb.get("HamSim").evaluate(
            Unit.QUERIES, d=4, A_max=1, t_sim=0.001, epsilon=1e-3).value
        tight = hb.get("HamSim").evaluate(
            Unit.QUERIES, d=4, A_max=1, t_sim=0.001, epsilon=1e-12).value
        assert tight > loose


class TestBerryFractionalQuery:
    PARAMS = dict(d=4, A_max=1.0, A_1=3.0, t_sim=10.0, epsilon=1e-3)

    def test_it_produces_a_gate_count(self):
        cost = hb.get("HamSim/berry").evaluate(**self.PARAMS)
        assert cost.unit is Unit.GATES
        assert cost.value > 0

    def test_longer_evolutions_cost_more(self):
        near = dict(self.PARAMS, t_sim=10.0)
        far = dict(self.PARAMS, t_sim=100.0)
        assert (hb.get("HamSim/berry").evaluate(**far).value
                > hb.get("HamSim/berry").evaluate(**near).value)

    def test_tighter_precision_costs_more(self):
        loose = dict(self.PARAMS, epsilon=1e-3)
        tight = dict(self.PARAMS, epsilon=1e-6)
        assert (hb.get("HamSim/berry").evaluate(**tight).value
                > hb.get("HamSim/berry").evaluate(**loose).value)

    def test_segment_count_grows_logarithmically(self):
        from hybrid_benchmarking.routines.hamsim import queries_per_segment

        coarse = queries_per_segment(1e-4)
        fine = queries_per_segment(1e-8)
        assert fine > coarse
        assert fine < 4 * coarse  # log growth, not polynomial


class TestCapabilityTable:
    def test_it_reports_what_formulas_exist(self):
        table = hb.capability_table()
        assert "QSearch" in table
        assert "HamSim" in table
