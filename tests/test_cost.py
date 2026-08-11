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


class TestHamiltonianSimulation:
    def test_long_evolutions_cost_linearly_in_time(self):
        short = hb.get("HamSim").evaluate(
            d=4, A_max=1, t_sim=100, epsilon=1e-8).value
        long = hb.get("HamSim").evaluate(
            d=4, A_max=1, t_sim=200, epsilon=1e-8).value
        assert long == pytest.approx(2 * short, rel=0.01)

    def test_tighter_precision_costs_more(self):
        loose = hb.get("HamSim").evaluate(
            d=4, A_max=1, t_sim=0.001, epsilon=1e-3).value
        tight = hb.get("HamSim").evaluate(
            d=4, A_max=1, t_sim=0.001, epsilon=1e-12).value
        assert tight > loose


class TestCapabilityTable:
    def test_it_reports_what_formulas_exist(self):
        table = hb.capability_table()
        assert "QSearch" in table
        assert "HamSim" in table
