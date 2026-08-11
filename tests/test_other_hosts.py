"""Quantum breadth-first search, interior point methods, and the Cade family.

Two further host algorithms and one deliberate boundary. The host algorithms
are the real test of the abstraction: if adding them meant new plumbing rather
than new lemmas, the design was wrong.
"""

from __future__ import annotations

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.cost import UnitMismatch, exact
from hybrid_benchmarking.provenance import Bound, Unit
from hybrid_benchmarking.routines.maxflow import (
    dinic_cost,
    grover_cycles,
    layer_cost,
    qbfs_cost,
)
from hybrid_benchmarking.routines.qipm import (
    newton_system_cycles,
    tomography_repetitions,
)


class TestMaxFlowReusesTheSameLemma:
    """The layer cost is the shared iteration count, not a second copy."""

    def test_it_is_find_all_not_find_one(self):
        from hybrid_benchmarking.routines.amplification import (
            qsearch_all_iterations,
        )

        vertices, marked = 300, 4
        assert layer_cost(vertices, marked) == pytest.approx(
            grover_cycles(vertices) * qsearch_all_iterations(vertices, marked)
        )

    def test_missing_one_vertex_would_be_cheaper_and_wrong(self):
        """Finding all of a layer costs more than finding one of it, which is
        why the algorithm cannot use the cheaper count."""
        from hybrid_benchmarking.routines.amplification import qsearch_iterations

        vertices, marked = 300, 4
        assert qsearch_all_iterations_gt(vertices, marked)

    def test_a_grover_iteration_is_linear_in_the_vertex_count(self):
        assert grover_cycles(150) == 300
        assert grover_cycles(1000) == 2000


def qsearch_all_iterations_gt(vertices, marked):
    from hybrid_benchmarking.routines.amplification import (
        qsearch_all_iterations,
        qsearch_iterations,
    )
    return qsearch_all_iterations(vertices, marked) > \
        qsearch_iterations(vertices, marked)


class TestDinic:
    LAYERS = [1, 3, 5, 2]
    PHASES = [[1, 3, 5, 2], [1, 2, 4]]

    def test_a_sweep_is_the_sum_of_its_layers(self):
        assert qbfs_cost(300, self.LAYERS) == pytest.approx(
            sum(layer_cost(300, t) for t in self.LAYERS)
        )

    def test_the_algorithm_is_the_sum_of_its_sweeps(self):
        assert dinic_cost(300, self.PHASES) == pytest.approx(
            sum(qbfs_cost(300, layers) for layers in self.PHASES)
        )

    def test_empty_layers_cost_nothing(self):
        assert qbfs_cost(300, [0, 0]) == 0

    def test_bigger_graphs_cost_more(self):
        assert qbfs_cost(3000, self.LAYERS) > qbfs_cost(300, self.LAYERS)

    def test_it_offers_gates_and_cycles_and_they_agree(self):
        """Only because the paper assumes one cycle is one gate -- which is
        recorded on the gate count rather than left implicit."""
        routine = hb.get("Dinic")
        gates = routine.evaluate(Unit.GATES, X=300, phases=self.PHASES)
        cycles = routine.evaluate(Unit.CYCLES, X=300, phases=self.PHASES)
        assert gates.value == pytest.approx(cycles.value)
        assert any("one cycle is one gate" in a
                   for a in gates.provenance.assumptions)
        assert not any("one cycle is one gate" in a
                       for a in cycles.provenance.assumptions)

    def test_the_logged_layer_sizes_are_the_input(self):
        assert "phases" in hb.get("Dinic").parameters
        with pytest.raises(ValueError, match="missing parameters: phases"):
            hb.get("Dinic").evaluate(Unit.CYCLES, X=300)


class TestInteriorPointReadout:
    """Tomography is the structural bottleneck, and it is now visible."""

    def test_it_scales_as_dimension_over_precision_squared(self):
        base = tomography_repetitions(1000, 0.1)
        assert tomography_repetitions(2000, 0.1) == pytest.approx(
            base * (1999 / 999), rel=1e-9
        )
        assert tomography_repetitions(1000, 0.01) == pytest.approx(
            base * 100, rel=1e-9
        )

    def test_a_one_dimensional_state_has_nothing_to_read(self):
        with pytest.raises(ValueError, match="vacuous"):
            tomography_repetitions(1, 0.1)

    def test_the_readout_multiplies_the_solve(self):
        """Every repetition the readout demands runs the whole solver again."""
        solve_and_read = newton_system_cycles(
            dimension=1000, sparsity=50, kappa=100.0, epsilon=0.1
        )
        assert solve_and_read > tomography_repetitions(1000, 0.1)

    def test_it_is_a_lower_bound_under_benevolent_assumptions(self):
        cost = hb.get("IPM/mnes").cost(Unit.CYCLES)
        assert cost.provenance.bound is Bound.LOWER
        assert any("single iteration" in a
                   for a in cost.provenance.assumptions)

    def test_both_newton_formulations_are_carried(self):
        assert {i.name for i in hb.get("IPM").implementations} == {"mnes", "oss"}

    def test_both_are_built_on_the_shared_chebyshev_entry(self):
        """Not a second copy of it: Lemma 1 there is Lemma 16 here."""
        for name in ("mnes", "oss"):
            built = hb.get("IPM/" + name).built_from
            assert "QLS-Chebyshev" in built
            assert "Tomography" in built

    def test_harder_systems_cost_more(self):
        base = dict(N=1000, d=50, kappa=100.0, epsilon=0.1)
        worse = dict(base, kappa=1000.0)
        assert newton_system_cycles(
            dimension=worse["N"], sparsity=worse["d"], kappa=worse["kappa"],
            epsilon=worse["epsilon"],
        ) > newton_system_cycles(
            dimension=base["N"], sparsity=base["d"], kappa=base["kappa"],
            epsilon=base["epsilon"],
        )


class TestTheCadeBoundary:
    """The most important naming decision in the merge."""

    def test_subroutine_calls_are_their_own_unit(self):
        assert Unit.SUBROUTINE_CALLS is not Unit.QUERIES
        assert str(Unit.SUBROUTINE_CALLS) == "subroutine calls"

    def test_they_cannot_be_added_to_oracle_queries(self):
        instrumented = exact(1, Unit.SUBROUTINE_CALLS)
        analytic = exact(1, Unit.QUERIES)
        with pytest.raises(UnitMismatch, match="cannot add"):
            instrumented + analytic

    def test_the_family_is_registered(self):
        for name in ("Cade-search", "Cade-max", "Cade-amplitude",
                     "Cade-linalg"):
            assert hb.get(name).summary

    def test_they_carry_no_formulas_because_they_are_measured(self):
        """Their counts come from instrumenting a run. Registering an
        invented formula would be worse than registering none."""
        for name in ("Cade-search", "Cade-max"):
            assert hb.get(name).units == ()

    def test_the_distinction_is_stated_where_a_reader_will_meet_it(self):
        assert "not" in hb.get("Cade-search").summary.lower()
        assert "oracle" in hb.get("Cade-search").summary.lower()


class TestTheAbstractionHeld:
    """Adding two host algorithms meant lemmas and probes, not plumbing."""

    def test_three_host_algorithms_share_one_iteration_count(self):
        """The simplex, maximum flow and knapsack search all reach the same
        expression -- which was the argument for building this at all."""
        from hybrid_benchmarking.routines import amplification, maxflow, simplex

        assert simplex.qsearch_iterations is amplification.qsearch_iterations
        assert maxflow.qsearch_all_iterations is \
            amplification.qsearch_all_iterations

    def test_every_registered_cost_declares_a_bound_direction(self):
        for impl in hb.all_implementations():
            for cost in impl.costs.values():
                assert cost.provenance.bound in tuple(Bound)

    def test_every_registered_cost_cites_a_source(self):
        for impl in hb.all_implementations():
            for cost in impl.costs.values():
                assert cost.provenance.sources, impl.path
