"""The interior point route, and the Newton system each iteration faces.

The strongest thing available here is that two independently written solvers
have to agree.  The simplex walks the boundary and the interior point method
does not go near it, they share nothing but the standard form, and they must
reach the same optimum -- so that is asserted on every program in this file,
and it is what makes the logs believable before any cost is computed.

After that the claims are about what a record is and what is in it.  A record is
an iteration rather than a solve, because Mehrotra's predictor and corrector
reuse one system; the dimension is the constraint count; and the density comes
out of order ``m``, which is what the ``IPM/mnes`` entry already assumes about
it and is here confirmed rather than restated.

One column is a decision made without the source to hand.  The construction is
Binkowski's "modified" normal equations, and a diagonal rescaling of a system
leaves its dimension and density alone but not its condition number -- which the
cost is quadratic in.  So both are logged and the route consumes the unmodified
one -- not because it is the safer number, since equilibration lowers it on a
vertex cover and raises it on an independent set and there is a test for exactly
that, but because inventing a modification the source may not mean is worse than
declaring which reading was taken.
"""

from __future__ import annotations

import numpy as np
import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.ipm import _pattern_sparsity
from hybrid_benchmarking.classical.ipm import solve as interior_point
from hybrid_benchmarking.classical.lp import (
    from_linear_program,
    independent_set,
    standard_form,
    vertex_cover,
)
from hybrid_benchmarking.classical.simplex import solve as pivot
from hybrid_benchmarking.instances import Graph
from hybrid_benchmarking.instances.dimacs import read_max_flow
from hybrid_benchmarking.instances.mps import read as read_mps
from hybrid_benchmarking.provenance import Bound, Derivation

CHOSEN = {"epsilon": 1e-3}


def graph(vertices: int, edges) -> Graph:
    return Graph(name="g", source="(hand built)", layout="dimacs-edge",
                 vertices=vertices, edges=tuple(sorted(edges)))


CYCLE_5 = graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
SQUARE = graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
STAR_5 = graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)])

PROGRAMS = [
    ("cover of a five-cycle", vertex_cover(CYCLE_5)),
    ("cover of a square", vertex_cover(SQUARE)),
    ("cover of a star", vertex_cover(STAR_5)),
    ("independent set of a five-cycle", independent_set(CYCLE_5)),
]


class TestTwoSolversMustAgree:
    @pytest.mark.parametrize("label,model", PROGRAMS,
                             ids=[label for label, _ in PROGRAMS])
    def test_the_boundary_and_the_interior_reach_the_same_optimum(
            self, label, model):
        form = standard_form(model)
        walked = pivot(form, Budget(60))
        approached = interior_point(form, Budget(60))
        assert walked.status is Status.COMPLETE
        assert approached.status is Status.COMPLETE
        assert approached.result["objective"] == pytest.approx(
            walked.result["objective"], abs=1e-5)

    def test_they_agree_on_the_worked_mps_example_too(self):
        form = standard_form(
            from_linear_program(read_mps("tests/fixtures/mps/testprob.mps")))
        assert interior_point(form, Budget(60)).result["objective"] == \
            pytest.approx(16.0, abs=1e-5)


class TestWhatARecordIs:
    def test_a_record_is_an_iteration_not_a_solve(self):
        # The predictor and the corrector reuse one system, so an iteration
        # faces one, which is what the route costs per record.
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert len(run.records) == run.result["iterations"]

    def test_the_dimension_is_the_constraint_count(self):
        form = standard_form(vertex_cover(CYCLE_5))
        run = interior_point(form, Budget(60))
        assert all(entry["N"] == form.rows for entry in run.records)

    def test_the_density_comes_out_of_order_m_as_the_entry_assumes(self):
        # The registered cost assumes the system inherits a density of order m
        # rather than the constraint matrix's. That is a claim about instances,
        # and this is where it is checked rather than repeated.
        form = standard_form(vertex_cover(CYCLE_5))
        run = interior_point(form, Budget(60))
        density = run.records[0]["d"]
        assert density == _pattern_sparsity(form.matrix)
        assert density > _column_sparsity(form.matrix)

    def test_the_condition_number_worsens_as_the_iterates_approach_the_boundary(
            self):
        # The whole negative result rests on kappa, and kappa is a property of
        # the path rather than of the instance -- which is why it has to be
        # logged rather than derived.
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert run.records[-1]["kappa"] > run.records[0]["kappa"]

    def test_the_duality_gap_is_logged_so_the_path_can_be_read(self):
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        gaps = [entry["duality_gap"] for entry in run.records]
        assert gaps == sorted(gaps, reverse=True)


def _column_sparsity(matrix: np.ndarray) -> int:
    return int(np.max(np.count_nonzero(matrix, axis=0)))


class TestTheConditionNumberThatIsConsumed:
    def test_the_column_the_route_reads_is_the_unmodified_systems(self):
        # The precise claim, checkable without the paper: kappa is the
        # condition number of A D A' as formed, with nothing done to it.
        form = standard_form(vertex_cover(CYCLE_5))
        run = interior_point(form, Budget(60))
        first = run.records[0]

        from hybrid_benchmarking.classical.ipm import _starting_point

        primal, _, slack = _starting_point(form.matrix, form.rhs,
                                           form.objective)
        system = (form.matrix * (primal / slack)) @ form.matrix.T
        assert first["kappa"] == pytest.approx(np.linalg.cond(system))

    def test_equilibrating_is_not_reliably_an_improvement(self):
        # Which is exactly why the modification is not invented here: if
        # rescaling always helped, taking the unrescaled number would be a
        # safe over-estimate, and it is not.
        lowered = interior_point(standard_form(vertex_cover(CYCLE_5)),
                                 Budget(60)).records
        raised = interior_point(standard_form(independent_set(CYCLE_5)),
                                Budget(60)).records
        assert any(e["kappa_equilibrated"] < e["kappa"] for e in lowered)
        assert any(e["kappa_equilibrated"] > e["kappa"] for e in raised)

    def test_both_readings_are_logged_so_the_choice_can_be_revisited(self):
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert all({"kappa", "kappa_equilibrated"} <= set(entry)
                   for entry in run.records)

    def test_the_choice_is_recorded_on_every_cost_it_produces(self):
        report = cost(generate(CYCLE_5, "vertex-cover",
                               "quantum-interior-point", Budget(60)), CHOSEN)
        joined = " ".join(report["assumptions"])
        assert "kappa_equilibrated" in joined
        assert "not established as favourable to the quantum side" in joined

    def test_a_condition_number_is_never_below_one(self):
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert all(entry["kappa"] >= 1.0 for entry in run.records)


class TestCostingIt:
    def test_it_comes_out_in_cycles_and_logged(self):
        report = cost(generate(CYCLE_5, "vertex-cover",
                               "quantum-interior-point", Budget(60)), CHOSEN)
        assert report["unit"] == "CYCLES"
        assert report["derivation"] == str(Derivation.LOGGED)
        assert report["bound"] == str(Bound.LOWER)

    def test_the_two_routes_through_one_problem_are_not_added_up(self):
        # They are different units by construction, and that is what stops
        # anyone tabulating them as though they were comparable.
        simplex = cost(generate(CYCLE_5, "vertex-cover", "quantum-simplex",
                                Budget(60)), {"epsilon": 1e-3, "delta": 1e-3})
        interior = cost(generate(CYCLE_5, "vertex-cover",
                                 "quantum-interior-point", Budget(60)), CHOSEN)
        assert simplex["unit"] != interior["unit"]

    def test_the_benevolent_assumptions_of_the_source_survive(self):
        report = cost(generate(CYCLE_5, "vertex-cover",
                               "quantum-interior-point", Budget(60)), CHOSEN)
        joined = " ".join(report["assumptions"])
        assert "no amplitude amplification overhead" in joined
        assert "converges in a single iteration" in joined


class TestWhenItCannotRun:
    def test_a_redundant_row_is_presolved_away_rather_than_refused(self):
        # Some programs are redundant by construction, not by accident -- a
        # maximum-flow model's conservation rows always sum to zero -- so
        # refusing them would refuse the whole problem.
        from hybrid_benchmarking.classical.lp import Model

        # Equalities, because an inequality gets a surplus column of its own and
        # so two proportional inequalities are independent once converted.
        model = Model(columns=3, objective=np.ones(3))
        model.add_row((0, 1), (1.0, 1.0), "E", 1.0)
        model.add_row((1, 2), (1.0, 1.0), "E", 1.0)
        model.add_row((0, 1, 2), (1.0, 2.0, 1.0), "E", 2.0)  # the sum of both
        run = interior_point(standard_form(model), Budget(60))
        assert run.status is Status.COMPLETE
        assert all(entry["N"] == 2 for entry in run.records)
        assert any("redundant constraint row" in note
                   for note in run.assumptions)

    def test_the_logged_dimension_is_the_system_that_was_actually_solved(self):
        from hybrid_benchmarking.classical.lp import maximum_flow

        network = read_max_flow("tests/fixtures/tiny.max")
        form = standard_form(maximum_flow(network))
        run = interior_point(form, Budget(60))
        # One conservation row is always dependent, so the Newton system is one
        # smaller than the program's row count.
        assert run.records[0]["N"] == form.rows - 1

    def test_rows_that_contradict_each_other_are_refused(self):
        from hybrid_benchmarking.classical.lp import Model

        model = Model(columns=3, objective=np.ones(3))
        model.add_row((0, 1), (1.0, 1.0), "E", 1.0)
        model.add_row((1, 2), (1.0, 1.0), "E", 1.0)
        model.add_row((0, 1, 2), (1.0, 2.0, 1.0), "E", 5.0)  # should be 2
        run = interior_point(standard_form(model), Budget(60))
        assert run.status is Status.FAILED
        assert "inconsistent" in run.reason

    def test_an_exhausted_budget_keeps_whatever_iterations_it_had(self):
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(1e-9))
        assert run.status is Status.FAILED and not run.records
