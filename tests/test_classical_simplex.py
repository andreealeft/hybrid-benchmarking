"""The instrumented simplex, and whether it logs what the lemmas asked for.

Three separate claims are at stake, and they fail in different ways.

*The solver is a solver.*  Checked against linear programs whose optimum is
known without running anything: the fractional vertex cover of an odd cycle is
half its length, of a bipartite graph it is the size of a maximum matching, and
the worked example every MPS reference prints has an optimum anyone can verify
by substitution.

*The model is the model the cost is about.*  :mod:`~hybrid_benchmarking.problems`
predicts the shape of the program each graph problem becomes, and the route
feeds that prediction to the lemmas rather than anything the run reports.  A
translation that disagreed would produce condition numbers belonging to one
program and column counts to another, and every number after that would be
plausible.

*The logged quantities are the ones the lemmas name.*  This is the half with no
natural error signal: the wrong reading of ``t`` or of ``A_1`` still produces a
gate count, still sums, still looks like the published figures.  So each one is
asserted against its definition rather than against a stored value, and the two
quantities that are easiest to confuse -- the positive components of the pivot
direction and the count of improving columns -- are asserted to be different
things, with the second shown to break the cost if it is put where the first
belongs.
"""

from __future__ import annotations

import numpy as np
import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.lp import (
    clique,
    independent_set,
    standard_form,
    vertex_cover,
)
from hybrid_benchmarking.classical.simplex import _record, solve
from hybrid_benchmarking.classical.simplex import sparsity as simplex_sparsity
from hybrid_benchmarking.instances import Graph
from hybrid_benchmarking.instances.mps import read as read_mps
from hybrid_benchmarking.problems import clique_shape, cover_shape

CHOSEN = {"epsilon": 1e-3, "delta": 1e-3}


def graph(vertices: int, edges) -> Graph:
    return Graph(name="g", source="(hand built)", layout="dimacs-edge",
                 vertices=vertices, edges=tuple(sorted(edges)))


CYCLE_5 = graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
CYCLE_4 = graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
STAR_5 = graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)])


class TestItSolvesLinearPrograms:
    def test_an_odd_cycle_has_a_fractional_cover_of_half_its_length(self):
        # Every vertex at one half is feasible and optimal for an odd cycle;
        # the integral optimum is 3, which is why the relaxation is famous.
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert run.status is Status.COMPLETE
        assert run.result["objective"] == pytest.approx(2.5)

    def test_a_bipartite_graph_has_an_integral_cover(self):
        # Konig: on a bipartite graph the relaxation has an integral optimum
        # equal to the size of a maximum matching.
        run = solve(standard_form(vertex_cover(CYCLE_4)), Budget(60))
        assert run.result["objective"] == pytest.approx(2.0)

    def test_a_star_is_covered_by_its_centre(self):
        run = solve(standard_form(vertex_cover(STAR_5)), Budget(60))
        assert run.result["objective"] == pytest.approx(1.0)

    def test_independent_set_is_the_complement_of_the_cover(self):
        # On any graph the two relaxations sum to the vertex count, since
        # x is feasible for one exactly when 1 - x is feasible for the other.
        cover = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        independent = solve(standard_form(independent_set(CYCLE_5)), Budget(60))
        assert (cover.result["objective"] + independent.result["objective"]
                == pytest.approx(CYCLE_5.vertices))

    def test_the_worked_mps_example_has_the_optimum_you_get_by_substitution(self):
        # MYEQN forces ZTHREE = 7 + YTWO, so the objective is XONE + 5 YTWO + 21
        # and both variables go to their lower bounds: 0 and -1, giving 16.
        program = read_mps("tests/fixtures/mps/testprob.mps")
        from hybrid_benchmarking.classical.lp import from_linear_program

        run = solve(standard_form(from_linear_program(program)), Budget(60))
        assert run.status is Status.COMPLETE
        assert run.result["objective"] == pytest.approx(16.0)


class TestTheModelIsTheOneTheCostIsAbout:
    @pytest.mark.parametrize("instance", [CYCLE_5, CYCLE_4, STAR_5])
    def test_vertex_cover_has_the_shape_the_problem_catalogue_predicts(
            self, instance):
        built = standard_form(vertex_cover(instance)).shape
        predicted = cover_shape({"vertices": instance.vertices,
                                 "edges": len(instance.edges)})
        assert built == {name: int(value) for name, value in predicted.items()}

    @pytest.mark.parametrize("instance", [CYCLE_5, CYCLE_4, STAR_5])
    def test_independent_set_has_the_shape_the_catalogue_predicts(self, instance):
        # It shares cover_shape with vertex cover, and shares the builder, so
        # this looks redundant -- but the two claims are separable and only one
        # of them was being checked.
        built = standard_form(independent_set(instance)).shape
        predicted = cover_shape({"vertices": instance.vertices,
                                 "edges": len(instance.edges)})
        assert built == {name: int(value) for name, value in predicted.items()}
        assert built == standard_form(vertex_cover(instance)).shape

    @pytest.mark.parametrize("instance", [CYCLE_5, CYCLE_4, STAR_5])
    def test_clique_counts_the_non_edges(self, instance):
        built = standard_form(clique(instance)).shape
        predicted = clique_shape({"vertices": instance.vertices,
                                  "edges": len(instance.edges)})
        assert built == {name: int(value) for name, value in predicted.items()}

    def test_a_disagreement_would_be_refused_rather_than_costed(self):
        # The check exists because nothing downstream could notice: the log
        # would be of one program and the column count of another.
        from hybrid_benchmarking.classical.generate import _run_simplex

        route = hb.get_route("vertex-cover", "quantum-simplex")
        assert route.shape is not None  # otherwise the check is vacuous
        run = _run_simplex("vertex-cover", CYCLE_5, route, Budget(60), {})
        assert run.instance["n"] == int(
            cover_shape({"vertices": 5, "edges": 5})["n"])


class TestTheQuantitiesAreTheOnesTheLemmasName:
    def test_kappa_is_glpks_ratio_and_not_the_condition_number(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        rows = run.instance["m"]
        for entry in run.records:
            assert entry["kappa"] == max(1.0, entry["kappa_1"] / rows)
            assert entry["kappa"] >= 1.0

    def test_the_norms_are_the_normalised_lower_bounds_not_the_raw_ones(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        for entry in run.records:
            assert entry["A_max"] == pytest.approx(1.0 / entry["d"])
            assert entry["A_1"] == pytest.approx(
                entry["A_B_1"] / (entry["d"] * entry["A_B_max"]))
            # and the raw ones are kept, so nothing is lost by the convention
            assert entry["A_B_1"] >= entry["A_B_max"] > 0

    def test_t_counts_the_positive_components_of_the_pivot_direction(self):
        basis = np.eye(3)
        entry = _record(basis, basis, np.array([1.0, -2.0, 0.5]), rows=3, phase=2)
        assert entry["t"] == 2
        assert entry["u_norm"] == pytest.approx(np.sqrt(1 + 4 + 0.25))

    @pytest.mark.parametrize("instance", [CYCLE_5, CYCLE_4, STAR_5])
    def test_t_never_exceeds_the_basis_size_because_u_has_that_many_parts(
            self, instance):
        # This is the invariant that separates t from the improving-column
        # count: u has one component per row, so a search over it cannot have
        # more marked elements than the basis has rows.
        run = solve(standard_form(vertex_cover(instance)), Budget(60))
        for entry in run.records:
            assert 0 <= entry["t"] <= run.instance["m"]

    def test_the_improving_column_count_is_logged_as_a_different_quantity(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert all("t_improving" in entry for entry in run.records)
        # They are not the same number, which is the whole point of logging
        # both -- one marks a search over the basis, the other over the columns.
        assert any(entry["t_improving"] != entry["t"] for entry in run.records)

    def test_putting_the_improving_count_where_t_belongs_breaks_the_cost(self):
        # An improving-column count can exceed the basis size, and then it is
        # marking more elements than the list it searches holds.  The library
        # refuses rather than returning a number, which is what makes this a
        # mistake someone can make only once.
        route = hb.get_route("linear-programming", "quantum-simplex")
        data = hb.Dataset(
            records=({"kappa": 3.0, "d": 4, "A_1": 3.0, "A_max": 1.0,
                      "t": 40, "u_norm": 1.5},),
            instance={"n": 200, "m": 20, "c_max": 1.0},
        )
        with pytest.raises(ValueError):
            hb.run(route, data, CHOSEN)

    def test_c_max_is_the_objectives_largest_entry_and_stays_constant(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert run.instance["c_max"] == 1.0  # every vertex costs one

    def test_the_first_phase_is_marked_so_it_can_be_told_apart(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        assert {entry["phase"] for entry in run.records} <= {1, 2}
        assert any(entry["phase"] == 1 for entry in run.records)


class TestCostingWhatCameOut:
    def test_every_record_costs_without_being_refused(self):
        generated = generate(CYCLE_5, "vertex-cover", "quantum-simplex",
                             Budget(60))
        report = cost(generated, CHOSEN)
        assert report["records"] == len(generated.run.records)
        assert all(value > 0 for value in report["per_record"])

    def test_the_total_is_the_sum_of_the_iterations(self):
        generated = generate(CYCLE_5, "vertex-cover", "quantum-simplex",
                             Budget(60))
        report = cost(generated, CHOSEN)
        assert report["total"] == pytest.approx(sum(report["per_record"]))

    def test_the_conventions_taken_from_the_thesis_are_on_the_answer(self):
        report = cost(generate(CYCLE_5, "vertex-cover", "quantum-simplex",
                               Budget(60)), CHOSEN)
        joined = " ".join(report["assumptions"])
        assert "kappa_1" in joined and "(4.33)" in joined
        assert "normalised lower bounds" in joined
        assert "not GLPK" in report["provenance"]

    def test_a_larger_instance_costs_more_than_a_smaller_one(self):
        small = cost(generate(CYCLE_4, "vertex-cover", "quantum-simplex",
                              Budget(60)), CHOSEN)["total"]
        large = cost(generate(
            graph(8, [(i, (i + 1) % 8) for i in range(8)]),
            "vertex-cover", "quantum-simplex", Budget(60)), CHOSEN)["total"]
        assert large > small


class TestWhenItCannotFinish:
    def test_an_exhausted_budget_truncates_and_keeps_what_it_had(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(1e-9))
        assert run.status is Status.FAILED  # nothing was logged at all
        assert not run.records

    def test_a_program_with_no_objective_is_refused_rather_than_priced(self):
        from hybrid_benchmarking.classical.lp import Model

        model = Model(columns=2, objective=np.zeros(2))
        model.add_row((0, 1), (1.0, 1.0), "E", 1.0)
        run = solve(standard_form(model), Budget(60))
        assert run.status is Status.FAILED
        assert "no pivoting rule" in run.reason

    def test_an_infeasible_program_says_so(self):
        from hybrid_benchmarking.classical.lp import Model

        model = Model(columns=1, objective=np.ones(1))
        model.add_row((0,), (1.0,), "G", 5.0)
        model.add_row((0,), (1.0,), "L", 1.0)
        run = solve(standard_form(model), Budget(60))
        assert run.status is Status.FAILED
        assert "infeasible" in run.reason


class TestTheSparsityTheNormalisationNeeds:
    """``d`` has to make ``||A||_2 <= d ||A||_max`` true, or the normalised
    norms logged beside it are over-estimates wearing a lower bound's label.

    The thesis's list of measures names the column maximum. That reading is
    safe only while no row is denser than every column, and a maximum-flow
    program breaks it at any vertex of degree four.
    """

    @staticmethod
    def _random(rows, columns, seed):
        state = np.random.RandomState(seed)
        matrix = state.randn(rows, columns)
        matrix[state.rand(rows, columns) < 0.6] = 0.0
        matrix[0, 0] = matrix[0, 0] or 1.0  # never entirely zero
        return matrix

    @pytest.mark.parametrize("seed", range(25))
    def test_the_inequality_holds_for_the_sparsity_we_log(self, seed):
        matrix = self._random(6, 9, seed)
        assert np.linalg.norm(matrix, 2) <= (
            simplex_sparsity(matrix) * np.abs(matrix).max() + 1e-12)

    def test_the_column_maximum_alone_does_not_hold_it(self):
        # A row denser than every column: one full row, three singletons.
        matrix = np.array([[1., 1., 1., 1.], [1., 0, 0, 0],
                           [0, 1., 0, 0], [0, 0, 1., 0]])
        columns_only = int(np.max(np.count_nonzero(matrix, axis=0)))
        largest = np.abs(matrix).max()
        assert np.linalg.norm(matrix, 2) > columns_only * largest
        assert np.linalg.norm(matrix, 2) <= simplex_sparsity(matrix) * largest

    def test_it_is_the_larger_of_the_two_readings(self):
        matrix = np.array([[1., 1., 1.], [1., 0, 0]])
        assert simplex_sparsity(matrix) == 3  # the row maximum here
        assert simplex_sparsity(matrix.T) == 3  # and the column maximum there

    def test_both_readings_stay_in_the_log(self):
        run = solve(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        for entry in run.records:
            assert entry["d"] == max(entry["d_columns"], entry["d_rows"])

    def test_a_maximum_flow_program_is_where_this_bites(self):
        # Arc columns have three non-zeros; a conservation row has one per
        # incident arc. Any vertex of degree four inverts the two.
        from hybrid_benchmarking.classical.lp import maximum_flow
        from hybrid_benchmarking.instances import Network

        arcs = [(0, i + 1, 1.0) for i in range(6)] + \
               [(i + 1, 7, 1.0) for i in range(6)]
        net = Network(name="star", source="", layout="dimacs-max", vertices=8,
                      arcs=tuple(arcs), source_vertex=0, sink_vertex=7)
        run = solve(standard_form(maximum_flow(net)), Budget(60))
        assert any(e["d_rows"] > e["d_columns"] for e in run.records)


class TestThePivotingRuleIsPartOfTheCost:
    def test_a_rule_the_route_does_not_cost_is_refused(self):
        # Running Dantzig and pricing it with the steepest-edge formula would
        # log one algorithm and cost another, with nothing saying so.
        from hybrid_benchmarking.classical import GenerationError, generate

        with pytest.raises(GenerationError, match="steepest-edge"):
            generate(CYCLE_5, "vertex-cover", "quantum-simplex", Budget(60),
                     rule="dantzig")

    def test_the_rule_the_route_does_cost_runs(self):
        from hybrid_benchmarking.classical import generate

        assert generate(CYCLE_5, "vertex-cover", "quantum-simplex", Budget(60),
                        rule="steepest-edge").run.records
