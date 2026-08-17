"""Problems, routes, and the log format.

The entry point for someone who has a network rather than a subroutine. What
matters here is that the modelling step is right -- problem shape becoming
program shape -- and that a log which does not carry what a route needs is
refused by name rather than quietly producing a number.
"""

from __future__ import annotations

import json

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.dataset import (
    Dataset,
    FormatError,
    check,
    load,
    parameters_for,
    required,
    run,
    template,
)
from hybrid_benchmarking.problems import (
    PROBLEMS,
    clique_shape,
    cover_shape,
    flow_shape,
    get_problem,
    get_route,
)
from hybrid_benchmarking.provenance import Unit

SIMPLEX_LOG = (
    "kappa,d,A_1,A_max,t,c_max,u_norm,vertices,edges\n"
    "3.1,4,3.0,1.0,12,1.0,1.4,300,900\n"
    "5.7,4,3.2,1.0,9,1.0,1.6,300,900\n"
)
CHOSEN = {"epsilon": 1e-3, "delta": 1e-3}


class TestTheCatalogue:
    def test_every_problem_has_a_friendly_name_and_a_technical_one(self):
        for problem in PROBLEMS:
            assert problem.label and problem.technical and problem.blurb
            assert problem.label != problem.technical

    def test_every_route_names_its_classical_algorithm_and_what_it_replaces(self):
        for problem in PROBLEMS:
            for route in problem.routes:
                assert route.classical and route.replaces
                assert "replaces" in route.describe()

    def test_every_route_points_at_something_registered(self):
        for problem in PROBLEMS:
            for route in problem.routes:
                assert route.unit in hb.get(route.target).units

    def test_maximum_flow_is_the_one_with_the_most_routes(self):
        # Quantum search inside Dinic, the simplex, and the two interior point
        # system formulations the interior point paper compares.
        assert len(get_problem("maximum-flow").routes) == 4
        assert {r.key for r in get_problem("maximum-flow").routes} == {
            "quantum-bfs", "quantum-simplex", "quantum-interior-point",
            "quantum-interior-point-oss",
        }

    def test_an_unknown_problem_lists_what_there_is(self):
        with pytest.raises(KeyError, match="knapsack"):
            get_problem("travelling-salesman")


class TestProblemShapeBecomesProgramShape:
    """The simplex works on the standard form, not on the formulation as
    written -- and getting that wrong produces a program with more rows than
    columns, which is not a program."""

    def test_a_cover_program_has_more_columns_than_rows(self):
        shape = cover_shape({"vertices": 1000, "edges": 5000})
        assert shape["n"] > shape["m"]

    def test_slacks_are_counted(self):
        """One surplus per edge constraint, one slack per upper bound."""
        assert cover_shape({"vertices": 10, "edges": 20}) == {"n": 40, "m": 30}

    def test_clique_constraints_count_absent_edges(self):
        """Sparse graphs give dense programs, which is the awkward case."""
        sparse = clique_shape({"vertices": 100, "edges": 100})
        dense = clique_shape({"vertices": 100, "edges": 4000})
        assert sparse["m"] > dense["m"]

    def test_flow_has_a_variable_per_edge_and_one_more_for_the_flow_itself(self):
        assert flow_shape({"vertices": 300, "edges": 900}) == {
            "n": 1801, "m": 1200
        }

    def test_without_that_extra_column_the_program_can_only_circulate(self):
        """Why the count is 2E + 1 and not 2E.

        Conservation at every vertex, over the edge variables alone, forces the
        net flow out of the source to zero: the rows sum to zero because every
        arc leaves one vertex and enters another. Such a program has maximum
        flow zero on every network, so it is not a maximum-flow program at all.
        The missing column is the flow value.
        """
        import numpy as np

        arcs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        vertices = 4
        conservation = np.zeros((vertices, len(arcs)))
        for column, (tail, head) in enumerate(arcs):
            conservation[tail, column] += 1.0
            conservation[head, column] -= 1.0
        assert np.allclose(conservation.sum(axis=0), 0.0)
        assert np.linalg.matrix_rank(conservation) < vertices

    def test_a_naive_shape_would_be_refused(self):
        """Guards the modelling fix: one variable per vertex and one constraint
        per edge asks for a basis larger than the column count."""
        data = Dataset(({"kappa": 3.0, "d": 4, "A_1": 3.0, "A_max": 1.0,
                         "t": 5, "c_max": 1.0, "u_norm": 1.4},),
                       {"n": 1000, "m": 5000})
        route = get_route("linear-programming", "quantum-simplex")
        with pytest.raises(ValueError, match="no meaning"):
            run(route, data, CHOSEN)


class TestTheFormat:
    def test_a_template_names_every_column_and_explains_it(self):
        text = template(get_route("vertex-cover", "quantum-simplex"))
        for name in ("kappa", "d", "A_1", "u_norm", "vertices", "edges"):
            assert name in text
        assert "not logged: epsilon, delta" in text

    def test_a_json_template_is_valid_json(self):
        text = template(get_route("knapsack", "tree-generator"), "json")
        body = text[text.index("{"):]
        assert json.loads(body)["instance"]["capacity"]

    def test_csv_columns_that_never_vary_describe_the_instance(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text(SIMPLEX_LOG)
        data = load(str(path))
        assert len(data) == 2
        assert data.instance["vertices"] == 300
        assert "kappa" not in data.instance

    def test_json_carries_records_that_are_not_numbers(self, tmp_path):
        path = tmp_path / "log.json"
        path.write_text(json.dumps({
            "instance": {"vertices": 300},
            "records": [{"layers": [1, 3, 5, 2]}, {"layers": [1, 2, 4]}],
        }))
        data = load(str(path))
        assert data.records[0]["layers"] == [1, 3, 5, 2]

    def test_comments_in_a_log_are_ignored(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text("# logged by an instrumented solver\n" + SIMPLEX_LOG)
        assert len(load(str(path))) == 2

    def test_a_missing_file_says_so(self):
        with pytest.raises(FormatError, match="no such file"):
            load("/nowhere/at/all.csv")

    def test_an_empty_log_says_so(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("\n")
        with pytest.raises(FormatError, match="no rows"):
            load(str(path))


class TestWhatIsMissingIsNamed:
    def test_missing_columns_are_listed_with_their_meaning(self):
        route = get_route("vertex-cover", "quantum-simplex")
        missing = check(route, Dataset(({"kappa": 3.0},), {}), {})
        assert any(m.startswith("u_norm") for m in missing)
        assert any("pivot direction of the entering column" in m
                   for m in missing)

    def test_a_route_that_costs_iterations_needs_at_least_one(self):
        route = get_route("vertex-cover", "quantum-simplex")
        missing = check(route, Dataset((), {"vertices": 10, "edges": 20}),
                        CHOSEN)
        assert any("at least one record" in m for m in missing)

    def test_running_an_incomplete_log_refuses_rather_than_guessing(self):
        route = get_route("vertex-cover", "quantum-simplex")
        with pytest.raises(FormatError, match="missing"):
            run(route, Dataset(({"kappa": 3.0},), {}), CHOSEN)


class TestCosting:
    def test_a_run_is_the_sum_of_its_iterations(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text(SIMPLEX_LOG)
        result = run(get_route("vertex-cover", "quantum-simplex"),
                     load(str(path)), CHOSEN)
        assert result["records"] == 2
        assert result["total"] == pytest.approx(sum(result["per_record"]))
        assert result["largest"] <= result["total"]

    def test_some_routes_take_the_whole_run_at_once(self, tmp_path):
        """Dinic's cost is stated over every sweep together, not per sweep."""
        path = tmp_path / "flow.json"
        path.write_text(json.dumps({
            "instance": {"vertices": 300},
            "records": [{"layers": [1, 3, 5, 2]}, {"layers": [1, 2, 4]}],
        }))
        result = run(get_route("maximum-flow", "quantum-bfs"), load(str(path)))
        assert result["whole_run"] is True
        assert result["total"] > 0

    def test_a_friendly_field_name_reaches_the_lemma_parameter(self):
        """Someone logs "vertices"; the lemma calls it a list length."""
        route = get_route("maximum-flow", "quantum-bfs")
        values = parameters_for(route, {}, Dataset((), {"vertices": 300}), {})
        assert values["X"] == 300

    def test_every_result_carries_its_bound_and_its_assumptions(self, tmp_path):
        path = tmp_path / "log.csv"
        path.write_text(SIMPLEX_LOG)
        result = run(get_route("vertex-cover", "quantum-simplex"),
                     load(str(path)), CHOSEN)
        assert result["bound"] == "lower bound"
        assert result["assumptions"]
        assert "lower bound" in result["note"]


class TestRoutesAreNotComparable:
    """Three routes through one problem, three different cost models. The
    interface must say so rather than tabulate the numbers side by side."""

    def test_the_three_flow_routes_rest_on_different_models(self):
        routes = get_problem("maximum-flow").routes
        assert len({(r.unit, r.note) for r in routes}) == 3

    def test_two_of_them_report_gates_and_still_do_not_compare(self):
        """Sharing a unit is not sharing a model: one is a scheduled circuit
        converted at one gate per cycle, the other an analytic bound with
        non-leading terms dropped."""
        by_key = {r.key: r for r in get_problem("maximum-flow").routes}
        assert by_key["quantum-bfs"].unit is by_key["quantum-simplex"].unit
        assert by_key["quantum-bfs"].note != by_key["quantum-simplex"].note

    def test_each_carries_a_note_explaining_its_model(self):
        for route in get_problem("maximum-flow").routes:
            assert len(route.note) > 40
