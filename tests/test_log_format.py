"""The log format as a contract, checked against every route at once.

Three properties, each of which was violated somewhere before these tests
existed, and none of which fails loudly when it breaks.

**A template must be a file this library can read back.** Handing someone a
blank log is the whole answer to "what format?", and that answer is worthless if
filling it in produces something :func:`~hybrid_benchmarking.dataset.load`
refuses -- or worse, accepts wrongly. A route whose record holds a list of layer
sizes cannot be written as a CSV column: the row reads back as the string ``[1``
and the vertex count silently becomes ``3``.

**A route must ask for exactly what it uses.** Asking for less means a log that
passes the check and then fails in the middle of an evaluation. Asking for more
means refusing a log that is complete for the route it was written for -- which
is what happened to the two solvers that never touch the largest matrix entry,
and to every interior point log, which was made to carry a vertex count that
nothing downstream can read.

**A field gathered from every record must be found or reported.** Reading a CSV
hoists a column that never varies into the instance, so the check cannot tell a
per-record field from an instance-wide one, and a genuinely missing one used to
surface as a bare ``KeyError`` from three frames down.
"""

from __future__ import annotations

import json
import warnings

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.cost import ValidityWarning
from hybrid_benchmarking.dataset import (
    FormatError,
    check,
    load,
    natural_format,
    parameters_for,
    template,
)

ROUTES = [(problem.key, route.key, route)
          for problem in hb.PROBLEMS for route in problem.routes]
IDS = ["{}/{}".format(problem, route) for problem, route, _ in ROUTES]

CHOSEN = {"epsilon": 1e-3, "delta": 1e-3}


class TestATemplateIsAFileThisLibraryCanRead:
    @pytest.mark.parametrize("problem,key,route", ROUTES, ids=IDS)
    def test_the_template_round_trips_through_load(self, problem, key, route,
                                                   tmp_path):
        shape = natural_format(route)
        path = tmp_path / ("blank." + shape)
        path.write_text(template(route))
        data = load(str(path))
        assert not check(route, data, CHOSEN)

    @pytest.mark.parametrize("problem,key,route", ROUTES, ids=IDS)
    def test_the_json_template_round_trips_too(self, problem, key, route,
                                               tmp_path):
        # It carries the same explanatory comments as the CSV one, which the
        # JSON reader used to choke on.
        path = tmp_path / "blank.json"
        path.write_text(template(route, "json"))
        assert not check(route, load(str(path)), CHOSEN)

    def test_a_route_with_a_list_in_its_record_is_not_offered_csv(self):
        route = hb.get_route("maximum-flow", "quantum-bfs")
        assert natural_format(route) == "json"
        assert template(route).lstrip().splitlines()[-1].strip() != ""
        assert '"layers"' in template(route)

    def test_a_route_of_plain_numbers_is_offered_csv(self):
        assert natural_format(hb.get_route("vertex-cover",
                                           "quantum-simplex")) == "csv"

    def test_what_a_csv_would_have_done_to_a_list(self, tmp_path):
        # The failure this prevents, spelled out: a comma-separated row cannot
        # hold a comma-separated value, and nothing downstream notices.
        path = tmp_path / "wrong.csv"
        path.write_text("layers,vertices\n[1, 3, 5, 2],1000\n")
        data = load(str(path))
        assert data.instance.get("vertices") == 3  # not 1000
        with pytest.raises(FormatError):
            hb.run(hb.get_route("maximum-flow", "quantum-bfs"), data)


class TestARouteAsksForExactlyWhatItUses:
    @pytest.mark.parametrize("problem,key,route", ROUTES, ids=IDS)
    def test_everything_the_cost_needs_is_something_the_route_asks_for(
            self, problem, key, route):
        supplied = {f.name for f in route.per_record + route.per_instance
                    + route.chosen}
        if route.shape is not None:
            supplied |= set(route.shape({"vertices": 10.0, "edges": 20.0}))
        supplied |= set(route.renames.values())
        if route.collects:
            supplied.add(route.collects[1])

        cost = hb.compose({"routine": route.target, "unit": route.unit.name})
        assert set(cost.parameters) <= supplied, (
            "{}/{} costs need {} which nothing supplies".format(
                problem, key, set(cost.parameters) - supplied))

    @pytest.mark.parametrize("problem,key,route", ROUTES, ids=IDS)
    def test_everything_the_route_asks_for_is_something_it_uses(
            self, problem, key, route):
        cost = hb.compose({"routine": route.target, "unit": route.unit.name})
        wanted = set(cost.parameters)
        if route.shape is not None:
            wanted |= {"vertices", "edges"}  # the shape's own inputs
        wanted |= set(route.renames)
        if route.collects:
            wanted.add(route.collects[0])

        asked = {f.name for f in route.per_record + route.per_instance}
        assert asked <= wanted, (
            "{}/{} asks for {} and never reads it".format(
                problem, key, asked - wanted))

    def test_the_two_solvers_that_never_simulate_do_not_ask_for_the_largest_entry(
            self):
        # Chebyshev reaches the matrix by a quantum walk and the singular value
        # transformation by a block encoding; neither ever sees ||A||_max.
        without = hb.Dataset(records=({"kappa": 10.0, "d": 4, "x_norm": 1.0},))
        for key in ("qsvt", "chebyshev"):
            route = hb.get_route("linear-systems", key)
            assert not check(route, without, {"epsilon": 1e-3})
        for key in ("fourier", "hhl"):
            route = hb.get_route("linear-systems", key)
            assert check(route, without, {"epsilon": 1e-3})

    def test_the_interior_point_route_does_not_ask_for_a_vertex_count(self):
        # A Newton system states its own dimension. The program's column and row
        # counts never reach the cost, so requiring them refused good logs.
        route = hb.get_route("vertex-cover", "quantum-interior-point")
        data = hb.Dataset(records=({"N": 100, "d": 50, "kappa": 30.0},))
        assert not check(route, data, {"epsilon": 1e-3})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ValidityWarning)
            assert hb.run(route, data, {"epsilon": 1e-3})["total"] > 0


class TestGatheringAFieldFromEveryRecord:
    def test_a_missing_one_is_named_rather_than_raised_from_inside(self):
        route = hb.get_route("maximum-flow", "quantum-bfs")
        data = hb.Dataset(records=({"layers": [1, 2]}, {"nothing": 0}),
                          instance={"vertices": 100})
        with pytest.raises(FormatError, match="record 2"):
            hb.run(route, data)

    def test_one_hoisted_to_the_instance_by_csv_folding_still_works(self):
        # A CSV column that never varies is read as a property of the instance.
        # A sweep whose layers happen to be identical every time is still a
        # sweep with those layers.
        route = hb.get_route("maximum-flow", "quantum-bfs")
        data = hb.Dataset(records=({}, {}),
                          instance={"vertices": 100, "layers": [1, 3, 5]})
        assert hb.run(route, data)["total"] > 0

    def test_the_check_alone_cannot_catch_it_which_is_why_run_does(self):
        # Documenting the reason the guard lives where it does: after hoisting,
        # `check` genuinely cannot tell the two cases apart.
        route = hb.get_route("maximum-flow", "quantum-bfs")
        data = hb.Dataset(records=({"nothing": 0},),
                          instance={"vertices": 100, "layers": [1, 3]})
        assert not check(route, data)
