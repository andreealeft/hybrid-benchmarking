"""Maximum flow attacked as a linear program, by both of the other two routes.

The problem now has three routes that all start from the same DIMACS file, and
they are the strongest cross-check in the library: Dinic walks augmenting paths,
the simplex walks the boundary of a polytope, the interior point method avoids
that boundary, and all three have to report the same maximum flow.  They share a
file reader and nothing else.

The modelling claim underneath is the one that took a correction.
:func:`~hybrid_benchmarking.problems.flow_shape` predicted ``2E`` columns, and a
program with one variable per arc and conservation at every vertex cannot
express a positive flow at all -- the conservation rows sum to zero, so the net
flow out of the source is forced to zero and the optimum is zero on every
network.  The missing column is the flow value itself.  Both natural repairs, a
flow variable or an uncapacitated return arc, give ``2E + 1``, so it is a
correction rather than a choice; the test that the predicted shape and the built
program agree is what keeps it one.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.dinic import solve as dinic
from hybrid_benchmarking.classical.ipm import solve as interior_point
from hybrid_benchmarking.classical.lp import (
    ModelError,
    maximum_flow,
    standard_form,
)
from hybrid_benchmarking.classical.simplex import solve as pivot
from hybrid_benchmarking.instances import Network
from hybrid_benchmarking.instances.dimacs import read_max_flow
from hybrid_benchmarking.problems import flow_shape


def network(vertices: int, arcs, source=0, sink=None) -> Network:
    return Network(name="n", source="(hand built)", layout="dimacs-max",
                   vertices=vertices, arcs=tuple(arcs), source_vertex=source,
                   sink_vertex=vertices - 1 if sink is None else sink)


DIAMOND = network(4, [(0, 1, 3.0), (0, 2, 2.0), (1, 2, 1.0),
                      (1, 3, 2.0), (2, 3, 3.0)])
CHAIN = network(4, [(0, 1, 7.0), (1, 2, 2.0), (2, 3, 9.0)])
PARALLEL = network(3, [(0, 1, 4.0), (0, 1, 5.0), (1, 2, 6.0)])
WITH_ZERO = network(3, [(0, 1, 0.0), (0, 1, 3.0), (1, 2, 8.0)])

NETWORKS = [("diamond", DIAMOND), ("chain", CHAIN),
            ("parallel arcs", PARALLEL), ("a zero-capacity arc", WITH_ZERO)]


def minimum_cut(instance: Network) -> float:
    """The smallest capacity crossing any source-side set, by enumeration.

    Exponential and deliberately so: a statement of what maximum flow means,
    written without reference to how any of the three solvers computes it.
    """
    others = [v for v in range(instance.vertices)
              if v not in (instance.source_vertex, instance.sink_vertex)]
    best = float("inf")
    for size in range(len(others) + 1):
        for chosen in combinations(others, size):
            side = {instance.source_vertex} | set(chosen)
            best = min(best, sum(capacity for tail, head, capacity in instance.arcs
                                 if tail in side and head not in side))
    return best


class TestThreeSolversOnOneFile:
    @pytest.mark.parametrize("label,instance", NETWORKS,
                             ids=[label for label, _ in NETWORKS])
    def test_they_all_agree_with_the_minimum_cut(self, label, instance):
        form = standard_form(maximum_flow(instance))
        expected = minimum_cut(instance)

        assert dinic(instance, Budget(60)).result["maximum_flow"] == \
            pytest.approx(expected)
        assert pivot(form, Budget(60)).result["objective"] == \
            pytest.approx(expected, abs=1e-6)
        assert interior_point(form, Budget(60)).result["objective"] == \
            pytest.approx(expected, abs=1e-5)

    def test_all_three_routes_run_from_the_same_file_on_disk(self):
        instance = read_max_flow("tests/fixtures/tiny.max")
        for route in ("quantum-bfs", "quantum-simplex",
                      "quantum-interior-point"):
            generated = generate(instance, "maximum-flow", route, Budget(60))
            assert generated.status is Status.COMPLETE, route

    def test_the_three_routes_are_not_in_the_same_unit(self):
        # Which is what stops them being tabulated as a comparison. The quantum
        # search route counts gates, the interior point one cycles.
        instance = read_max_flow("tests/fixtures/tiny.max")
        units = {
            route: cost(generate(instance, "maximum-flow", route, Budget(60)),
                        {"epsilon": 1e-3, "delta": 1e-3})["unit"]
            for route in ("quantum-bfs", "quantum-simplex",
                          "quantum-interior-point")
        }
        assert units["quantum-bfs"] == "GATES"
        assert units["quantum-interior-point"] == "CYCLES"


class TestTheShapeThatWasCorrected:
    @pytest.mark.parametrize("label,instance", NETWORKS,
                             ids=[label for label, _ in NETWORKS])
    def test_the_built_program_matches_what_the_route_predicts(
            self, label, instance):
        built = standard_form(maximum_flow(instance)).shape
        predicted = flow_shape({"vertices": instance.vertices,
                                "edges": len(instance.arcs)})
        assert built == {name: int(value) for name, value in predicted.items()}

    def test_a_zero_capacity_arc_does_not_shrink_the_program(self):
        # Capacities are rows rather than column bounds precisely so that this
        # holds: a zero-capacity column has equal bounds, and the converter
        # quite correctly folds away a column that cannot move.
        with_zero = standard_form(maximum_flow(WITH_ZERO)).shape
        without = standard_form(maximum_flow(
            network(3, [(0, 1, 1.0), (0, 1, 3.0), (1, 2, 8.0)]))).shape
        assert with_zero == without

    def test_dropping_the_flow_column_would_force_the_optimum_to_zero(self):
        # The reason the count is 2E + 1. Conservation over the arc variables
        # alone has rows summing to zero, so no positive flow satisfies them.
        conservation = np.zeros((DIAMOND.vertices, len(DIAMOND.arcs)))
        for column, (tail, head, _) in enumerate(DIAMOND.arcs):
            conservation[tail, column] += 1.0
            conservation[head, column] -= 1.0
        assert np.allclose(conservation.sum(axis=0), 0.0)
        assert minimum_cut(DIAMOND) > 0  # the real answer is not zero

    def test_the_source_and_sink_cannot_be_the_same_vertex(self):
        # The flow column would enter that row twice with opposite signs and
        # cancel, leaving the objective unbounded.
        with pytest.raises(ModelError, match="unbounded"):
            maximum_flow(network(3, [(0, 1, 1.0)], source=1, sink=1))


class TestCostingIt:
    def test_the_simplex_route_costs_a_network_in_gates(self):
        instance = read_max_flow("tests/fixtures/tiny.max")
        report = cost(generate(instance, "maximum-flow", "quantum-simplex",
                               Budget(60)), {"epsilon": 1e-3, "delta": 1e-3})
        assert report["total"] > 0
        assert report["unit_label"] == "gates"
        assert report["derivation"] == "logged from a classical run"

    def test_the_log_carries_the_networks_own_size(self):
        instance = read_max_flow("tests/fixtures/tiny.max")
        generated = generate(instance, "maximum-flow", "quantum-simplex",
                             Budget(60))
        assert generated.data.instance["vertices"] == instance.vertices
        assert generated.data.instance["edges"] == len(instance.arcs)

    def test_the_interior_point_route_says_it_dropped_a_redundant_row(self):
        instance = read_max_flow("tests/fixtures/tiny.max")
        report = cost(generate(instance, "maximum-flow",
                               "quantum-interior-point", Budget(60)),
                      {"epsilon": 1e-3})
        assert any("redundant constraint row" in note
                   for note in report["assumptions"])
