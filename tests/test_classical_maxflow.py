"""Generating a maximum-flow log by running Dinic, and costing what came out.

The first problem taken end to end: a network in, a log of layer sizes out, a
gate count from the log.  What is worth asserting here is in two halves.

The classical half has a right answer independent of this library -- the maximum
flow equals the minimum cut, and for a small enough network every cut can be
enumerated -- so the instrumented solver is checked against the theorem rather
than against itself.

The logging half is about what survives.  A generated log has to say who
generated it and whether that run finished, it has to still say so after a trip
through a file, and a run that was cut off has to produce a number that is
marked as a lower bound for that reason as well as the lemmas'.
"""

from __future__ import annotations

from itertools import combinations

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.dinic import layer_sizes, solve
from hybrid_benchmarking.dataset import load, render, write
from hybrid_benchmarking.instances import Network
from hybrid_benchmarking.provenance import Bound, Derivation

# Four vertices, source 0, sink 3.  By hand: the cut {0} has capacity 5, and
# 0->1->3 plus 0->2->3 carries 2 + 2 = 4, with one more unit going 0->1->2->3.
TOY = Network(
    name="toy", source="(hand built)", layout="dimacs-max", vertices=4,
    arcs=((0, 1, 3.0), (0, 2, 2.0), (1, 2, 1.0), (1, 3, 2.0), (2, 3, 3.0)),
    source_vertex=0, sink_vertex=3,
)


def minimum_cut(network: Network) -> float:
    """The smallest capacity crossing any source-side set, by enumeration.

    Exponential and deliberately so: it is a statement of what maximum flow
    means, written without reference to how Dinic computes it.
    """
    others = [v for v in range(network.vertices)
              if v not in (network.source_vertex, network.sink_vertex)]
    best = float("inf")
    for size in range(len(others) + 1):
        for chosen in combinations(others, size):
            side = {network.source_vertex} | set(chosen)
            crossing = sum(
                capacity for tail, head, capacity in network.arcs
                if tail in side and head not in side
            )
            best = min(best, crossing)
    return best


class TestTheClassicalRun:
    def test_the_flow_it_finds_equals_the_minimum_cut(self):
        run = solve(TOY, Budget(60))
        assert run.status is Status.COMPLETE
        assert run.result["maximum_flow"] == minimum_cut(TOY)

    @pytest.mark.parametrize("seed_arcs", [
        ((0, 1, 4.0), (0, 2, 2.0), (1, 2, 3.0), (1, 3, 1.0), (2, 3, 6.0)),
        ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 7.0)),
        ((0, 1, 5.0), (0, 2, 5.0), (1, 3, 5.0), (2, 3, 5.0), (1, 2, 100.0)),
    ])
    def test_max_flow_min_cut_holds_on_every_network_we_try(self, seed_arcs):
        network = Network(name="n", source="(hand built)", layout="dimacs-max",
                          vertices=4, arcs=seed_arcs, source_vertex=0,
                          sink_vertex=3)
        run = solve(network, Budget(60))
        assert run.result["maximum_flow"] == minimum_cut(network)

    def test_a_network_with_no_path_to_the_sink_costs_one_failed_sweep(self):
        network = Network(name="cut", source="(hand built)",
                          layout="dimacs-max", vertices=3,
                          arcs=((0, 1, 1.0),), source_vertex=0, sink_vertex=2)
        run = solve(network, Budget(60))
        assert run.result["maximum_flow"] == 0.0
        assert len(run.records) == 1  # the sweep that discovers there is nothing

    def test_source_and_sink_cannot_be_the_same_vertex(self):
        network = Network(name="silly", source="(hand built)",
                          layout="dimacs-max", vertices=2,
                          arcs=((0, 1, 1.0),), source_vertex=1, sink_vertex=1)
        assert solve(network, Budget(60)).status is Status.FAILED


class TestWhatGetsLogged:
    def test_a_layer_list_starts_at_the_source_and_has_no_empty_layers(self):
        run = solve(TOY, Budget(60))
        for record in run.records:
            layers = record["layers"]
            assert layers[0] == 1  # the source is its own layer
            assert all(size > 0 for size in layers)

    def test_the_layers_of_a_sweep_account_for_every_reachable_vertex(self):
        assert layer_sizes([0, 1, 1, 2]) == [1, 2, 1]
        assert layer_sizes([0, -1, 1, -1]) == [1, 1]  # unreached ones are absent

    def test_the_final_failing_sweep_is_recorded_like_any_other(self):
        run = solve(TOY, Budget(60))
        # Dinic stops because a sweep failed to reach the sink; that sweep
        # happened, so the last record is a layer list that does not span the
        # network.
        assert sum(run.records[-1]["layers"]) < TOY.vertices

    def test_the_log_names_the_implementation_that_produced_it(self):
        generated = generate(TOY, budget=Budget(60))
        assert "Dinic" in generated.data.generated["implementation"]
        assert "published" in generated.data.generated["implementation"]


class TestCostingAGeneratedLog:
    def test_the_cost_is_logged_rather_than_analytic(self):
        report = cost(generate(TOY, budget=Budget(60)))
        assert report["derivation"] == str(Derivation.LOGGED)
        assert report["bound"] == str(Bound.LOWER)
        assert report["unit"] == "GATES"

    def test_the_provenance_names_both_the_lemma_and_our_solver(self):
        report = cost(generate(TOY, budget=Budget(60)))
        assert "Lemmas 1 and 2" in report["provenance"]
        assert "Dinic" in report["provenance"]

    def test_a_generated_log_costs_the_same_after_a_round_trip_through_a_file(
            self, tmp_path):
        generated = generate(TOY, budget=Budget(60))
        direct = cost(generated)

        path = generated.save(str(tmp_path / "toy.json"))
        reloaded = load(path)
        from_file = hb.run(generated.route, reloaded)

        assert from_file["total"] == pytest.approx(direct["total"])
        assert from_file["derivation"] == direct["derivation"]
        assert from_file["provenance"] == direct["provenance"]

    def test_a_hand_written_log_is_untouched_by_any_of_this(self):
        # No `generated` block, so nothing is added: the numbers stay analytic.
        by_hand = hb.Dataset(
            records=({"layers": [1, 2, 1]},), instance={"vertices": 4},
        )
        report = hb.run(hb.get_route("maximum-flow", "quantum-bfs"), by_hand)
        assert report["derivation"] == str(Derivation.ANALYTIC)
        assert report["status"] == ""


class TestARunThatWasCutOff:
    def test_an_exhausted_budget_truncates_rather_than_failing(self):
        run = solve(TOY, Budget(1e-9))
        assert run.status is Status.TRUNCATED
        assert run.records  # what did happen is kept

    def test_a_truncated_total_is_a_lower_bound_for_its_own_reason(self):
        generated = generate(TOY, budget=Budget(1e-9))
        report = cost(generated)
        assert report["status"] == "truncated"
        assert report["bound"] == str(Bound.LOWER)
        assert any("cut off" in note and "unrelated to the lemmas" in note
                   for note in report["assumptions"])

    def test_a_truncated_total_is_below_the_complete_one(self):
        partial = cost(generate(TOY, budget=Budget(1e-9)))["total"]
        whole = cost(generate(TOY, budget=Budget(60)))["total"]
        assert 0 < partial < whole

    def test_the_advice_offers_a_longer_budget_before_it_offers_a_person(self):
        advice = generate(TOY, budget=Budget(1e-9)).run.advice()
        assert advice.index("--budget") < advice.index("telling us")


class TestTheFileTheUserSees:
    def test_a_log_with_a_list_in_it_is_written_as_json(self):
        generated = generate(TOY, budget=Budget(60))
        text = render(generated.data, generated.route)
        assert text.lstrip().startswith("{")
        assert '"layers"' in text

    def test_a_flat_log_is_written_as_csv_with_its_provenance_in_the_header(
            self, tmp_path):
        data = hb.Dataset(
            records=({"kappa": 3.0, "t": 2}, {"kappa": 5.0, "t": 1}),
            instance={"n": 200, "m": 50},
            generated={"status": "complete", "implementation": "a test"},
        )
        path = write(data, str(tmp_path / "flat.csv"))
        text = open(path).read()
        # A sentence first, for whoever opens the file; the machine-readable
        # twin of it further down, for whoever loads it.
        assert text.splitlines()[0] == "# a test -- complete, 0 records in 0s"
        assert "# generated: {" in text

        back = load(path)
        assert back.generated["implementation"] == "a test"
        assert back.instance["n"] == 200
        assert [record["kappa"] for record in back.records] == [3.0, 5.0]
