"""Counts on a clock, and the clock the counts would need.

The projection is the number somebody will quote out of context, so most of
what is asserted here is about what travels with it: which units may be timed
at all, which direction the optimism runs, and that the required rate -- the
one the simplex study reports, and the one that does not go stale -- is derived
from a measurement rather than from an assumption.
"""

from __future__ import annotations

import pytest

from hybrid_benchmarking.classical import Budget, compare
from hybrid_benchmarking.classical.generate import timing
from hybrid_benchmarking.provenance import Unit
from hybrid_benchmarking.runtime import (
    CONTROL_FLOOR_SECONDS,
    RECORD_SECONDS,
    NoClock,
    humanise,
    per_iteration,
    project,
    timeable,
)


class TestWhichUnitsHaveADuration:
    def test_gates_and_cycles_do(self):
        assert timeable(Unit.GATES) and timeable(Unit.CYCLES)

    @pytest.mark.parametrize("unit", [Unit.QUERIES, Unit.SUBROUTINE_CALLS,
                                      Unit.ITERATIONS, Unit.REPETITIONS])
    def test_nothing_else_does(self, unit):
        """A query has no length: nothing here fixes what answering one costs,
        so a duration would be invented rather than derived."""
        assert not timeable(unit)
        with pytest.raises(NoClock):
            project(1e6, unit)

    def test_the_refusal_says_what_can_be_timed(self):
        with pytest.raises(NoClock, match="Gates and cycles"):
            project(1e6, Unit.QUERIES)

    def test_an_unknown_rate_is_refused_rather_than_defaulted(self):
        with pytest.raises(NoClock, match="unknown rate"):
            project(1e6, Unit.GATES, rate="wishful")


class TestTheProjection:
    def test_it_is_the_count_times_the_rate(self):
        made = project(1e9, Unit.GATES)
        assert made.seconds == pytest.approx(1e9 * RECORD_SECONDS)

    def test_the_control_floor_is_faster_than_the_record(self):
        """Electronics bound gate speed where physics does not, and the bound
        is the more defensible of the two reference lines."""
        assert CONTROL_FLOOR_SECONDS < RECORD_SECONDS
        assert (project(1e9, Unit.GATES, rate="control").seconds
                < project(1e9, Unit.GATES, rate="record").seconds)

    def test_every_projection_says_why_it_is_optimistic(self):
        made = project(1e9, Unit.GATES)
        joined = " ".join(made.assumptions).lower()
        assert "error correction" in joined
        assert "isolated gate operation" in joined
        assert "rotation" in joined


class TestTheRequiredRate:
    def test_it_is_the_measured_time_over_the_count(self):
        made = project(1e6, Unit.GATES, classical_seconds=2.0)
        assert made.required == pytest.approx(2e-6)

    def test_the_shortfall_is_how_much_faster_than_the_record(self):
        made = project(1e6, Unit.GATES, classical_seconds=2.0)
        assert made.shortfall == pytest.approx(RECORD_SECONDS / 2e-6)

    def test_a_method_that_needs_less_than_the_record_is_reachable(self):
        """The whole point of the comparison: some routes are on the right side
        of the line and the tool has to be able to say so."""
        made = project(10, Unit.GATES, classical_seconds=1.0)
        assert made.shortfall < 1
        assert "already provides" in made.describe()

    def test_one_that_needs_more_says_by_how_much(self):
        made = project(1e18, Unit.GATES, classical_seconds=1.0)
        assert made.shortfall > 1
        assert "faster than" in made.describe()

    def test_without_a_classical_time_there_is_no_requirement(self):
        made = project(1e6, Unit.GATES)
        assert made.required is None and made.shortfall is None


class TestPerIteration:
    def test_stamps_become_the_cost_of_each_iteration(self):
        records = ({"at_seconds": 0.5}, {"at_seconds": 0.9}, {"at_seconds": 2.0})
        assert per_iteration(records) == pytest.approx((0.5, 0.4, 1.1))

    def test_a_log_without_stamps_gives_nothing_rather_than_a_guess(self):
        """Someone else's log has no timings in it, and inventing them would
        put a fabricated denominator under a published-looking number."""
        assert per_iteration(({"at_seconds": 0.5}, {"kappa": 3.0})) == ()


class TestTheSolversStampTheirRecords:
    def _run(self, problem, values):
        result = compare(problem, values, budget=Budget(60))
        return [r for r in result["routes"] if "error" not in r]

    def test_a_max_flow_comparison_times_every_route(self):
        for route in self._run("maximum-flow", {"things": "40", "links": "90"}):
            block = route["timing"]
            assert block["classical_seconds"] >= 0
            assert block["seconds"] > 0
            assert block["required"] > 0
            assert block["rate"] == RECORD_SECONDS

    def test_the_classical_time_is_the_run_that_produced_the_log(self):
        """Not a shared number: each route runs its own classical algorithm --
        a breadth-first search, a simplex, an interior point step -- and is
        timed against that one."""
        blocks = [r["timing"]["classical_seconds"]
                  for r in self._run("maximum-flow",
                                     {"things": "40", "links": "90"})]
        assert len(set(blocks)) > 1


class TestTheTimingBlock:
    def test_a_unit_with_no_clock_comes_back_explained_not_missing(self):
        block = timing(1e6, "QUERIES", 1.0)
        assert "why_not" in block and "seconds" not in block
        assert block["classical"]

    def test_it_answers_to_either_spelling_of_a_unit(self):
        assert timing(1e6, "GATES", 1.0)["seconds"] > 0
        assert timing(1e6, "gates", 1.0)["seconds"] > 0


class TestHumanise:
    @pytest.mark.parametrize("seconds,expected", [
        (2.34e-16, "234 as"), (6.5e-9, "6.5 ns"), (0.31, "310 ms"),
        (3.6e5, "4.17 days"),
    ])
    def test_it_picks_a_unit_somebody_can_picture(self, seconds, expected):
        assert humanise(seconds) == expected

    def test_the_very_large_stops_pretending_to_be_a_duration(self):
        assert "age of the universe" in humanise(1e25)


class TestWhatTheQuantumRoutineActuallyReplaces:
    """The comparison is between algorithms, not between a count and a clock.

    Dinic's quantum version replaces the layering sweeps and leaves the
    blocking-flow phases classical.  Costing the sweeps and setting that
    against the whole solve would credit the quantum side with work it never
    does -- and on a sixty-node network the sweeps are around a tenth of the
    solve, which is the difference between clearing the speed record and
    missing it.
    """

    def _flow(self):
        result = compare("maximum-flow", {"things": "60", "links": "180"},
                         budget=Budget(60))
        return {r["route"]: r for r in result["routes"] if "error" not in r}

    def test_the_layering_sweeps_are_only_part_of_the_solve(self):
        block = self._flow()["quantum-bfs"]["timing"]
        assert block["partial"] is True
        assert 0 < block["replaced_seconds"] < block["classical_seconds"]
        assert block["retained_seconds"] > 0

    def test_the_quantum_bar_carries_the_part_that_stays_classical(self):
        block = self._flow()["quantum-bfs"]["timing"]
        assert block["hybrid_seconds"] == pytest.approx(
            block["seconds"] + block["retained_seconds"])

    def test_the_required_rate_divides_by_the_replaced_part_only(self):
        route = self._flow()["quantum-bfs"]
        block = route["timing"]
        assert block["required"] == pytest.approx(
            block["replaced_seconds"] / route["total"])
        # ... which is a stiffer requirement than dividing by the whole solve,
        # and stiffer is the point: the easier number is not the true one.
        assert block["required"] < block["classical_seconds"] / route["total"]

    def test_a_route_that_replaces_the_whole_solve_says_so(self):
        block = self._flow()["quantum-simplex"]["timing"]
        assert block["partial"] is False
        assert block["retained_seconds"] == 0
        assert block["hybrid_seconds"] == pytest.approx(block["seconds"])
        assert "note" not in block

    def test_the_split_is_recorded_in_the_log_the_user_is_shown(self):
        """Not only in the comparison: the log file states it too."""
        from hybrid_benchmarking.classical import generate_from_parameters

        made = generate_from_parameters("maximum-flow",
                                        {"things": "40", "links": "90"},
                                        "quantum-bfs", Budget(60))
        stated = made.run.stated()
        assert stated["replaced_seconds"] > 0
        assert all("sweep_seconds" in record for record in made.run.records)
