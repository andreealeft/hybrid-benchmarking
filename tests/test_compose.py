"""Assembling algorithms out of routines.

The published analyses are compositions, and nothing about them is privileged:
the same three moves -- fill a slot, add, multiply -- build any of them. What
matters is that an assembled cost stays honest, carrying the union of the
parameters and the weaker of the bound directions without anyone remembering to
propagate it.
"""

from __future__ import annotations

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.compose import build, code, describe, entries, fillings_for
from hybrid_benchmarking.cost import UnitMismatch
from hybrid_benchmarking.provenance import Bound, Unit

MARKER = {"routine": "CanEnterNFP", "unit": "GATES"}
ESTIMATE = {"routine": "QAE", "unit": "GATES", "bind": {"oracle_gates": MARKER}}


class TestSlots:
    def test_a_wrapper_declares_what_it_takes(self):
        slots = hb.get("QAE").cost(Unit.GATES).slots
        assert slots == {"oracle_gates": Unit.GATES,
                         "prepare_gates": Unit.GATES}

    def test_left_empty_the_slot_stays_a_parameter(self):
        """Which is the useful default: it reports how many times it calls
        something, without needing to know what."""
        cost = hb.get("QAE").cost(Unit.GATES)
        assert "oracle_gates" in cost.parameters

    def test_filling_it_replaces_the_parameter_with_the_inner_ones(self):
        cost = build(ESTIMATE)
        assert "oracle_gates" not in cost.parameters
        assert {"d", "kappa", "A_1", "A_max"} <= set(cost.parameters)

    def test_the_remaining_slots_are_reported(self):
        assert set(build(ESTIMATE).slots) == {"prepare_gates"}

    def test_a_slot_refuses_the_wrong_unit(self):
        with pytest.raises(UnitMismatch, match="takes gates"):
            hb.get("QAE").cost(Unit.GATES).bind(
                oracle_gates=hb.get("QSearch").cost(Unit.ITERATIONS)
            )

    def test_an_unknown_slot_is_named(self):
        with pytest.raises(ValueError, match="no slot 'nonesuch'"):
            hb.get("QAE").cost(Unit.GATES).bind(
                nonesuch=hb.get("FindRow").cost(Unit.GATES)
            )


class TestCompositionStaysHonest:
    def test_the_weaker_bound_wins(self):
        """An exact count wrapped around a lower bound is a lower bound."""
        composed = build(ESTIMATE)
        assert composed.provenance.bound is Bound.LOWER

    def test_assumptions_accumulate_across_papers(self):
        single = len(hb.get("QAE").cost(Unit.GATES).provenance.assumptions)
        composed = len(build(ESTIMATE).provenance.assumptions)
        assert composed > single

    def test_validity_domains_union(self):
        composed = build(ESTIMATE)
        assert len(composed.validity.conditions) >= len(
            hb.get("CanEnterNFP").cost(Unit.GATES).validity.conditions
        )

    def test_a_composition_still_refuses_meaningless_input(self):
        with pytest.raises(ValueError, match="no meaning"):
            build(ESTIMATE).evaluate(
                epsilon=1e-3, delta=0.25, n_qubits=10, prepare_gates=1.0,
                d=4, kappa=0.5, A_1=3.0, A_max=1.0,
            )


class TestTheThreeMoves:
    def test_adding_two_counts_of_the_same_thing(self):
        spec = {"op": "+", "terms": [ESTIMATE, {"routine": "FindRow",
                                                "unit": "GATES"}]}
        assert build(spec).unit is Unit.GATES

    def test_adding_different_things_is_refused(self):
        spec = {"op": "+", "terms": [{"routine": "QSearch", "unit": "ITERATIONS"},
                                     {"routine": "FindRow", "unit": "GATES"}]}
        with pytest.raises(UnitMismatch, match="cannot add"):
            build(spec)

    def test_multiplying_repetitions_by_what_is_repeated(self):
        spec = {"op": "*", "terms": [{"routine": "QSearch", "unit": "ITERATIONS"},
                                     {"routine": "CanEnterNFN", "unit": "GATES"}]}
        assert build(spec).unit is Unit.GATES

    def test_multiplying_two_absolute_counts_is_refused(self):
        spec = {"op": "*", "terms": [{"routine": "FindRow", "unit": "GATES"},
                                     {"routine": "IsOptimal", "unit": "GATES"}]}
        with pytest.raises(UnitMismatch, match="must be a multiplier"):
            build(spec)

    def test_an_empty_combination_is_refused(self):
        with pytest.raises(ValueError, match="at least one term"):
            build({"op": "+", "terms": []})


class TestRebuildingSomethingKnown:
    """A hand-assembled iteration should agree with the registered one.

    Not a coincidence -- it is the same four costs, added the same way -- but
    if it ever stopped agreeing, one of the two would be wrong.
    """

    PARAMS = dict(d=4, kappa=10.0, epsilon=1e-3, delta=1e-3, A_1=3.0,
                  A_max=1.0, n=200, m=50, t=5, c_max=2.0, u_norm=1.5,
                  t_improving=40)

    def test_it_matches_the_registered_iteration(self):
        spec = {"op": "+", "terms": [
            {"routine": "IsOptimal", "unit": "GATES"},
            {"routine": "FindColumn/steepest-edge", "unit": "GATES"},
            {"routine": "IsUnbounded", "unit": "GATES"},
            {"routine": "FindRow", "unit": "GATES"},
        ]}
        assembled = build(spec).evaluate(**self.PARAMS).value
        registered = hb.get("SimplexIter/steepest-edge").evaluate(
            Unit.GATES, **self.PARAMS
        ).value
        assert assembled == pytest.approx(registered)

    def test_swapping_the_pivoting_rule_changes_it(self):
        def total(rule):
            spec = {"op": "+", "terms": [
                {"routine": "IsOptimal", "unit": "GATES"},
                {"routine": "FindColumn/" + rule, "unit": "GATES"},
                {"routine": "IsUnbounded", "unit": "GATES"},
                {"routine": "FindRow", "unit": "GATES"},
            ]}
            return build(spec).evaluate(**self.PARAMS).value

        assert total("steepest-edge") != total("random")


class TestWhatTheInterfaceNeeds:
    def test_describe_reports_everything_needed_to_draw_it(self):
        data = describe(ESTIMATE)
        assert data["unit"] == "GATES"
        assert data["bound"] == "lower bound"
        assert "prepare_gates" in data["open_slots"]
        assert data["parameters"] and data["conditions"] and data["snippet"]

    def test_the_snippet_reproduces_the_composition(self):
        values = dict(epsilon=1e-3, delta=0.25, n_qubits=10,
                      prepare_gates=1.0, d=4, kappa=10.0, A_1=3.0, A_max=1.0)
        expected = build(ESTIMATE).evaluate(**values).value
        namespace = {}
        script = code(ESTIMATE, values).replace(
            "cost.evaluate(", "result = cost.evaluate("
        )
        exec(compile(script, "<snippet>", "exec"), namespace)
        assert namespace["result"].value == pytest.approx(expected)

    def test_fillings_only_offers_compatible_routines(self):
        for path in fillings_for("GATES"):
            assert Unit.GATES in hb.get(path).units

    def test_every_entry_offered_has_a_cost(self):
        for entry in entries():
            assert entry["units"]

    def test_a_step_without_a_routine_is_refused(self):
        with pytest.raises(ValueError, match="names a routine"):
            build({"unit": "GATES"})
