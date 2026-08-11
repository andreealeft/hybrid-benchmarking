"""The generic wrappers of Appendix A.3.

These take the cost of whatever they wrap as a parameter, so handing them a
cost of one turns the gate count into a call count -- which is the clearest way
to check that the prefactors are right.
"""

from __future__ import annotations

import math

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.provenance import Bound, Unit
from hybrid_benchmarking.routines.standard import (
    clock_qubits,
    controlled_gates,
    lcu_gates,
    minimum_finding_queries,
    oaa_gates,
    qaa_gates,
    qae_gates,
    qmf_gates,
    qpe_gates,
)

WRAPPERS = ("QPE", "QAE", "QMF", "QAA-fixed", "OAA", "LCU", "CtrlU")


class TestAllRegistered:
    @pytest.mark.parametrize("name", WRAPPERS)
    def test_costed_in_gates_as_a_lower_bound(self, name):
        cost = hb.get(name).cost(Unit.GATES)
        assert cost.unit is Unit.GATES
        assert cost.provenance.bound is Bound.LOWER

    @pytest.mark.parametrize("name", WRAPPERS)
    def test_the_gate_model_is_recorded_not_assumed_silently(self, name):
        assumptions = hb.get(name).cost(Unit.GATES).provenance.assumptions
        assert any("cost one gate" in a for a in assumptions)


class TestPhaseEstimation:
    def test_the_clock_width_is_logarithmic_in_both_parameters(self):
        assert clock_qubits(1e-3, 0.25) == pytest.approx(
            math.ceil(math.log2(1e3) + math.log2(3.0))
        )
        assert clock_qubits(1e-6, 0.25) > clock_qubits(1e-3, 0.25)
        assert clock_qubits(1e-3, 0.01) > clock_qubits(1e-3, 0.25)

    def test_the_cost_is_exponential_in_that_width(self):
        """A logarithmic register width, and then two to its power in calls."""
        coarse = qpe_gates(1e-2, 0.25, inner_gates=1.0)
        fine = qpe_gates(1e-4, 0.25, inner_gates=1.0)
        assert fine > 50 * coarse

    def test_with_a_unit_inner_cost_it_counts_calls(self):
        n_c = clock_qubits(1e-3, 0.25)
        assert qpe_gates(1e-3, 0.25, 1.0) == n_c + 2 ** n_c - 1


class TestAmplitudeEstimation:
    def test_it_is_one_preparation_plus_phase_estimation(self):
        params = dict(epsilon=1e-3, delta=0.25, qubits=10,
                      prepare_gates=1.0, oracle_gates=1.0)
        grover = 1.0 + 2 * 1.0 + 2.0 + 1.0 + 2.0 * 9
        assert qae_gates(**params) == pytest.approx(
            1.0 + qpe_gates(1e-3, 0.25, grover)
        )

    def test_a_dearer_oracle_costs_more(self):
        base = dict(epsilon=1e-3, delta=0.25, qubits=10, prepare_gates=1.0)
        assert qae_gates(oracle_gates=100.0, **base) > \
            qae_gates(oracle_gates=1.0, **base)


class TestMinimumFinding:
    def test_the_rank_sum_starts_at_zero_by_default(self):
        """The final search finds nothing and exhausts its schedule; that is
        what tells the algorithm it is done, and it is the largest term."""
        from_zero = minimum_finding_queries(50, first_rank=0)
        from_one = minimum_finding_queries(50, first_rank=1)
        assert from_zero > from_one

    def test_the_two_readings_differ_by_the_terminating_search(self):
        from hybrid_benchmarking.routines.amplification import qsearch_iterations

        assert minimum_finding_queries(50, 0) - minimum_finding_queries(50, 1) \
            == pytest.approx(qsearch_iterations(50, 0))

    def test_longer_lists_cost_more(self):
        assert minimum_finding_queries(200) > minimum_finding_queries(50)

    def test_the_gate_count_pays_per_round(self):
        rounds = minimum_finding_queries(40)
        per_round = (1.0 + 2 * 1.0 + 2.0 + 1.0 + 2.0 * 5) + 6 + 1.0
        assert qmf_gates(40, qubits=6, prepare_gates=1.0, oracle_gates=1.0) \
            == pytest.approx(rounds * per_round)


class TestTwoSchedulesForAmplification:
    """The fixed schedule and the geometric one are different algorithms."""

    def test_a_known_probability_gets_the_fixed_schedule(self):
        assert hb.get("QAA-fixed").summary.startswith("Amplitude amplification")
        assert "fixed schedule" in hb.get("QAA-fixed").summary

    def test_the_geometric_one_is_a_separate_entry(self):
        assert hb.get("QAA").units == (Unit.ITERATIONS,)
        assert hb.get("QAA-fixed").units == (Unit.GATES,)

    def test_a_higher_success_probability_needs_fewer_rounds(self):
        base = dict(qubits=10, prepare_gates=1.0, oracle_gates=1.0)
        assert qaa_gates(0.5, **base) < qaa_gates(0.01, **base)

    def test_certain_success_needs_no_rounds(self):
        assert qaa_gates(1.0, qubits=4, prepare_gates=1.0, oracle_gates=1.0) \
            == pytest.approx(1.0)

    def test_oblivious_amplification_needs_no_marker(self):
        """Its reflection is about the ancilla register, so nothing marks."""
        import inspect

        assert "oracle" not in inspect.signature(oaa_gates).parameters


class TestCombinationAndControl:
    def test_a_combination_pays_for_every_term(self):
        assert lcu_gates(terms=64, inner_gates=100.0) > \
            lcu_gates(terms=8, inner_gates=100.0)

    def test_controls_cost_toffolis(self):
        assert controlled_gates(8, 1.0) == 2 * 7 + 1
        assert controlled_gates(1, 1.0) == 1


class TestSimplexCoreWrappers:
    def test_interference_is_at_least_both_unitaries(self):
        assert hb.get("Interfere").evaluate(
            Unit.GATES, inner_gates=10.0, inner_gates_2=5.0
        ).value == pytest.approx(15.0)

    def test_no_false_positives_costs_nine_times_no_false_negatives(self):
        """Same construction, a factor of nine in the estimation precision."""
        nfn = hb.get("SignEstNFN").evaluate(
            Unit.GATES, epsilon=1e-6, inner_gates=1.0).value
        nfp = hb.get("SignEstNFP").evaluate(
            Unit.GATES, epsilon=1e-6, inner_gates=1.0).value
        assert nfp / nfn == pytest.approx(9.0, rel=1e-5)

    def test_both_are_built_on_interference_and_estimation(self):
        for name in ("SignEstNFN", "SignEstNFP"):
            built = hb.get(name).implementation().built_from
            assert "Interfere" in built and "QAE" in built


class TestTheRegistryIsComplete:
    def test_every_entry_has_a_summary_and_a_citation(self):
        for impl in hb.all_implementations():
            assert impl.summary, impl.path
            assert impl.citation, impl.path

    def test_nothing_claims_a_unit_it_cannot_compute(self):
        for impl in hb.all_implementations():
            for unit, cost in impl.costs.items():
                assert cost.unit is unit, impl.path

    def test_names_never_contain_the_path_separator(self):
        for name in hb.names():
            assert "/" not in name
