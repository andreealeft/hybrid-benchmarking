"""The Cade family, checked against QUBRABENCH rather than against itself.

These four entries carried no formulas at all until now, on the stated grounds
that supplying them would mean transcribing results this library had not read.
The reason was sound and it expired: the results have now been read, from
QUBRABENCH's ``algorithms/`` modules and the papers they cite, and reimplemented
here the way everything else in this repository is -- from the statement, with
the original kept as a fixture to check against rather than a source to copy.

So the checking is the point of this file. Every value below was produced by
running QUBRABENCH's own functions, and is recorded to seventeen digits so that
a drift of one part in ten thousand fails rather than passes. They are golden
values rather than a live comparison because the tests here run offline and
QUBRABENCH is not a dependency; regenerating them needs only its four
``qubrabench/algorithms/*.py`` files and the harness in this module's history.

The other half of the file is the boundary that matters more than the numbers.
A count here is calls to a subroutine the caller supplied. A count in
``QUERIES`` is calls to a sparse-access oracle. Both traditions say "query",
they are not the same quantity, and the units refuse to add -- which is the
single most dangerous collision in this merge and the reason these entries exist
at all.
"""

from __future__ import annotations

import math

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.cost import UnitMismatch, exact
from hybrid_benchmarking.provenance import Bound, Derivation, Unit
from hybrid_benchmarking.routines import cade

#: (kind, arguments, what QUBRABENCH returns). See the module docstring.
REFERENCE = [
    ('search', (1000, 1, 0.001, 10), 200.63762744907854),
    ('search', (1000, 0, 0.001, 10), 2036.5068131484359),
    ('search', (1000000, 5, 0.01, 100), 2196.0892214522837),
    ('search', (500, 120, 0.001, 0), 6.5893528185599255),
    ('search', (10000, 2500, 1e-06, 7), 0.54372107008102533),
    ('search_classical', (1000, 1, 10), 9.9551197902517643),
    ('search_classical', (1000, 0, 10), 10),
    ('search_classical', (500, 120, 3), 2.3376000000000001),
    ('search_worst', (1000, 0.001), 423.19667392950714),
    ('search_worst', (1000000, 0.01), 9469.7779607693792),
    ('max', (100, 0.001), 894.3052474176144),
    ('max', (1000, 0.01), 2299.2430086831082),
    ('max', (2000, 1e-06), 8463.7991932038694),
    ('linalg', (2.0, 50.0, 0.001), 19825.720104922726),
    ('linalg', (1.0, 1000.0, 1e-06), 281028.09484427981),
    ('linalg', (5.0, 3.0, 0.1), 2119.5329893146436),
    ('linalg', (1.0, 2.0, 0.01), 844.77579424164401),
    ('linalg_fail', (2.0, 50.0, 0.001, 0.001), 277247.79052145057),
    ('linalg_fail', (1.0, 100.0, 0.01, 0.5), 31188.656732821157),
    ('linalg_fail', (1.0, 100.0, 0.01, 0.9), 22093.198241153703),
]

_CALL = {
    "search": lambda a: cade.search_quantum_calls(*a),
    "search_classical": lambda a: cade.search_classical_calls(*a),
    "search_worst": lambda a: cade.search_worst_case_calls(*a),
    "max": lambda a: cade.max_quantum_calls(*a),
    "linalg": lambda a: cade.dalzell_queries(*a),
    "linalg_fail": lambda a: cade.dalzell_queries_with_failure(*a),
}


class TestAgainstQubrabench:
    @pytest.mark.parametrize("kind,args,expected", REFERENCE,
                             ids=["{}{}".format(k, a) for k, a, _ in REFERENCE])
    def test_we_reproduce_what_qubrabench_computes(self, kind, args, expected):
        assert _CALL[kind](args) == pytest.approx(expected, rel=1e-12)

    def test_the_search_constant_is_equation_threes_own(self):
        # F is 2.0344 outside 1 <= t < N/4, because past a quarter of the space
        # being marked there is nothing left for amplification to buy.
        assert cade.cade_F(1000, 500) == 2.0344
        assert cade.cade_F(1000, 0) == 2.0344
        assert cade.cade_F(1000, 1) != 2.0344

    def test_nothing_marked_costs_the_full_schedule(self):
        # The same shape this library's FindRow pays at r = 0, reached
        # independently from a different paper.
        exhausted = cade.search_quantum_calls(1000, 0, 1e-3, 10)
        found = cade.search_quantum_calls(1000, 1, 1e-3, 10)
        assert exhausted > found

    def test_a_looser_failure_probability_never_costs_more(self):
        tight = cade.search_quantum_calls(1000, 1, 1e-6, 0)
        loose = cade.search_quantum_calls(1000, 1, 1e-1, 0)
        assert loose <= tight

    def test_the_condition_number_is_raised_where_the_expression_dies(self):
        # QUBRABENCH clamps to sqrt(12); below it Dalzell's Theorem 1 has no
        # meaning, and a number out of it would be worse than none.
        assert cade.dalzell_queries(1.0, 1.0, 1e-3) == \
            cade.dalzell_queries(1.0, math.sqrt(12), 1e-3)


class TestTheBoundaryTheseEntriesExistFor:
    def test_subroutine_calls_are_their_own_unit(self):
        assert Unit.SUBROUTINE_CALLS is not Unit.QUERIES
        assert str(Unit.SUBROUTINE_CALLS) == "subroutine calls"

    def test_they_cannot_be_added_to_oracle_queries(self):
        with pytest.raises(UnitMismatch, match="cannot add"):
            exact(1, Unit.SUBROUTINE_CALLS) + exact(1, Unit.QUERIES)

    @pytest.mark.parametrize("name", ["Cade-search", "Cade-max",
                                      "Cade-amplitude", "Cade-linalg"])
    def test_every_entry_now_carries_a_formula(self, name):
        assert hb.get(name).units == (Unit.SUBROUTINE_CALLS,)

    def test_the_linear_solver_is_kept_out_of_the_query_unit_on_purpose(self):
        # Dalzell's Theorem 1 counts queries to a block encoding; the four
        # functional solvers count sparse-access oracle queries. Different
        # access models, same word. Putting them in one unit would let a table
        # compare them.
        assert hb.get("Cade-linalg").units == (Unit.SUBROUTINE_CALLS,)
        assert hb.get("QLS-QSVT").units == (Unit.QUERIES,)
        with pytest.raises(UnitMismatch):
            (hb.get("Cade-linalg").cost(Unit.SUBROUTINE_CALLS)
             + hb.get("QLS-QSVT").cost(Unit.QUERIES))

    def test_the_distinction_is_stated_where_a_reader_will_meet_it(self):
        summary = hb.get("Cade-search").summary.lower()
        assert "not" in summary and "oracle" in summary

    @pytest.mark.parametrize("name", ["Cade-search", "Cade-max",
                                      "Cade-amplitude", "Cade-linalg"])
    def test_the_count_is_logged_rather_than_analytic(self, name):
        # The formula is exact; its inputs are what one classical run met.
        cost = hb.get(name).cost(Unit.SUBROUTINE_CALLS)
        assert cost.provenance.derivation is Derivation.LOGGED
        assert cost.provenance.bound is Bound.UPPER


class TestTheyEvaluateThroughTheOrdinaryPath:
    def test_each_evaluates_from_the_registry(self):
        assert hb.get("Cade-search").evaluate(
            X=1000, t=1, eta=1e-3, K=10).value > 0
        assert hb.get("Cade-max").evaluate(X=1000, eta=1e-3).value > 0
        assert hb.get("Cade-amplitude").evaluate(
            a=0.25, epsilon=1e-3, eta=1e-3).value > 0
        assert hb.get("Cade-linalg").evaluate(
            alpha=2.0, kappa=50.0, epsilon=1e-3, eta=1e-3).value > 0

    def test_the_registry_count_includes_the_phase_oracle_factor(self):
        # QUBRABENCH applies the factor of two at its call site; it is applied
        # inside the cost here, so what a reader sees is calls that happen.
        bare = cade.search_quantum_calls(1000, 1, 1e-3, 10)
        registered = hb.get("Cade-search").evaluate(
            X=1000, t=1, eta=1e-3, K=10).value
        assert registered == pytest.approx(cade.PHASE_ORACLE_CALLS * bare)

    def test_marking_more_than_the_space_holds_is_refused(self):
        with pytest.raises(ValueError, match="no meaning"):
            hb.get("Cade-search").evaluate(X=10, t=50, eta=1e-3, K=1)
