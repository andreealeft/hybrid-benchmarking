"""The amplification generic, and the claim that it is genuinely one thing.

The design rests on the assertion that quantum search, amplitude amplification
and find-all-marked are the same expression with different parameters.  If that
is true it should be checkable, so it is checked here.
"""

from __future__ import annotations

import math

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.routines.amplification import (
    qaa_overhead,
    qsearch_all_iterations,
    qsearch_iterations,
    qsearch_k_max,
    rounds,
)


class TestOneGeneric:
    def test_search_is_truncated_amplification(self):
        """Quantum search is amplification, truncated, times one half.

        The two published statements differ only in the round budget and in a
        factor that reflects counting Grover iterations rather than
        applications of the amplified algorithm.
        """
        for length, marked in [(64, 1), (1024, 1), (4096, 7), (10_000, 3)]:
            assert qsearch_iterations(length, marked) == pytest.approx(
                0.5 * rounds(
                    p=marked / length,
                    p0=1 / length,
                    k_max=qsearch_k_max(length),
                )
            )

    def test_find_all_sums_over_a_shrinking_list(self):
        """Finding all t marked elements is t successive searches."""
        length, marked = 500, 4
        expected = sum(
            qsearch_iterations(length - i, marked - i) for i in range(marked)
        )
        assert qsearch_all_iterations(length, marked) == pytest.approx(expected)

    def test_finding_all_costs_more_than_finding_one(self):
        assert qsearch_all_iterations(500, 4) > qsearch_iterations(500, 4)


class TestScaling:
    def test_search_is_quadratically_better_than_scanning(self):
        """Iterations should track sqrt of the list length, not the length."""
        for length in [100, 1_000, 10_000, 100_000]:
            iterations = qsearch_iterations(length, 1)
            assert 0.3 * math.sqrt(length) < iterations < 5 * math.sqrt(length)

    def test_more_marked_elements_are_cheaper_to_hit(self):
        many = qsearch_iterations(10_000, 50)
        few = qsearch_iterations(10_000, 1)
        assert many < few

    def test_overhead_falls_as_success_probability_rises(self):
        assert qaa_overhead(0.5, 0.5) < qaa_overhead(0.01, 0.01)

    def test_certain_success_needs_one_application(self):
        assert qaa_overhead(1.0, 1.0) == 1.0

    def test_a_looser_bound_never_helps(self):
        """The lower bound only caps how far the schedule may grow.

        Because the schedule grows geometrically from one and stops on
        success, being handed a much weaker bound costs surprisingly little --
        it raises the ceiling without changing how quickly the search gets
        there.  That is the point of the geometric schedule, and it is worth
        having pinned down: the expensive ignorance is about the true success
        probability, not about its bound.
        """
        informed = qaa_overhead(p=0.25, p0=0.25)
        ignorant = qaa_overhead(p=0.25, p0=0.0001)
        assert ignorant >= informed
        assert ignorant < 2 * informed


class TestGuards:
    def test_lower_bound_must_actually_bound(self):
        with pytest.raises(ValueError, match="not a lower bound"):
            rounds(p=0.01, p0=0.5)

    def test_probabilities_stay_probabilities(self):
        with pytest.raises(ValueError):
            rounds(p=1.5, p0=0.1)
        with pytest.raises(ValueError):
            rounds(p=0.5, p0=0.0)


class TestThroughTheRegistry:
    def test_evaluate_uses_the_kernel(self):
        cost = hb.get("QSearch").evaluate(X=1_000_000, t=1)
        assert cost.unit is hb.Unit.ITERATIONS
        assert cost.value == pytest.approx(qsearch_iterations(1_000_000, 1))

    def test_symbolic_until_evaluated(self):
        cost = hb.get("QSearch").cost(hb.Unit.ITERATIONS)
        assert set(cost.parameters) == {"X", "t"}
        assert not cost.is_numeric
        with pytest.raises(ValueError):
            _ = cost.value

    def test_missing_parameters_are_named(self):
        with pytest.raises(ValueError, match="missing parameters: t"):
            hb.get("QSearch").evaluate(X=100)

    def test_meaningless_parameters_are_refused_outright(self):
        """More marked elements than list entries is not a regime question."""
        with pytest.raises(ValueError, match="no meaning"):
            hb.get("QSearch").evaluate(X=10, t=50)


class TestRegimeVersusNonsense:
    """A formula used outside its regime still computes; nonsense does not.

    The linear solvers will exercise the soft path for real -- their query
    counts hold only while the precision sits far below the inverse condition
    number -- so the mechanism is checked here on a minimal cost.
    """

    @staticmethod
    def _cost():
        from hybrid_benchmarking import symbols as S
        from hybrid_benchmarking.cost import Cost
        from hybrid_benchmarking.validity import Validity, much_less_than

        return Cost(
            expr=S.kappa / S.epsilon,
            unit=hb.Unit.QUERIES,
            validity=Validity((much_less_than(S.epsilon, 1 / S.kappa),)),
        )

    def test_inside_the_regime_is_quiet(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._cost().evaluate(kappa=10, epsilon=1e-6).value > 0

    def test_outside_the_regime_warns_and_is_recorded(self):
        with pytest.warns(hb.ValidityWarning, match="derived regime"):
            cost = self._cost().evaluate(kappa=10, epsilon=0.5)
        assert cost.value == pytest.approx(20)
        assert any("outside its derived regime" in a
                   for a in cost.provenance.assumptions)

    def test_strict_refuses_instead_of_warning(self):
        with pytest.raises(ValueError, match="outside its derived regime"):
            self._cost().evaluate(kappa=10, epsilon=0.5, strict=True)
