"""The four functional quantum linear solvers.

The thesis's conclusions about these are specific enough to be tests: HHL needs
orders of magnitude more oracle queries than the rest and is excluded on that
basis, Chebyshev and QSVT stay lowest, Fourier sits about an order above them.
Those orderings are asserted here so that a later change to any shared
ingredient cannot quietly invert them.
"""

from __future__ import annotations

import math

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.routines.linsolve import (
    binkowski_chebyshev_queries,
    chebyshev_alpha,
    chebyshev_degree_parameter,
    chebyshev_queries,
    chebyshev_terms,
    fourier_alpha,
    fourier_queries,
    hhl_queries,
    qsvt_queries,
    _binomial_upper_tail,
)

SOLVERS = ("HHL", "QLS-Fourier", "QLS-Chebyshev", "QLS-QSVT")

#: A well-conditioned sparse instance, in the regime all four were derived for.
INSTANCE = dict(d=4, kappa=10.0, epsilon=1e-8, x_norm=1.0, A_max=1.0)


def _queries(name, **overrides):
    params = dict(INSTANCE, **overrides)
    routine = hb.get(name)
    wanted = set(routine.parameters)
    return routine.evaluate(
        hb.Unit.QUERIES, **{k: v for k, v in params.items() if k in wanted}
    ).value


class TestAllFour:
    @pytest.mark.parametrize("name", SOLVERS)
    def test_each_is_registered_and_costs_queries(self, name):
        cost = hb.get(name).cost(hb.Unit.QUERIES)
        assert cost.unit is hb.Unit.QUERIES
        assert not cost.is_numeric

    @pytest.mark.parametrize("name", ("HHL", "QLS-Chebyshev", "QLS-QSVT"))
    def test_no_gate_count_until_the_oracles_are_fixed(self, name):
        with pytest.raises(ValueError, match="no gates count|has no gates"):
            hb.get(name).cost(hb.Unit.GATES)

    def test_fourier_is_the_exception(self):
        """"Linear solvers are query-only" is true of the comparison, not in
        general.  Nannicini's setting pins the oracles down, so the Fourier
        solver acquires a gate count there -- and it is a different
        construction, evolving the matrix by Berry's reduction rather than by
        qubitization.
        """
        routine = hb.get("QLS-Fourier")
        assert {i.name for i in routine.implementations} == {
            "via-qubitization", "via-berry"
        }
        assert routine.cost(hb.Unit.QUERIES).unit is hb.Unit.QUERIES
        assert routine.cost(hb.Unit.GATES).unit is hb.Unit.GATES

    def test_the_gate_count_is_a_lower_bound_the_query_count_is_exact(self):
        assert hb.get("QLS-Fourier/via-berry").cost().provenance.bound \
            is hb.Bound.LOWER
        assert hb.get("QLS-Fourier/via-qubitization").cost().provenance.bound \
            is hb.Bound.EXACT

    def test_the_gate_count_composes_as_amplification_times_simulation(self):
        from hybrid_benchmarking.routines.hamsim import berry_gates
        from hybrid_benchmarking.routines.linsolve import (
            fourier_alpha, fourier_gates, fourier_time, oaa_rounds,
        )

        d, kappa, epsilon, norm_1, norm_max = 4, 10.0, 1e-3, 3.0, 1.0
        expected = oaa_rounds(fourier_alpha(kappa, epsilon)) * berry_gates(
            epsilon=epsilon, d=d, t=fourier_time(kappa, epsilon),
            norm_1=norm_1, norm_max=norm_max,
        )
        assert fourier_gates(d, kappa, epsilon, norm_1, norm_max) == \
            pytest.approx(expected)

    @pytest.mark.parametrize("name", SOLVERS)
    def test_all_positive_and_finite(self, name):
        value = _queries(name)
        assert value > 0
        assert math.isfinite(value)

    @pytest.mark.parametrize("name", SOLVERS)
    def test_worse_conditioning_costs_more(self, name):
        assert _queries(name, kappa=100.0) > _queries(name, kappa=10.0)

    @pytest.mark.parametrize("name", SOLVERS)
    def test_tighter_precision_costs_more(self, name):
        assert _queries(name, epsilon=1e-10) > _queries(name, epsilon=1e-6)


class TestPublishedOrdering:
    """The qualitative results of chapter 5, as executable assertions."""

    def test_hhl_is_far_more_expensive_than_the_rest(self):
        hhl = _queries("HHL")
        others = [_queries(n) for n in SOLVERS if n != "HHL"]
        assert hhl > 1e4 * max(others)

    def test_chebyshev_and_qsvt_are_the_cheapest(self):
        cheapest = min(SOLVERS, key=_queries)
        assert cheapest in ("QLS-Chebyshev", "QLS-QSVT")

    def test_fourier_sits_above_chebyshev_and_qsvt(self):
        fourier = _queries("QLS-Fourier")
        assert fourier > _queries("QLS-Chebyshev")
        assert fourier > _queries("QLS-QSVT")

    def test_the_cheap_solvers_stay_below_a_hundred_million(self):
        for name in ("QLS-Chebyshev", "QLS-QSVT"):
            assert _queries(name) < 1e8

    def test_hhls_precision_dependence_is_the_reason(self):
        """HHL scales as 1/eps where the others scale as log(1/eps)."""
        ratio_hhl = _queries("HHL", epsilon=1e-9) / _queries("HHL", epsilon=1e-8)
        ratio_cheb = (_queries("QLS-Chebyshev", epsilon=1e-9)
                      / _queries("QLS-Chebyshev", epsilon=1e-8))
        assert ratio_hhl > 5
        assert ratio_cheb < 2


class TestAgreementBetweenPapers:
    """Lemma 16 of the thesis and Lemma 1 of the interior point paper are the
    same result, reached independently. One registry entry has to reproduce
    both, otherwise the two papers disagree and that is worth knowing.
    """

    @pytest.mark.parametrize("d,kappa,epsilon", [
        (4, 10.0, 1e-8),
        (2, 50.0, 1e-6),
        (16, 3.0, 1e-10),
    ])
    def test_the_query_counts_coincide(self, d, kappa, epsilon):
        ours = 8.0 * chebyshev_terms(d * kappa, epsilon)
        theirs = binkowski_chebyshev_queries(d, kappa, epsilon, n_qaa=1.0)
        assert ours == pytest.approx(theirs)

    def test_they_would_disagree_under_the_linear_reading(self):
        """Guards the (B.74) ruling: dropping the square breaks the agreement."""
        d, kappa, epsilon = 4, 10.0, 1e-8
        linear_s = math.ceil(d * kappa * math.log2(d * kappa / epsilon))
        linear_j0 = math.ceil(
            math.sqrt(linear_s * math.log2(4.0 * linear_s / epsilon))
        )
        assert 8.0 * linear_j0 != pytest.approx(
            binkowski_chebyshev_queries(d, kappa, epsilon, n_qaa=1.0)
        )


class TestChebyshevInternals:
    def test_the_binomial_tail_matches_its_normal_limit(self):
        """The closed form used for large s agrees with exact summation."""
        s = 1500
        exact = sum(_binomial_upper_tail(s, j) for j in range(0, 400))
        limit = 0.5 * math.sqrt(s / math.pi)
        assert exact == pytest.approx(limit, rel=0.02)

    def test_the_truncation_never_binds(self):
        """j0 grows like sqrt(s log s); the deviation has scale sqrt(s)."""
        for kappa in (5.0, 50.0, 500.0):
            s = chebyshev_degree_parameter(4 * kappa, 1e-8)
            assert chebyshev_terms(4 * kappa, 1e-8) > 6 * math.sqrt(s / 2)

    def test_alpha_is_positive_and_grows_with_conditioning(self):
        low = chebyshev_alpha(4, 4 * 10.0, 1e-8)
        high = chebyshev_alpha(4, 4 * 100.0, 1e-8)
        assert 0 < low < high


class TestFourierInternals:
    def test_alpha_tracks_its_integral_approximation(self):
        """The summand is a Riemann sum for the integral of u exp(-u^2/2)."""
        kappa, epsilon = 50.0, 1e-8
        log_term = math.log(1.0 + 8.0 * kappa / epsilon)
        delta_z = (2.0 * math.pi / (kappa + 1.0)) / math.sqrt(log_term)
        approximate = 4.0 * math.sqrt(math.pi) * kappa / (kappa + 1.0) / delta_z
        assert fourier_alpha(kappa, epsilon) == pytest.approx(
            approximate, rel=0.05
        )


class TestRegime:
    def test_outside_the_precision_regime_it_warns(self):
        """All four hold only while eps sits far below 1/kappa."""
        with pytest.warns(hb.ValidityWarning, match="derived regime"):
            hb.get("QLS-Chebyshev").evaluate(
                hb.Unit.QUERIES, d=4, kappa=10.0, epsilon=0.5, x_norm=1.0
            )

    def test_a_condition_number_below_one_is_refused(self):
        with pytest.raises(ValueError, match="no meaning"):
            hb.get("QLS-Chebyshev").evaluate(
                hb.Unit.QUERIES, d=4, kappa=0.5, epsilon=1e-8, x_norm=1.0
            )


class TestWhatEachIsBuiltFrom:
    def test_only_hhl_and_fourier_simulate(self):
        """Chebyshev walks and QSVT block-encodes; neither pays for simulation."""
        assert "HamSim/qubitization" in hb.get("HHL").implementation().built_from
        assert "HamSim/qubitization" in \
            hb.get("QLS-Fourier/via-qubitization").built_from
        for name in ("QLS-Chebyshev", "QLS-QSVT"):
            built = hb.get(name).implementation().built_from
            assert not any(b.startswith("HamSim") for b in built)

    def test_query_counts_never_reach_for_berrys_construction(self):
        """Berry's reduction is costed in gates; mixing it into a query count
        would be the substitution the implementation layer exists to prevent.
        """
        for impl in hb.all_implementations():
            if hb.Unit.QUERIES in impl.costs:
                assert "HamSim/berry" not in impl.built_from

    def test_the_gate_count_never_reaches_for_qubitization(self):
        for impl in hb.all_implementations():
            if hb.Unit.GATES in impl.costs:
                assert "HamSim/qubitization" not in impl.built_from
