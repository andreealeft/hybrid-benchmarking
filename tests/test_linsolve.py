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
    fourier_time,
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
        # Base two, as everything in this chapter's query counts now is.
        log_term = math.log2(1.0 + 8.0 * kappa / epsilon)
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


class TestTheLogarithmIsBaseTwo:
    """Chapter 5 writes a bare ``log`` in some lemmas and ``log2`` in others.

    Lemma 15 states the Fourier weight and simulation time with a bare ``log``;
    Lemma 16 states the Chebyshev truncation degree with an explicit ``log2``.
    The two readings differ by a factor of ``ln 2`` that does not cancel -- it
    propagates through a square root into every query count -- so following each
    lemma's own typography would give one solver a base the others do not have.
    Andreea ruled base two throughout, and these tests hold it there.
    """

    def test_the_fourier_weight_uses_base_two(self):
        kappa, epsilon = 50.0, 1e-8
        # Written out independently: alpha ~ 4 sqrt(pi) kappa/(kappa+1) / dz,
        # with dz set by the logarithm. Base two fixes the value.
        log_term = math.log2(1.0 + 8.0 * kappa / epsilon)
        delta_z = (2.0 * math.pi / (kappa + 1.0)) / math.sqrt(log_term)
        assert fourier_alpha(kappa, epsilon) == pytest.approx(
            4.0 * math.sqrt(math.pi) * kappa / (kappa + 1.0) / delta_z, rel=0.05)

    def test_the_simulation_time_uses_base_two(self):
        kappa, epsilon = 50.0, 1e-8
        assert fourier_time(kappa, epsilon) == pytest.approx(
            2.0 * math.sqrt(2.0) * kappa
            * math.log2(1.0 + 8.0 * kappa / epsilon))

    def test_the_truncation_degree_uses_base_two(self):
        # Lemma 16 spells this one out, and it always agreed.
        y, epsilon = 200.0, 1e-8
        s = math.ceil(y ** 2 * math.log2(y / epsilon))
        assert chebyshev_terms(y, epsilon) == math.ceil(
            math.sqrt(s * math.log2(4.0 * s / epsilon)))

    def test_taking_natural_logs_instead_would_move_fourier_by_root_two(self):
        # Why it matters: the discrepancy is not a rounding difference. The
        # weight scales as the square root of the logarithm, so the two bases
        # differ by sqrt(ln 2) ~ 0.83 -- and the query count by its inverse.
        natural = 4.0 * math.sqrt(math.pi) * 50.0 / 51.0 / (
            (2.0 * math.pi / 51.0) / math.sqrt(math.log(1.0 + 8.0 * 50.0 / 1e-8)))
        assert fourier_alpha(50.0, 1e-8) / natural == pytest.approx(
            1.0 / math.sqrt(math.log(2.0)), rel=0.05)


class TestAgainstTheQlsComparisonRepository:
    """What the original Method 2 code computes, and where it parts from the thesis.

    Checked against ``evaluation/query_costs.py`` in the qls-comparison
    repository -- a fixture, not an input. The values below were produced by
    running it. Where we agree, agreement is asserted; where we do not, the
    reason is a lemma this library follows and that code does not, and the test
    records which.
    """

    def test_the_fourier_weight_now_matches_the_original(self):
        """Both base two after the ruling, and agreeing to seven figures.

        Not to the last bit: the original forms the step as
        ``log**-0.5 * 2 pi / (k+1)`` and this forms it as
        ``(2 pi / (k+1)) / sqrt(log)``, which are the same number and not the
        same rounding, over a sum of some hundred terms.
        """
        for kappa, epsilon, expected in [(50.0, 1e-8, 334.81099265936706),
                                         (10.0, 1e-3, 45.463135220920765),
                                         (100.0, 1e-6, 613.6432392787432)]:
            assert fourier_alpha(kappa, epsilon) == pytest.approx(
                expected, rel=1e-6)

    def test_the_amplification_overhead_is_lemma_13_not_the_repositorys(self):
        """The original omits the floor on m_k and divides by 4 m_l where
        (5.40) divides by 4 (m_l + 1). Both push the same way at small success
        probabilities, where it reaches a factor of six."""
        from hybrid_benchmarking.routines.amplification import rounds

        def lemma_13(p, p0, c=1.2):
            theta, cap = math.asin(math.sqrt(p)), 1.0 / math.sqrt(p0)
            total, remaining, k = 0.0, 1.0, 1
            while remaining >= 1e-15 and k < 4000:
                m = math.floor(min(c ** k, cap))
                total += m * remaining
                remaining *= 1.0 - (0.5 - math.sin(4 * (m + 1) * theta)
                                    / (4 * (m + 1) * math.sin(2 * theta)))
                k += 1
            return total

        for p, p0 in [(0.25, 0.01), (0.5, 0.5), (1e-3, 1e-4), (0.01, 1e-3)]:
            assert rounds(p, p0, half=False) == pytest.approx(
                lemma_13(p, p0), rel=1e-12)

    def test_the_repositorys_amplification_values_are_the_ones_we_differ_from(self):
        """Recorded so the size of the gap is visible rather than asserted away.

        These are what qls-comparison returns. Ours is Lemma 13 exactly, times
        the ruled one half."""
        from hybrid_benchmarking.routines.amplification import rounds

        theirs = {(0.25, 0.01): 1.6845496382546455,
                  (1e-3, 1e-4): 446.28777261754886}
        for (p, p0), value in theirs.items():
            ours = rounds(p, p0, half=False)
            assert ours != pytest.approx(value, rel=1e-3)
            assert 0.15 < ours / value < 1.1  # same order, not the same number


class TestWhichLemmaGovernsWhichBase:
    """The base-two ruling settles the lemmas that leave the base open.

    It does not overrule one that says which base it means. Lemma 12 writes its
    crossover as ``t' >= ln(1/delta)/e`` and divides by ``log(e + ...)``, which
    is the identity it appears to be only in base e; Lemma 16 spells out
    ``log2``. Between them sit Lemma 15's weight and time and Lemma 17's
    exponential degree, which write a bare ``log`` and take base two.
    """

    def test_hamiltonian_simulation_keeps_the_natural_logs_lemma_12_states(self):
        from hybrid_benchmarking.routines.linsolve import _segments

        epsilon, t_prime = 1e-8, 1.0
        assert t_prime < math.log(1.0 / epsilon) / math.e  # the second branch
        assert _segments(t_prime, epsilon) == math.ceil(
            4.0 * math.log(1.0 / epsilon)
            / math.log(math.e + math.log(1.0 / epsilon) / t_prime))

    def test_the_crossover_is_tested_on_the_rescaled_time(self):
        # Lemma 12's condition is on t' = d ||A||max t, not on t. The original
        # repository tests the unscaled time; this follows the lemma.
        from hybrid_benchmarking.routines.linsolve import (
            _segments,
            simulation_queries,
        )

        d, norm_max, t, epsilon = 8.0, 1.0, 1.0, 1e-8
        assert simulation_queries(d, norm_max, t, epsilon) == \
            48.0 * _segments(d * norm_max * t, epsilon)
        # d = 8 pushes t' past the crossover although t alone is well below it.
        assert t < math.log(1.0 / epsilon) / math.e <= d * norm_max * t

    def test_the_truncation_degree_keeps_the_base_two_lemma_16_states(self):
        y, epsilon = 200.0, 1e-8
        s = math.ceil(y ** 2 * math.log2(y / epsilon))
        assert chebyshev_terms(y, epsilon) == math.ceil(
            math.sqrt(s * math.log2(4.0 * s / epsilon)))
