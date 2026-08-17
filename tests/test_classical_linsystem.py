"""Solving a system classically, and what the four quantum solvers make of it.

The claim being tested is mostly about the scaling convention, because that is
where a defensible-looking choice quietly breaks the analysis rather than the
arithmetic.  All four solvers state their amplification overhead as a success
probability ``||x||^2 / (4 kappa^2)`` against a bound ``1 / (4 kappa^2)``.  Those
two are a probability and a lower bound on it exactly when ``||x||`` lies
between one and the condition number, and that is what scaling the matrix to
unit spectral norm buys.  An unscaled matrix does not cost differently -- it
makes the statement false, and the library refuses it.  So the invariant is
asserted directly, on every matrix here.

The other thing worth asserting is that one log serves all four solvers, and
that when it does they come out in the order the comparison found: singular
value transformation cheapest, Chebyshev close behind, Fourier about an order of
magnitude dearer, HHL far away.  That ordering is a published conclusion, and
reproducing it from a matrix on disk -- rather than from figures typed in -- is
the point of generating logs at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.linsystem import solve, sparsity
from hybrid_benchmarking.instances import Matrix
from hybrid_benchmarking.instances.matrixmarket import read as read_matrix

CHOSEN = {"epsilon": 1e-3}


def matrix(entries, rows, columns=None, symmetric=False) -> Matrix:
    return Matrix(name="m", source="(hand built)", layout="matrix-market",
                  rows=rows, columns=columns or rows,
                  entries=tuple(entries), symmetric=symmetric)


#: A well-conditioned symmetric matrix: the second difference operator on four
#: points, whose condition number is known in closed form.
LAPLACIAN = matrix(
    [(i, i, 2.0) for i in range(4)]
    + [(i, i + 1, -1.0) for i in range(3)]
    + [(i + 1, i, -1.0) for i in range(3)],
    rows=4, symmetric=True,
)

#: Deliberately not symmetric, and with a row maximum different from its column
#: maximum: row 0 has three entries, column 0 has one.
LOPSIDED = matrix(
    [(0, 0, 4.0), (0, 1, 1.0), (0, 2, 1.0),
     (1, 1, 3.0), (2, 2, 2.0), (1, 2, 1.0)],
    rows=3,
)


class TestItReallySolves:
    def test_the_residual_is_at_the_level_of_rounding(self):
        run = solve(LAPLACIAN, Budget(60))
        assert run.status is Status.COMPLETE
        assert run.result["residual"] < 1e-10

    def test_the_condition_number_is_the_ratio_of_singular_values(self):
        run = solve(LAPLACIAN, Budget(60))
        dense = np.array([[2.0, -1, 0, 0], [-1, 2, -1, 0],
                          [0, -1, 2, -1], [0, 0, -1, 2]])
        assert run.records[0]["kappa"] == pytest.approx(np.linalg.cond(dense))

    def test_scaling_does_not_move_the_condition_number(self):
        # The whole point of scaling is that it changes what has to be true of
        # ||x|| without changing the quantity the cost is dominated by.
        plain = solve(LAPLACIAN, Budget(60)).records[0]["kappa"]
        doubled = solve(
            matrix([(r, c, 2 * v) for r, c, v in LAPLACIAN.entries], rows=4,
                   symmetric=True), Budget(60)).records[0]["kappa"]
        assert plain == pytest.approx(doubled)


class TestTheScalingConvention:
    @pytest.mark.parametrize("instance", [LAPLACIAN, LOPSIDED])
    def test_the_solution_norm_lies_between_one_and_the_condition_number(
            self, instance):
        # This is what makes the four amplification statements statements about
        # a probability rather than about a number above one.
        entry = solve(instance, Budget(60)).records[0]
        assert 1.0 <= entry["x_norm"] <= entry["kappa"] * (1 + 1e-9)

    @pytest.mark.parametrize("instance", [LAPLACIAN, LOPSIDED])
    def test_the_largest_entry_is_at_most_one_after_scaling(self, instance):
        assert solve(instance, Budget(60)).records[0]["A_max"] <= 1.0

    def test_a_matrix_arriving_scaled_differently_gives_the_same_cost(self):
        # Someone else's copy of the same operator, off by a factor: the
        # quantum cost cannot depend on that, and after scaling it does not.
        scaled = matrix([(r, c, 1000 * v) for r, c, v in LAPLACIAN.entries],
                        rows=4, symmetric=True)
        first = cost(generate(LAPLACIAN, budget=Budget(60)), CHOSEN)["total"]
        second = cost(generate(scaled, budget=Budget(60)), CHOSEN)["total"]
        assert first == pytest.approx(second)

    def test_the_convention_is_stated_on_the_answer(self):
        report = cost(generate(LAPLACIAN, budget=Budget(60)), CHOSEN)
        joined = " ".join(report["assumptions"])
        assert "unit spectral norm" in joined
        assert "uniform superposition" in joined


class TestSparsity:
    def test_it_is_the_larger_of_the_row_and_column_maxima(self):
        # Row 0 has three entries; no column has more than two.  A quantum
        # solver acts on the Hermitian dilation, whose rows are both.
        assert sparsity(LOPSIDED) == 3

    def test_a_symmetric_matrix_has_only_one_answer(self):
        assert sparsity(LAPLACIAN) == 3

    def test_the_reason_for_taking_the_larger_is_recorded(self):
        run = solve(LOPSIDED, Budget(60))
        assert any("Hermitian dilation" in note for note in run.assumptions)


class TestTheRightHandSide:
    def test_the_default_is_uniform_and_says_so(self):
        run = solve(LAPLACIAN, Budget(60))
        assert any("uniform superposition" in note for note in run.assumptions)

    def test_a_supplied_one_is_used_and_recorded(self):
        run = solve(LAPLACIAN, Budget(60), rhs=[1.0, 0.0, 0.0, 0.0])
        assert any("supplied alongside" in note for note in run.assumptions)
        assert run.records[0]["x_norm"] != solve(
            LAPLACIAN, Budget(60)).records[0]["x_norm"]

    def test_it_is_normalised_however_it_arrives(self):
        # The solver prepares a state, so only the direction of b matters.
        one = solve(LAPLACIAN, Budget(60), rhs=[1.0, 2.0, 3.0, 4.0])
        many = solve(LAPLACIAN, Budget(60), rhs=[100.0, 200.0, 300.0, 400.0])
        assert one.records[0]["x_norm"] == pytest.approx(
            many.records[0]["x_norm"])

    def test_a_zero_right_hand_side_is_refused(self):
        run = solve(LAPLACIAN, Budget(60), rhs=[0.0, 0.0, 0.0, 0.0])
        assert run.status is Status.FAILED

    def test_one_of_the_wrong_length_is_refused_by_name(self):
        run = solve(LAPLACIAN, Budget(60), rhs=[1.0, 2.0])
        assert run.status is Status.FAILED
        assert "2 entries" in run.reason


class TestTheFourSolversOnOneLog:
    def test_one_log_serves_all_four(self):
        # They differ in how they reach the matrix, not in what they need to
        # know about it, which is what makes this a comparison.
        instance = read_matrix("tests/fixtures/matrixmarket/spec-example.mtx")
        totals = {}
        for route in ("qsvt", "chebyshev", "fourier", "hhl"):
            generated = generate(instance, "linear-systems", route, Budget(60))
            totals[route] = cost(generated, CHOSEN)["total"]
        assert all(value > 0 for value in totals.values())

    def test_they_come_out_in_the_order_the_comparison_found(self):
        instance = read_matrix("tests/fixtures/matrixmarket/spec-example.mtx")

        def total(route):
            return cost(generate(instance, "linear-systems", route,
                                 Budget(60)), CHOSEN)["total"]

        assert total("qsvt") < total("chebyshev") < total("fourier") \
            < total("hhl")

    def test_hhl_is_the_one_that_scales_with_the_precision_not_its_log(self):
        instance = read_matrix("tests/fixtures/matrixmarket/spec-example.mtx")

        def total(route, epsilon):
            return cost(generate(instance, "linear-systems", route,
                                 Budget(60)), {"epsilon": epsilon})["total"]

        # A hundredfold tighter precision costs HHL far more than it costs a
        # solver whose dependence is logarithmic.
        assert (total("hhl", 1e-5) / total("hhl", 1e-3)
                > 10 * total("qsvt", 1e-5) / total("qsvt", 1e-3))


class TestItRefusesRatherThanGuessing:
    def test_a_singular_matrix_has_no_condition_number(self):
        run = solve(matrix([(0, 0, 1.0), (1, 1, 0.0)], rows=2), Budget(60))
        assert run.status is Status.FAILED
        assert "singular" in run.reason

    def test_a_rectangular_matrix_is_not_a_linear_system(self):
        run = solve(matrix([(0, 0, 1.0)], rows=2, columns=3), Budget(60))
        assert run.status is Status.FAILED
        assert "square" in run.reason

    def test_a_matrix_too_large_to_factorise_says_a_budget_will_not_help(self):
        huge = matrix([(0, 0, 1.0)], rows=20_000)
        run = solve(huge, Budget(60))
        assert run.status is Status.FAILED
        assert "longer budget will not help" in run.reason
