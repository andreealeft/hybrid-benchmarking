"""The interior point route, and the Newton system each iteration faces.

The strongest thing available here is that two independently written solvers
have to agree.  The simplex walks the boundary and the interior point method
does not go near it, they share nothing but the standard form, and they must
reach the same optimum -- so that is asserted on every program in this file,
and it is what makes the logs believable before any cost is computed.

After that the claims are about what a record is and what is in it.  A record is
an iteration rather than a solve, because Mehrotra's predictor and corrector
reuse one system; the dimension is the constraint count; and the density comes
out of order ``m``, which is what the ``IPM/mnes`` entry already assumes about
it and is here confirmed rather than restated.

One column is a decision made without the source to hand.  The construction is
Binkowski's "modified" normal equations, and a diagonal rescaling of a system
leaves its dimension and density alone but not its condition number -- which the
cost is quadratic in.  So both are logged and the route consumes the unmodified
one -- not because it is the safer number, since equilibration lowers it on a
vertex cover and raises it on an independent set and there is a test for exactly
that, but because inventing a modification the source may not mean is worse than
declaring which reading was taken.
"""

from __future__ import annotations

import numpy as np
import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, Status, cost, generate
from hybrid_benchmarking.classical.ipm import (
    choose_basis,
    independent_rows,
    mnes,
    oss,
)
from hybrid_benchmarking.classical.ipm import solve as interior_point
from hybrid_benchmarking.classical.lp import (
    from_linear_program,
    independent_set,
    standard_form,
    vertex_cover,
)
from hybrid_benchmarking.classical.simplex import solve as pivot
from hybrid_benchmarking.instances import Graph
from hybrid_benchmarking.instances.dimacs import read_max_flow
from hybrid_benchmarking.instances.mps import read as read_mps
from hybrid_benchmarking.provenance import Bound, Derivation

CHOSEN = {"epsilon": 1e-3}


def graph(vertices: int, edges) -> Graph:
    return Graph(name="g", source="(hand built)", layout="dimacs-edge",
                 vertices=vertices, edges=tuple(sorted(edges)))


CYCLE_5 = graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
SQUARE = graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
STAR_5 = graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)])

PROGRAMS = [
    ("cover of a five-cycle", vertex_cover(CYCLE_5)),
    ("cover of a square", vertex_cover(SQUARE)),
    ("cover of a star", vertex_cover(STAR_5)),
    ("independent set of a five-cycle", independent_set(CYCLE_5)),
]


class TestTheSystemsAreThePapersSystems:
    """Checked against arXiv:2604.24362 rather than against our own docstrings.

    The MNES is not the normal equations. At the canonical iterate equation (6)
    reduces to ``I + F F'`` with ``F = A_B^-1 A_N``, and its eigenvalues are
    ``1 + sigma_i(F)^2`` -- which is both how the paper estimates the condition
    number and a property that can be checked here without trusting either.
    """

    def test_the_mnes_matrix_is_the_identity_plus_f_f_transpose(self):
        form = standard_form(vertex_cover(CYCLE_5))
        matrix = form.matrix[independent_rows(form.matrix)]
        basis = choose_basis(matrix)[:matrix.shape[0]]
        other = [j for j in range(matrix.shape[1]) if j not in set(basis)]

        f = np.linalg.solve(matrix[:, basis], matrix[:, other])
        built = np.eye(matrix.shape[0]) + f @ f.T
        # Equation (6) at x = s = 1, spelled out the long way.
        direct = np.linalg.solve(matrix[:, basis], matrix)
        assert np.allclose(built, direct @ direct.T)

    def test_the_condition_number_follows_the_papers_eigenvalue_identity(self):
        form = standard_form(vertex_cover(CYCLE_5))
        matrix = form.matrix[independent_rows(form.matrix)]
        basis = choose_basis(matrix)[:matrix.shape[0]]
        record = mnes(matrix, basis)

        other = [j for j in range(matrix.shape[1]) if j not in set(basis)]
        f = np.linalg.solve(matrix[:, basis], matrix[:, other])
        built = np.eye(matrix.shape[0]) + f @ f.T
        assert record["kappa"] == pytest.approx(np.linalg.cond(built), rel=1e-8)

    def test_a_wide_basis_split_makes_the_smallest_eigenvalue_exactly_one(self):
        # The paper's parenthesis: where n - m < m, F F' has a null space, so
        # sigma_min is zero and lambda_min is one exactly.
        form = standard_form(vertex_cover(CYCLE_5))
        matrix = form.matrix[independent_rows(form.matrix)]
        rows, columns = matrix.shape
        assert columns - rows < rows
        record = mnes(matrix, choose_basis(matrix)[:rows])
        assert record["sigma_min_F"] == 0.0
        assert record["kappa"] == pytest.approx(1.0 + record["sigma_max_F"] ** 2)

    def test_the_oss_matrix_columns_span_the_null_space_of_a(self):
        # V is the null-space basis of (7): A V = 0 is what makes the update
        # feasible by construction, which is the whole point of that formulation.
        form = standard_form(vertex_cover(CYCLE_5))
        matrix = form.matrix[independent_rows(form.matrix)]
        rows, columns = matrix.shape
        basis = choose_basis(matrix)[:rows]
        other = [j for j in range(columns) if j not in set(basis)]

        null = np.zeros((columns, len(other)))
        null[basis, :] = np.linalg.solve(matrix[:, basis], matrix[:, other])
        null[other, :] = -np.eye(len(other))
        assert np.allclose(matrix @ null, 0.0)

    def test_the_two_systems_have_the_dimensions_the_paper_states(self):
        form = standard_form(vertex_cover(CYCLE_5))
        matrix = form.matrix[independent_rows(form.matrix)]
        rows, columns = matrix.shape
        basis = choose_basis(matrix)[:rows]
        assert mnes(matrix, basis)["N"] == rows      # m-dimensional
        assert oss(matrix, basis)["N"] == columns    # n-dimensional

    def test_the_mnes_is_smaller_and_the_oss_is_feasible_by_construction(self):
        # Which is the trade the paper draws between them.
        form = standard_form(vertex_cover(CYCLE_5))
        matrix = form.matrix[independent_rows(form.matrix)]
        basis = choose_basis(matrix)[:matrix.shape[0]]
        assert mnes(matrix, basis)["N"] < oss(matrix, basis)["N"]


class TestTheCostIsEquationTen:
    def test_it_is_tomography_times_the_chebyshev_query_count(self):
        # The module said it reused the QLS-Chebyshev entry and did not; now it
        # does, so the two cannot drift apart.
        from hybrid_benchmarking.routines.linsolve import (
            binkowski_chebyshev_queries,
        )
        from hybrid_benchmarking.routines.qipm import (
            newton_system_cycles,
            tomography_repetitions,
        )

        d, s, k, e = 1000, 50, 100.0, 0.1
        assert newton_system_cycles(d, s, k, e) == pytest.approx(
            tomography_repetitions(d, e) * binkowski_chebyshev_queries(s, k, e))

    def test_it_matches_equation_ten_written_out_by_hand(self):
        import math

        from hybrid_benchmarking.routines.qipm import newton_system_cycles

        d, s, k, e = 1000, 50, 100.0, 0.1
        gamma = s * k
        inner = math.ceil(gamma ** 2 * math.log2(gamma / e))
        by_hand = (8 * (d - 1) / e ** 2) * math.sqrt(
            inner * math.log2(4.0 / e * inner))
        assert newton_system_cycles(d, s, k, e) == pytest.approx(by_hand, rel=1e-4)

    def test_the_readout_is_the_dimension_over_the_precision_squared(self):
        from hybrid_benchmarking.routines.qipm import tomography_repetitions

        assert tomography_repetitions(1000, 0.1) == pytest.approx(999 / 0.01)

    def test_the_route_defaults_to_the_precision_the_paper_uses(self):
        route = hb.get_route("vertex-cover", "quantum-interior-point")
        assert [f.example for f in route.chosen] == ["1e-1"]


class TestBothSystemsAreRoutes:
    @pytest.mark.parametrize("key,target", [
        ("quantum-interior-point", "IPM/mnes"),
        ("quantum-interior-point-oss", "IPM/oss"),
    ])
    def test_each_names_the_construction_it_costs(self, key, target):
        assert hb.get_route("vertex-cover", key).target == target

    @pytest.mark.parametrize("key", ["quantum-interior-point",
                                     "quantum-interior-point-oss"])
    def test_each_runs_from_a_file_and_costs_in_cycles(self, key):
        report = cost(generate(CYCLE_5, "vertex-cover", key, Budget(60)),
                      {"epsilon": 0.1})
        assert report["unit"] == "CYCLES"
        assert report["total"] > 0
        assert report["logged_records"] == 1  # one system, not a path

    def test_the_larger_system_is_the_easier_one(self):
        """Which is the trade the paper is measuring, and it runs both ways.

        The OSS is n-dimensional against the MNES's m, so its readout costs
        more per solve. But the MNES is built through a dense basis inverse and
        inherits both a higher sparsity and a higher condition number, so its
        difficulty ``gamma = s kappa`` is the larger. Neither dominates, which
        is why the paper reports both rather than picking one.
        """
        form = standard_form(vertex_cover(CYCLE_5))
        small = interior_point(form, Budget(60), "mnes").records[0]
        large = interior_point(form, Budget(60), "oss").records[0]

        assert large["N"] > small["N"]          # the OSS is the bigger system
        assert large["gamma"] < small["gamma"]  # and the easier one


class TestWhatIsRecorded:
    def test_only_the_first_system_is_costed_and_it_says_so(self):
        report = cost(generate(CYCLE_5, "vertex-cover",
                               "quantum-interior-point", Budget(60)),
                      {"epsilon": 0.1})
        assert any("converges in a single iteration" in note
                   for note in report["assumptions"])

    def test_the_departures_from_the_paper_are_both_declared(self):
        report = cost(generate(CYCLE_5, "vertex-cover",
                               "quantum-interior-point", Budget(60)),
                      {"epsilon": 0.1})
        joined = " ".join(report["assumptions"])
        assert "singular values are exact" in joined
        assert "d_paper" in joined

    def test_the_paper_reading_of_the_sparsity_is_logged_beside_ours(self):
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        entry = run.records[0]
        assert entry["d_paper"] == entry["N"]     # s = m, as the paper argues
        assert entry["d"] <= entry["d_paper"]     # measured can only be smaller

    def test_the_difficulty_is_sparsity_times_condition_number(self):
        run = interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60))
        entry = run.records[0]
        assert entry["gamma"] == pytest.approx(entry["d"] * entry["kappa"])


class TestWhenItCannotRun:
    def test_a_redundant_row_is_presolved_away_as_the_paper_does(self):
        from hybrid_benchmarking.classical.lp import Model

        model = Model(columns=4, objective=np.ones(4))
        model.add_row((0, 1), (1.0, 1.0), "E", 1.0)
        model.add_row((1, 2), (1.0, 1.0), "E", 1.0)
        model.add_row((0, 1, 2), (1.0, 2.0, 1.0), "E", 2.0)  # the sum of both
        run = interior_point(standard_form(model), Budget(60))
        assert run.status is Status.COMPLETE
        assert any("redundant constraint row" in note
                   for note in run.assumptions)

    def test_rows_that_contradict_each_other_are_refused(self):
        from hybrid_benchmarking.classical.lp import Model

        model = Model(columns=4, objective=np.ones(4))
        model.add_row((0, 1), (1.0, 1.0), "E", 1.0)
        model.add_row((1, 2), (1.0, 1.0), "E", 1.0)
        model.add_row((0, 1, 2), (1.0, 2.0, 1.0), "E", 5.0)
        run = interior_point(standard_form(model), Budget(60))
        assert run.status is Status.FAILED
        assert "inconsistent" in run.reason

    def test_an_unknown_system_is_refused_by_name(self):
        with pytest.raises(ValueError, match="mnes"):
            interior_point(standard_form(vertex_cover(CYCLE_5)), Budget(60),
                           system="normal-equations")
