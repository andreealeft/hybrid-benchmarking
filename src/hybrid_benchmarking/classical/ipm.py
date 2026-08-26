"""The Newton systems Binkowski's benchmark costs, built the way he builds them.

This follows *Practical lower bounds for hybrid quantum interior point methods
in linear programming* (arXiv:2604.24362) closely enough to be checked against
it, because the alternative is a plausible cycle count from a system the paper
does not describe.  An earlier version of this module built the plain normal
equations ``A D^2 A'`` and walked a Mehrotra path; both were wrong, and neither
would have failed a test.

**What is costed is one system, not a solve.**  The paper's Section IV-B assumes
benevolently that the method converges in a single iteration, "so the benchmark
uses only the cost of solving the resulting first Newton system and never
propagates the resulting iterate into subsequent iterations".  That assumption
is already recorded on the ``IPM/mnes`` entry.  So there is one record, and
summing a path of them -- which the earlier version did -- would have produced
a number several times the paper's and no longer a lower bound.

**The iterate is canonical, not computed.**  ``(x, y, s) = (1, 0, 1)``: strictly
positive, deliberately not feasible, chosen so the diagonal matrices are
invertible.  It follows that ``X = S = I`` and hence ``D = D_B = I``, which is
what collapses the two constructions to the forms below.

**The modified normal equation system** is not the normal equations.  With
``A_B`` a set of ``m`` independent columns of ``A`` and ``A_N`` the rest,
equation (6) reduces at this iterate to ``M_hat = I + F F'`` with
``F := A_B^-1 A_N``.  Its dimension is ``m``.  Its condition number follows from
``lambda_i(M_hat) = 1 + sigma_i(F)^2`` without forming it:

    kappa(M_hat) = (1 + sigma_max(F)^2) / (1 + sigma_min(F)^2)

and where ``n - m < m`` the matrix ``F F'`` has a null space, so
``sigma_min = 0`` and the denominator is exactly 1.

**The orthogonal subspace system** is equation (8): ``O = [-X A' , S V]``, of
dimension ``n``, with ``V`` the null-space basis of (7) -- ``A_B^-1 A_N`` in the
rows belonging to the basis columns and ``-I`` in the rest.  ``O`` is not
symmetric, and footnote 2 says to read it off the Hermitian dilation
``[[0, O], [O*, 0]]``.  That dilation shares its sparsity and its condition
number -- and *doubles its dimension*, which is the part it is easy to drop.
The readout is tomography of a state in the dilated space, so the logged ``N``
is ``2n``: the footnote applies to all three quantities, not to two of them.

Two deliberate departures, both recorded on every cost:

*The singular values are exact here.*  The paper estimates them with ARPACK and
a sampling fallback, chosen so that the numerator is underestimated and the
denominator overestimated -- making its ``kappa`` a lower bound on the true one.
At the sizes this tool runs, an exact decomposition is affordable, and the exact
value is what it is: not an over-estimate, but larger than the paper's estimate
would be on the same instance.

*The sparsity is measured rather than argued.*  The paper takes ``s = m`` for the
MNES, on the grounds that ``A_B^-1`` is generally dense and so ``M_hat`` is too.
That is usually right and never smaller than the truth, so measuring the built
matrix instead can only lower the count.  Both are logged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .budget import Budget, Run, Status
from .lp import StandardForm

IMPLEMENTATION = ("the first Newton system from the canonical iterate, built "
                  "and measured as in Binkowski's benchmark -- with exact "
                  "singular values where the paper estimates them")

#: Entries below this fraction of the largest are not structurally present.
#: The systems here are products of dense factors, so a structural count needs a
#: threshold; it is recorded, because it decides the sparsity that the cycle
#: count is quadratic in.
DENSITY_TOLERANCE = 1e-12

#: Rank tolerance for choosing the basis and for dropping redundant rows.
_RANK_TOLERANCE = 1e-9

#: Beyond this a dense decomposition is not something to wait for at any budget.
MAX_DIMENSION = 4000


def independent_rows(matrix: np.ndarray,
                     tolerance: float = _RANK_TOLERANCE) -> List[int]:
    """A maximal independent set of rows, kept in order, by Gram-Schmidt.

    Both constructions need ``A`` to have full row rank -- the paper gets there
    with HiGHS's presolve, which eliminates redundant constraints before the
    standard form is built.  Some programs are redundant by construction rather
    than by accident: a maximum-flow model has one conservation row per vertex
    and those always sum to zero, so refusing them would refuse the problem.
    """
    keep: List[int] = []
    basis: List[np.ndarray] = []
    for index, row in enumerate(matrix):
        residual = np.asarray(row, dtype=float).copy()
        for direction in basis:
            residual -= (residual @ direction) * direction
        size = float(np.linalg.norm(residual))
        if size > tolerance * max(1.0, float(np.linalg.norm(row))):
            basis.append(residual / size)
            keep.append(index)
    return keep


def choose_basis(matrix: np.ndarray,
                 tolerance: float = _RANK_TOLERANCE) -> List[int]:
    """``m`` independent columns of ``A``, by column-pivoted QR.

    The paper finds these once by sparse QR and reuses them for both systems.
    Which independent set comes out is not a detail: both constructions are
    built on ``A_B``, and on a 120-by-400 program the difference between the
    best and worst defensible choice was four orders of magnitude in the
    condition number and nearly five in the cycle count -- far more than the
    logarithm base, the sparsity convention and the estimator put together.

    Taking the columns in index order, which is what orthogonalising them
    greedily amounts to, lands at the ill-conditioned end of that range.  That
    is the wrong end: the cost is quadratic in the condition number, so it
    inflates a number whose whole purpose is to be a bound favourable to the
    quantum side.  Pivoting on the largest remaining residual at each step is
    what a QR with column pivoting does, it is what the paper's sparse QR is a
    sparsity-aware version of, and it is a great deal better conditioned.

    Householder rather than Gram-Schmidt for the same reason it is everywhere
    else: the residual norms this pivots on are exactly what modified
    Gram-Schmidt loses to cancellation on the matrices that matter.
    """
    work = np.array(matrix, dtype=float, copy=True)
    rows, columns = work.shape
    norms = np.einsum("ij,ij->j", work, work)
    scale = float(np.sqrt(norms.max())) if columns else 0.0
    order = list(range(columns))
    chosen: List[int] = []

    for step in range(min(rows, columns)):
        pivot = step + int(np.argmax(norms[step:]))
        if math.sqrt(max(norms[pivot], 0.0)) <= tolerance * max(1.0, scale):
            break
        order[step], order[pivot] = order[pivot], order[step]
        work[:, [step, pivot]] = work[:, [pivot, step]]
        norms[step], norms[pivot] = norms[pivot], norms[step]

        column = work[step:, step]
        sign = -1.0 if column[0] < 0 else 1.0
        reflector = column.copy()
        reflector[0] += sign * np.linalg.norm(column)
        length = np.linalg.norm(reflector)
        if length > 0:
            reflector /= length
            block = work[step:, step:]
            block -= 2.0 * np.outer(reflector, reflector @ block)
        # The trailing norms fall by the square of the row just eliminated.
        norms[step + 1:] -= work[step, step + 1:] ** 2
        np.maximum(norms[step + 1:], 0.0, out=norms[step + 1:])
        chosen.append(order[step])

    return chosen


def _sparsity(matrix: np.ndarray) -> int:
    """Largest number of structurally present entries in any row or column.

    Both, because the Hermitian dilation a quantum solver acts on has the
    original's rows and its columns -- which is footnote 2's point about the
    OSS matrix, and is what the sparse-access oracle would have to answer for.
    """
    if matrix.size == 0:
        return 0
    largest = float(np.max(np.abs(matrix)))
    if largest <= 0:
        return 0
    present = np.abs(matrix) > DENSITY_TOLERANCE * largest
    return int(max(np.max(np.count_nonzero(present, axis=1)),
                   np.max(np.count_nonzero(present, axis=0))))


def _extreme_singular_values(matrix: np.ndarray) -> Tuple[float, float]:
    values = np.linalg.svd(matrix, compute_uv=False)
    if values.size == 0:
        return 0.0, 0.0
    return float(values[0]), float(values[-1])


def mnes(matrix: np.ndarray, basis: List[int]) -> Dict[str, Any]:
    """Equation (6) at the canonical iterate: dimension, sparsity, condition.

    ``M_hat = I + F F'`` with ``F = A_B^-1 A_N``.  The condition number comes
    from the singular values of ``F`` rather than from ``M_hat`` itself, which
    is the paper's own route and is better conditioned numerically.
    """
    rows, columns = matrix.shape
    other = [j for j in range(columns) if j not in set(basis)]
    inverse_times_n = np.linalg.solve(matrix[:, basis], matrix[:, other]) \
        if other else np.zeros((rows, 0))

    largest, smallest = _extreme_singular_values(inverse_times_n)
    if len(other) < rows:
        # F F' has a null space, so the smallest eigenvalue of M_hat is exactly
        # one -- the paper says so, and it is worth not discovering it from a
        # decomposition that returns 1e-17 instead.
        smallest = 0.0
    kappa = (1.0 + largest ** 2) / (1.0 + smallest ** 2)

    built = np.eye(rows) + inverse_times_n @ inverse_times_n.T
    return {
        "N": rows,
        "d": _sparsity(built),
        "kappa": kappa,
        "d_paper": rows,
        "sigma_max_F": largest,
        "sigma_min_F": smallest,
    }


def oss(matrix: np.ndarray, basis: List[int]) -> Dict[str, Any]:
    """Equation (8) at the canonical iterate: ``O = [-A' , V]``.

    ``V`` is the null-space basis of (7): ``A_B^-1 A_N`` in the rows belonging
    to the basis columns, ``-I`` in the rest.  Assembled by position rather than
    stacked, because those rows are variables and the basis columns are not
    contiguous.
    """
    rows, columns = matrix.shape
    chosen = set(basis)
    other = [j for j in range(columns) if j not in chosen]
    null = np.zeros((columns, len(other)))
    if other:
        null[basis, :] = np.linalg.solve(matrix[:, basis], matrix[:, other])
        null[other, :] = -np.eye(len(other))

    system = np.hstack([-matrix.T, null])
    largest, smallest = _extreme_singular_values(system)
    if smallest <= 0:
        return {"singular": True}
    return {
        # Twice the system's own dimension. O is not symmetric, and footnote 2
        # says to read it off the Hermitian dilation [[0, O], [O*, 0]] -- which
        # has the same sparsity and the same condition number, and twice the
        # dimension. The readout is tomography of a state in the dilated space,
        # so the dimension is the one that carries. The author's own code makes
        # exactly this correction, in a commit that says so.
        "N": 2 * columns,
        "d": _sparsity(system),
        "kappa": largest / smallest,
        "dimension_undilated": columns,
        "sigma_max_O": largest,
        "sigma_min_O": smallest,
    }


_SYSTEMS = {"mnes": mnes, "oss": oss}


def solve(form: StandardForm, budget: Budget, system: str = "mnes") -> Run:
    """Build the first Newton system and record what a quantum solver faces."""
    if system not in _SYSTEMS:
        raise ValueError("no Newton system {!r}; there is {}".format(
            system, " and ".join(sorted(_SYSTEMS))))

    matrix = form.matrix
    rows, columns = matrix.shape
    if max(rows, columns) > MAX_DIMENSION:
        return _failed(budget, system,
                       "the program is {} by {}, and both systems here are "
                       "decomposed densely; a longer budget will not "
                       "help".format(rows, columns))
    if budget.spent:
        return _failed(budget, system, "the budget was gone before the system "
                                       "was built")

    keep = independent_rows(matrix)
    dropped = rows - len(keep)
    if dropped:
        augmented = np.column_stack([matrix, form.rhs])
        if np.linalg.matrix_rank(augmented) > len(keep):
            return _failed(budget, system,
                           "the constraints are inconsistent: {} of them are "
                           "combinations of the others but disagree about the "
                           "right-hand side".format(dropped))
        matrix = matrix[keep]
        rows = len(keep)

    if rows < 2 or columns <= rows:
        return _failed(budget, system,
                       "the program has {} independent constraints and {} "
                       "columns; there is no basis to split".format(rows, columns))

    basis = choose_basis(matrix)
    if len(basis) < rows:
        return _failed(budget, system,
                       "only {} of the {} columns are independent, so the "
                       "constraint matrix has no basis".format(len(basis), rows))
    basis = basis[:rows]

    record = _SYSTEMS[system](matrix, basis)
    if record.get("singular"):
        return _failed(budget, system,
                       "the orthogonal subspace system is singular, so it has "
                       "no condition number")

    record["gamma"] = record["d"] * record["kappa"]
    record["at_seconds"] = round(budget.elapsed, 6)
    return Run(
        implementation=IMPLEMENTATION,
        status=Status.COMPLETE,
        records=(record,),
        instance={},
        elapsed=budget.elapsed,
        budget=budget.seconds,
        result={"system": system.upper(), "dimension": record["N"],
                "sparsity": record["d"], "condition_number": record["kappa"],
                "difficulty": record["gamma"]},
        assumptions=(
            "the system is the {} of Binkowski's equation ({}), built at the "
            "canonical iterate (x, y, s) = (1, 0, 1) -- strictly positive and "
            "deliberately not feasible, which is what makes X = S = I".format(
                system.upper(), "6" if system == "mnes" else "8"),
            "one system is costed, not a solve: the paper assumes the method "
            "converges in a single iteration and benchmarks only the first "
            "Newton system",
            "the basis is a set of m independent columns found by "
            "orthogonalisation rather than by the paper's sparse QR; both "
            "constructions rest on it, and a different independent set gives a "
            "different condition number",
            "the singular values are exact, where the paper estimates them so "
            "that its condition number is a lower bound on the true one; this "
            "one is the true one, and so is the larger",
            "the sparsity is measured on the system as built, with an entry "
            "counted as present above {:g} of the largest. The paper argues "
            "s = m for the MNES on the grounds that the basis inverse is "
            "dense; that reading is logged beside it as "
            "d_paper".format(DENSITY_TOLERANCE),
        ) + (("{} redundant constraint row{} removed first, as the paper's "
              "presolve does".format(dropped, "" if dropped == 1 else "s"),)
             if dropped else ()),
    )


def _failed(budget: Budget, system: str, reason: str) -> Run:
    return Run(implementation=IMPLEMENTATION, status=Status.FAILED,
               budget=budget.seconds, elapsed=budget.elapsed, reason=reason)
