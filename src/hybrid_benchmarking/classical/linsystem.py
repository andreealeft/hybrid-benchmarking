"""Solving a linear system classically, to log what a quantum solver would face.

The four functional solvers are costed per system, from four numbers: the
condition number, the sparsity, the largest entry, and the norm of the solution.
Three of those are properties of the matrix and one requires actually solving,
which is what this does -- a dense factorisation for the solution and a singular
value decomposition for the condition number, both from the same numpy that
would do it for anyone else.

Two conventions here are decisions rather than readings, and both change the
number.

**The matrix is scaled to unit spectral norm.**  Every one of these solvers is
derived for a matrix a block encoding or a Hamiltonian simulation can reach,
which fixes ``||A|| <= 1``; and their amplification overheads are stated in terms
of a success probability ``||x||^2 / (4 kappa^2)`` against a bound
``1 / (4 kappa^2)``.  Those two are a probability and a lower bound on it exactly
when ``1 <= ||x|| <= kappa``, which is what scaling by the largest singular
value guarantees and what an unscaled matrix does not.  So an unscaled matrix
would not merely cost differently, it would make the amplification statement
false -- and the library would refuse it, which is the right behaviour and the
wrong error message.

**The right-hand side is the uniform superposition unless one is supplied.**  It
enters only through ``||x||``, and there is no neutral choice: a right-hand side
aligned with the smallest singular vector makes ``||x||`` as large as ``kappa``
and the amplification cheap, one aligned with the largest makes it 1 and the
amplification dearest.  The uniform state is the one a quantum computer prepares
for nothing -- which matters, because "preparing the right-hand side is not
counted" is already an assumption of all four lemmas -- and it is the same on
every machine.  A real right-hand side from a companion file is used when given,
and which was used is recorded.

The sparsity is the larger of the row and column maxima.  A quantum linear
solver acts on a Hermitian matrix; where the file is not symmetric that is the
dilation ``[[0, A], [A*, 0]]``, whose rows are A's rows and A's columns, and
whose condition number and solution norm are the same as A's.  Taking only the
row maximum would understate the oracle the cost is counted in.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..instances import Matrix
from .budget import Budget, Run, Status

IMPLEMENTATION = ("a dense LU factorisation for the solution and a singular "
                  "value decomposition for the condition number, both numpy's "
                  "-- an iterative solver would report a different condition "
                  "number and a different solution norm")

#: Beyond this a dense decomposition is not something to wait for at any budget.
#: A matrix this size is already 800 MB as a dense array of doubles, and the
#: decomposition is cubic in it, so failing here with a reason beats failing
#: later without one.
MAX_DIMENSION = 10_000

#: Singular values below this multiple of the largest are treated as zero, which
#: is numpy's own rank tolerance.  A matrix that trips it has no condition
#: number to report rather than a very large one.
_RANK_TOLERANCE = 1e-14


def densify(instance: Matrix) -> np.ndarray:
    matrix = np.zeros((instance.rows, instance.columns))
    for row, column, value in instance.entries:
        matrix[row, column] = value
    return matrix


def sparsity(instance: Matrix) -> int:
    """The larger of the row and column maxima, from the file's own pattern.

    Counted from the entries the file states rather than from the dense array,
    because a stated zero is a position the sparse-access oracle still has to
    answer for -- the file is describing its pattern, and that is the pattern.
    """
    per_row: Dict[int, int] = {}
    per_column: Dict[int, int] = {}
    for row, column, _ in instance.entries:
        per_row[row] = per_row.get(row, 0) + 1
        per_column[column] = per_column.get(column, 0) + 1
    return max(max(per_row.values(), default=0),
               max(per_column.values(), default=0))


def solve(instance: Matrix, budget: Budget,
          rhs: Optional[Sequence[float]] = None) -> Run:
    """Solve one system and record what a quantum solver would be facing."""
    if instance.rows != instance.columns:
        return _failed(budget, "the matrix is {} by {}; a linear system needs "
                               "a square one".format(instance.rows,
                                                     instance.columns))
    dimension = instance.rows
    if dimension < 2:
        return _failed(budget, "a system of one unknown has nothing to solve")
    if dimension > MAX_DIMENSION:
        return _failed(
            budget,
            "the matrix is {0} by {0}, and this tool factorises densely; a "
            "longer budget will not help".format(dimension),
        )
    if budget.spent:
        return _failed(budget, "the budget was gone before the solve began")

    matrix = densify(instance)
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest, smallest = float(singular[0]), float(singular[-1])
    if largest <= 0:
        return _failed(budget, "the matrix is entirely zero")
    if smallest <= _RANK_TOLERANCE * largest:
        return _failed(
            budget,
            "the matrix is singular to working precision, so it has no "
            "condition number and the system has no unique solution",
        )

    scaled = matrix / largest
    kappa = largest / smallest

    if rhs is None:
        right = np.full(dimension, 1.0 / np.sqrt(dimension))
        source = ("the uniform superposition, which is what a quantum computer "
                  "prepares for nothing")
    else:
        right = np.asarray(rhs, dtype=float)
        if right.shape != (dimension,):
            return _failed(budget, "the right-hand side has {} entries and the "
                                   "matrix has {} rows".format(right.size,
                                                               dimension))
        norm = float(np.linalg.norm(right))
        if norm <= 0:
            return _failed(budget, "the right-hand side is zero, so the "
                                   "solution is zero and there is nothing to "
                                   "prepare")
        right = right / norm
        source = "a right-hand side supplied alongside the matrix, normalised"

    solution = np.linalg.solve(scaled, right)
    density = sparsity(instance)

    record = {
        "kappa": kappa,
        "d": density,
        "x_norm": float(np.linalg.norm(solution)),
        "A_max": float(np.max(np.abs(scaled))),
    }
    return Run(
        implementation=IMPLEMENTATION,
        status=Status.COMPLETE,
        records=(record,),
        instance={"N": dimension},
        elapsed=budget.elapsed,
        budget=budget.seconds,
        result={"dimension": dimension, "condition_number": kappa,
                "residual": float(np.linalg.norm(scaled @ solution - right))},
        assumptions=(
            "the matrix is scaled to unit spectral norm, which is the regime "
            "all four solvers are derived in and what makes their success "
            "probability a probability",
            "the right-hand side is {}".format(source),
            "d is the larger of the row and column maxima, being the sparsity "
            "of the Hermitian dilation a quantum solver actually acts on",
            "the condition number is the exact ratio of largest to smallest "
            "singular value, not an estimate as an iterative solver would use",
        ),
    )


def _failed(budget: Budget, reason: str) -> Run:
    return Run(implementation=IMPLEMENTATION, status=Status.FAILED,
               budget=budget.seconds, elapsed=budget.elapsed, reason=reason)
