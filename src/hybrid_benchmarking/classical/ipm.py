"""A primal-dual interior point method, instrumented per Newton system.

The interior point route costs one Newton system per iteration, from three
numbers: its dimension, its sparsity and its condition number.  Getting them
means running the method, because the condition number of the system is the
whole story of that analysis -- it is what the solver's cost is quadratic in,
and it degrades as the iterates approach the boundary, which is a property of
the path and not of the instance.

The method here is Mehrotra's predictor-corrector, with the standard starting
point and the standard two solves against one system per iteration.  That last
detail is why a record is an iteration rather than a solve: the predictor and
the corrector reuse the same matrix, so an iteration faces one system, which is
what the route's ``per_record`` means.

**The system logged is the normal-equation system** ``M = A D A'`` with
``D = diag(x / s)``, in the dual variables, of dimension ``m``.  That is the
dimension and the density the ``IPM/mnes`` entry describes, and its own recorded
assumption -- that the system inherits a density of order ``m`` rather than the
constraint matrix's -- is what these runs bear out.

**Where the modification in "modified normal equations" would go, this does
nothing**, and that is worth saying out loud rather than leaving to be
discovered.  The construction is Binkowski's, and it is not reimplemented here
from a one-line statement of it -- only the three quantities it needs are.
Dimension and sparsity are the same under any diagonal rescaling of the system.
The condition number is not, and the cost is quadratic in it.

So a diagonal equilibration of the same system -- one candidate reading of what
"modified" does, and the usual one -- is logged beside it under
``kappa_equilibrated``, and the column the route consumes carries the
**unmodified** one.  Equilibration is not reliably an improvement: on the graph
relaxations here it lowers the condition number by around a third for a vertex
cover and raises it slightly for an independent set, which is what the classical
bound on Jacobi scaling would lead one to expect.  So this is not a case of
taking the safer of two numbers -- there is no safer one.  It is a case of not
inventing a modification the source may not mean, logging what the alternative
reading would give, and saying plainly that this one quantity is not established
as favourable to the quantum side in either direction.  If the paper's rescaling
does turn out to apply, the correction is a column swap rather than a rerun.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .budget import Budget, Run, Status
from .lp import StandardForm

IMPLEMENTATION = ("Mehrotra's predictor-corrector interior point method, this "
                  "library's own numpy implementation, logging the "
                  "normal-equation system of each iteration")

#: Duality gap, relative to the starting one, at which the method has converged.
_GAP = 1e-8

#: Fraction of the distance to the boundary a step is allowed to take.  Every
#: interior point implementation has one and they do not agree; it changes how
#: fast the iterates approach the boundary and so changes every condition number
#: logged after the first.
_FRACTION = 0.99

#: Iterations after which a run is declared not to be converging.  Reached only
#: on instances a longer budget would not rescue.
_MAX_ITERATIONS = 200

#: A step this short is not progress.  Degenerate programs -- which the graph
#: relaxations routinely are -- stall here rather than grinding through two
#: hundred iterations that all describe the same point.
_STALLED = 1e-12


def independent_rows(matrix: np.ndarray, tolerance: float = 1e-9) -> List[int]:
    """A maximal independent set of rows, kept in order, by Gram-Schmidt.

    The normal equations are singular the moment two constraints say the same
    thing, and some programs are redundant by construction rather than by
    accident: a maximum-flow model has one conservation row per vertex, and
    those always sum to zero, because every arc leaves one vertex and enters
    another.  Refusing such a program would refuse the whole problem.

    So the redundancy is removed, which is what any interior point code's
    presolve does.  It changes ``m``, and therefore changes the dimension of
    every Newton system logged afterwards -- the reduced one is what gets
    logged, being the system that was actually solved.
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


def _starting_point(matrix: np.ndarray, rhs: np.ndarray, objective: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mehrotra's starting point: the least-squares solution, pushed inside.

    A bad start is not merely slow here -- it puts the first few iterates in a
    part of the path nobody would visit, and those iterates are records.
    """
    gram = matrix @ matrix.T
    dual = np.linalg.solve(gram, rhs)
    primal = matrix.T @ dual
    multipliers = np.linalg.solve(gram, matrix @ objective)
    slack = objective - matrix.T @ multipliers

    shift_x = max(-1.5 * float(np.min(primal)), 0.0)
    shift_s = max(-1.5 * float(np.min(slack)), 0.0)
    primal = primal + shift_x
    slack = slack + shift_s

    product = float(primal @ slack)
    if product <= 0:
        primal = np.maximum(primal, 1.0)
        slack = np.maximum(slack, 1.0)
        return primal, multipliers, slack
    primal = primal + 0.5 * product / float(np.sum(slack))
    slack = slack + 0.5 * product / float(np.sum(primal))
    return primal, multipliers, slack


def _step(direction: np.ndarray, value: np.ndarray) -> float:
    """How far along a direction the iterate stays strictly positive."""
    falling = direction < 0
    if not falling.any():
        return 1.0
    return float(min(1.0, _FRACTION * np.min(-value[falling] / direction[falling])))


def _pattern_sparsity(matrix: np.ndarray) -> int:
    """Most non-zeros in any row of ``A A'``, from the pattern rather than the values.

    Counted structurally: ``A D A'`` has the pattern of ``A A'`` whatever the
    diagonal holds, and counting numerically would need a tolerance whose choice
    would then be a logged quantity in its own right.
    """
    # Counted in a width that cannot wrap. An int8 product accumulates in
    # int8, so two rows sharing a multiple of 256 columns would come out as
    # structurally disjoint -- and d enters the cycle count as gamma = d kappa
    # with gamma squared inside it, so the understatement is not small.
    occupied = (matrix != 0).astype(np.int64)
    return int(np.max(np.count_nonzero(occupied @ occupied.T, axis=1)))


def _ratio(system: np.ndarray) -> float:
    """Largest singular value over smallest, or infinity if there is no answer.

    A decomposition that does not converge is itself a statement about the
    system -- it has gone past what double precision can describe -- so it comes
    back as infinity and ends the run, rather than as an exception from three
    frames down.
    """
    try:
        singular = np.linalg.svd(system, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf")
    if not np.all(np.isfinite(singular)) or singular[-1] <= 0:
        return float("inf")
    return float(singular[0] / singular[-1])


def _condition(system: np.ndarray) -> Tuple[float, float]:
    """The system's condition number, and what equilibrating it would give."""
    plain = _ratio(system)
    diagonal = np.abs(np.diag(system))
    if not np.all(np.isfinite(diagonal)) or not np.all(diagonal > 0):
        return plain, plain
    scale = 1.0 / np.sqrt(diagonal)
    return plain, _ratio(system * scale[:, None] * scale[None, :])


def solve(form: StandardForm, budget: Budget) -> Run:
    """Run the method, logging one Newton system per iteration."""
    matrix, rhs, objective = form.matrix, form.rhs, form.objective
    rows, columns = matrix.shape

    # Redundant rows make the normal equations singular whatever the iterates
    # do, and some programs are redundant by construction rather than by
    # accident, so they are presolved away rather than refused.
    keep = independent_rows(matrix)
    dropped = rows - len(keep)
    if dropped:
        augmented = np.column_stack([matrix, rhs])
        if np.linalg.matrix_rank(augmented) > len(keep):
            return _failed(
                budget,
                "the constraints are inconsistent: {} of them are linear "
                "combinations of the others but disagree with them about the "
                "right-hand side, so the program has no feasible point"
                .format(dropped),
            )
        matrix, rhs = matrix[keep], rhs[keep]
        rows = len(keep)

    if rows < 2:
        return _failed(budget, "a Newton system of one unknown is vacuous; the "
                               "program has {} independent constraint".format(rows))

    primal, dual, slack = _starting_point(matrix, rhs, objective)
    records: List[Dict[str, Any]] = []
    status, reason = Status.COMPLETE, ""
    gap = float(primal @ slack) / columns
    # Everything below is measured against the size of the data, not against
    # the first iterate: a program stated in different units is the same
    # program, and should stop at the same place.
    primal_scale = 1.0 + float(np.linalg.norm(rhs))
    dual_scale = 1.0 + float(np.linalg.norm(objective))

    for _ in range(_MAX_ITERATIONS):
        if budget.spent:
            status = Status.TRUNCATED if records else Status.FAILED
            reason = "the budget ran out with the duality gap at {:.3g}".format(gap)
            break

        if not (np.all(np.isfinite(primal)) and np.all(np.isfinite(dual))
                and np.all(np.isfinite(slack))):
            status = Status.TRUNCATED if records else Status.FAILED
            reason = ("the iterates left the range of double precision at "
                      "iteration {}, which is a breakdown of this "
                      "implementation".format(len(records)))
            break

        gap = float(primal @ slack) / columns
        residual_p = rhs - matrix @ primal
        residual_d = objective - matrix.T @ dual - slack
        if (gap / (1.0 + abs(float(objective @ primal))) < _GAP
                and float(np.linalg.norm(residual_p)) / primal_scale < _GAP
                and float(np.linalg.norm(residual_d)) / dual_scale < _GAP):
            break

        scaling = primal / slack
        system = (matrix * scaling) @ matrix.T
        plain, equilibrated = _condition(system)
        if not np.isfinite(plain):
            # The system has gone past what double precision describes.  The
            # iteration would still have a cost, but not one this can report.
            status = Status.TRUNCATED if records else Status.FAILED
            reason = ("the normal equations became numerically singular at "
                      "iteration {}, with the duality gap at {:.3g}"
                      .format(len(records), gap))
            break
        records.append({
            "N": rows,
            "d": _pattern_sparsity(matrix),
            "kappa": plain,
            "kappa_equilibrated": equilibrated,
            "duality_gap": gap,
        })

        # Predictor: aim straight at complementarity and see how far that gets.
        try:
            affine = _direction(matrix, system, scaling, slack, residual_p,
                                residual_d, -primal * slack)
        except np.linalg.LinAlgError:
            status = Status.TRUNCATED if records else Status.FAILED
            reason = "the normal equations became singular at iteration {}" \
                .format(len(records))
            break
        step_p = _step(affine[0], primal)
        step_d = _step(affine[2], slack)
        probe = float((primal + step_p * affine[0])
                      @ (slack + step_d * affine[2])) / columns
        centring = (probe / gap) ** 3 if gap > 0 else 0.0

        # Corrector: aim at a point on the central path instead.
        target = (-primal * slack + centring * gap
                  - affine[0] * affine[2])
        combined = _direction(matrix, system, scaling, slack, residual_p,
                              residual_d, target)
        step_p = _step(combined[0], primal)
        step_d = _step(combined[2], slack)

        primal = primal + step_p * combined[0]
        dual = dual + step_d * combined[1]
        slack = slack + step_d * combined[2]
        if np.min(primal) <= 0 or np.min(slack) <= 0:
            status = Status.TRUNCATED if records else Status.FAILED
            reason = "an iterate reached the boundary, which is a breakdown of " \
                     "this implementation rather than of the instance"
            break
        if max(step_p, step_d) < _STALLED:
            status = Status.TRUNCATED if records else Status.FAILED
            reason = ("the steps became too small to make progress at "
                      "iteration {}, with the duality gap at {:.3g}"
                      .format(len(records), gap))
            break
    else:
        status = Status.TRUNCATED if records else Status.FAILED
        reason = "the method had not converged after {} iterations".format(
            _MAX_ITERATIONS)

    return Run(
        implementation=IMPLEMENTATION,
        status=status,
        records=tuple(records),
        instance={},
        elapsed=budget.elapsed,
        budget=budget.seconds,
        result={"objective": float(form.value_of(float(objective @ primal))),
                "duality_gap": gap, "iterations": len(records)}
        if status is Status.COMPLETE else {"duality_gap": gap},
        reason=reason,
        assumptions=(
            "the system logged is the normal-equation system A D A' in the "
            "dual variables, of dimension m",
            "the condition number is the unmodified system's. The paper's "
            "modified form is not reimplemented here; a diagonal "
            "equilibration, the usual reading of it, is logged beside it as "
            "kappa_equilibrated and is sometimes smaller and sometimes larger. "
            "The cost is quadratic in this quantity, so unlike everything else "
            "here it is not established as favourable to the quantum side in "
            "either direction",
            "d is the structural row density of A A', which is what A D A' has "
            "whatever the diagonal holds",
            "the path is Mehrotra's, taking {} of the distance to the "
            "boundary; another implementation's iterates sit elsewhere and "
            "have other condition numbers".format(_FRACTION),
        ) + ((
            "{} redundant constraint row{} removed before solving, as any "
            "interior point code's presolve does, so the logged system "
            "dimension is the reduced one".format(
                dropped, "" if dropped == 1 else "s"),
        ) if dropped else ()),
    )


def _direction(matrix: np.ndarray, system: np.ndarray, scaling: np.ndarray,
               slack: np.ndarray, residual_p: np.ndarray,
               residual_d: np.ndarray, target: np.ndarray):
    """One Newton solve against the same system.

    The predictor and the corrector differ only in ``target``, which is why an
    iteration faces one system and not two -- and therefore why a record is an
    iteration.

    Solved rather than multiplied by an inverse, and the difference is not
    stylistic here.  These systems reach condition numbers of ``10^16`` by the
    last iteration -- that is the phenomenon being logged -- and an explicit
    inverse at that conditioning destroys primal feasibility faster than the
    duality gap closes, so the method stalls short of the optimum and the last
    and most ill-conditioned iterations, the ones the cost is dominated by,
    never get logged. Forming the inverse would corrupt the measurement.
    """
    inner = residual_p + matrix @ (scaling * residual_d - target / slack)
    step_dual = np.linalg.solve(system, inner)
    step_slack = residual_d - matrix.T @ step_dual
    step_primal = target / slack - scaling * step_slack
    return step_primal, step_dual, step_slack


def _failed(budget: Budget, reason: str) -> Run:
    return Run(implementation=IMPLEMENTATION, status=Status.FAILED,
               budget=budget.seconds, elapsed=budget.elapsed, reason=reason)
