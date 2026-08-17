"""A revised simplex with the callbacks already in place.

The quantum simplex route wants seven numbers per iteration, and every one of
them is a property of the basis at that moment.  There is no way to get them
except to run the simplex and look, which is what Section 4.6.2 of the thesis
did to GLPK and what this does to a few hundred lines of numpy.

Each logged quantity is a decision, and each of them has a way of being subtly
wrong that produces a plausible gate count no test would catch.  So they are set
out here with the reason, against the text they come from.

**kappa** is not the condition number of the basis.  Computing ``kappa(A_B)``
every iteration was the one major efficiency drawback the thesis names, so it
takes what GLPK computes instead -- ``kappa_1(A_B) = ||A_B||_1 ||A_B^-1||_1``,
equation (4.33) -- and records ``max(1, kappa_1 / m)`` as a lower bound on the
true condition number, justified by ``kappa(A_B) >= 1`` together with
``||A_B||_2 >= ||A_B||_1 / sqrt(m)``.  This is deliberately the smaller
quantity, and the cost is increasing in it, so the total stays a bound
favourable to the quantum side.  Using an exact ``kappa_2`` here would produce a
larger number that is not the published one and not a lower bound either.

**d is the larger of the two sparsities, not the column maximum.**  The
thesis's list of iteration-related measures names the column maximum; taking it
literally makes the normalisation below come out false whenever a row is denser
than every column, which a maximum-flow program manages at any vertex of degree
four.  See :func:`sparsity`.

**A_1 and A_max are not the norms of the basis.**  ``SimplexIter`` operates on a
normalised matrix, so the thesis lower-bounds Lemma 7's two norms by
``||A_B||_1 / (d ||A_B||_max)`` and ``1 / d`` respectively, and those are what go
in the log under those names.  The raw norms are logged too, under their own
names, so nothing is lost -- but a log carrying raw norms in these columns
would silently cost a differently scaled program.

**t is the number of positive components of u**, where ``u = A_B^-1 A_k`` for
the entering column.  Lemma 10 says so in as many words.  It is *not* the number
of improving columns: that quantity is what Lemma 24's search over the
``n - m`` non-basic columns is marked by, it routinely exceeds ``m``, and
feeding it to a search over the ``m`` components of ``u`` would mark more
elements than the list holds -- which the validity domain refuses outright.  The
improving-column count is logged as well, under ``t_improving``, because the
thesis logs it and because the random pivoting rule needs it.

**u_norm is the norm of that same u**, for the column that actually entered, not
an average over candidates: Lemma 11's binary search is on the ratio test for
one specific column.

**c_max is the largest absolute entry of the objective**, an instance-wide
constant, and it stays the instance's objective through the first phase even
though the first phase is minimising something else.  That is what
"instance-related" means in the thesis's list, and it is what the cost formula
is stated over.

Two further things about this run rather than about the quantities.  The
iteration that discovers optimality is not logged, because it has no entering
column and so no ``u``, and a record with two of its seven fields undefined is
worse than a record fewer.  That happens once per phase, so a complete two-phase
solve understates by two optimality tests.  And the first phase is logged along with the second, marked by
a ``phase`` column, because the solver the thesis instrumented has a first phase
too -- but its search is over a wider program than the ``n`` recorded here, so
those records understate as well.  Both are recorded as assumptions on the cost
rather than left in a docstring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .budget import Budget, Run, Status
from .lp import StandardForm

IMPLEMENTATION = ("a revised simplex with steepest-edge pricing, this "
                  "library's own numpy implementation -- not GLPK, which is "
                  "what every published figure logged")

#: Below this a number is zero.  The standard forms built here have entries of
#: modest size, and the quantities this decides -- which columns improve, which
#: components of the pivot direction are positive -- are counts that go straight
#: into a search's marked-element count, so it is recorded on every run rather
#: than left as a constant someone would have to come here to find.
TOLERANCE = 1e-9

#: Consecutive degenerate pivots before falling back on Bland's rule, which
#: cannot cycle but prices badly.  Recorded when it happens, because it changes
#: which columns enter and so changes every number logged afterwards.
_BLAND_AFTER = 50


def _column_sparsity(basis_matrix: np.ndarray) -> int:
    return int(np.max(np.count_nonzero(basis_matrix, axis=0)))


def _row_sparsity(basis_matrix: np.ndarray) -> int:
    return int(np.max(np.count_nonzero(basis_matrix, axis=1)))


def sparsity(basis_matrix: np.ndarray) -> int:
    """The larger of the row and column maxima, and it has to be the larger.

    The thesis's list of iteration-related measures names the column maximum,
    and taking it literally would be wrong here -- not as a matter of taste but
    because it makes the normalisation the same section states come out false.

    Those bounds are ``||A_hat||_1 >= ||A_B||_1 / (d ||A_B||max)`` and
    ``||A_hat||max >= 1/d``, and both rest on ``||A_B||_2 <= d ||A_B||max``.
    What actually holds is ``||A||_2 <= sqrt(d_col d_row) ||A||max``, so the
    divisor has to be at least the geometric mean of the two sparsities; the
    column maximum alone is below it whenever a row is denser than every column.
    Then ``1/d`` is *above* the true normalised largest entry rather than below
    it, the cost is increasing in both norms, and a number that calls itself a
    lower bound is an over-estimate.

    That is not hypothetical.  A maximum-flow program has three non-zeros in
    every arc column and one per incident arc in every conservation row, so any
    vertex of degree four or more inverts the two: on a thirty-relay network
    fifty-six of a hundred and twenty-one bases violate the bound, and the total
    lands sixteen per cent above what the convention it declares would give.

    The larger of the two is also the honest sparsity for the other place ``d``
    is used -- as the sparsity of the Hermitian matrix a quantum linear solver
    acts on, which for a matrix that is not symmetric is the dilation, whose
    rows are the original's rows and its columns.  :mod:`.linsystem` reached the
    same reading independently, and now they agree.
    """
    return max(_column_sparsity(basis_matrix), _row_sparsity(basis_matrix))


def _record(basis_matrix: np.ndarray, inverse: np.ndarray, direction: np.ndarray,
            rows: int, phase: int) -> Dict[str, Any]:
    """One iteration's worth of log, with every convention above applied."""
    kappa_1 = float(np.linalg.norm(basis_matrix, 1)
                    * np.linalg.norm(inverse, 1))
    density = sparsity(basis_matrix)
    norm_1 = float(np.linalg.norm(basis_matrix, 1))
    norm_max = float(np.max(np.abs(basis_matrix)))

    return {
        # what the route consumes
        "kappa": max(1.0, kappa_1 / rows),
        "d": density,
        "A_1": norm_1 / (density * norm_max),
        "A_max": 1.0 / density,
        "t": int(np.count_nonzero(direction > TOLERANCE)),
        "u_norm": float(np.linalg.norm(direction, 2)),
        # what it was computed from, so none of the above is a bare assertion
        "kappa_1": kappa_1,
        "A_B_1": norm_1,
        "A_B_max": norm_max,
        "d_columns": _column_sparsity(basis_matrix),
        "d_rows": _row_sparsity(basis_matrix),
        "phase": phase,
    }


def _entering_steepest_edge(reduced: np.ndarray, improving: np.ndarray,
                            inverse: np.ndarray, matrix: np.ndarray
                            ) -> Tuple[int, np.ndarray]:
    """The column with the most negative reduced cost per unit of movement.

    The classical rule the thesis ran on both sides of its comparison.  The
    pivot directions of every candidate are needed to price it, and one of them
    is the direction the chosen column moves in, so the second is taken from the
    first rather than recomputed.
    """
    directions = inverse @ matrix[:, improving]
    lengths = np.linalg.norm(directions, axis=0)
    lengths[lengths < TOLERANCE] = TOLERANCE
    scores = reduced[improving] / lengths
    best = int(np.argmin(scores))
    return int(improving[best]), directions[:, best]


def _entering_dantzig(reduced: np.ndarray, improving: np.ndarray,
                      inverse: np.ndarray, matrix: np.ndarray
                      ) -> Tuple[int, np.ndarray]:
    """The most negative reduced cost, priced without normalising."""
    best = int(improving[int(np.argmin(reduced[improving]))])
    return best, inverse @ matrix[:, best]


def _entering_bland(reduced: np.ndarray, improving: np.ndarray,
                    inverse: np.ndarray, matrix: np.ndarray
                    ) -> Tuple[int, np.ndarray]:
    """The lowest-numbered improving column: slow, and cannot cycle."""
    best = int(improving[0])
    return best, inverse @ matrix[:, best]


_RULES = {
    "steepest-edge": _entering_steepest_edge,
    "dantzig": _entering_dantzig,
    "bland": _entering_bland,
}


def solve(form: StandardForm, budget: Budget,
          rule: str = "steepest-edge") -> Run:
    """Solve a standard-form program, logging every iteration that pivots."""
    if rule not in _RULES:
        raise ValueError("no pivoting rule {!r}; there is {}".format(
            rule, ", ".join(sorted(_RULES))))

    rows, columns = form.matrix.shape
    objective = np.asarray(form.objective, dtype=float)
    c_max = float(np.max(np.abs(objective))) if columns else 0.0
    if c_max <= TOLERANCE:
        return Run(
            implementation=IMPLEMENTATION, status=Status.FAILED,
            budget=budget.seconds, elapsed=budget.elapsed,
            reason="the objective is identically zero, so there is no pivoting "
                   "rule to instrument -- every column looks the same to a "
                   "steepest-edge comparison",
        )

    # The first phase runs on the program widened by an artificial identity,
    # which is the only basis available before anything is known to be feasible.
    widened = np.hstack([form.matrix, np.eye(rows)])
    artificial = np.zeros(rows + columns, dtype=bool)
    artificial[columns:] = True
    basis = list(range(columns, columns + rows))

    records: List[Dict[str, Any]] = []
    state = _State(records=records, rows=rows, columns=columns, c_max=c_max)

    phase_one_cost = np.zeros(columns + rows)
    phase_one_cost[columns:] = 1.0
    outcome = _run_phase(widened, form.rhs, phase_one_cost, basis, state,
                         budget, rule, phase=1, forbidden=None)
    if outcome is not None:
        return outcome(state, budget)

    # What the first phase was minimising: the total weight still carried by
    # artificial columns.  Anything above rounding means no feasible point.
    at_optimum = np.linalg.solve(widened[:, basis], form.rhs)
    infeasibility = float(phase_one_cost[basis] @ at_optimum)
    scale = max(1.0, float(np.max(np.abs(form.rhs))) if rows else 1.0)
    if infeasibility > TOLERANCE * scale:
        return state.finish(Status.FAILED, budget,
                            reason="the program is infeasible: the first phase "
                                   "could not drive the artificial variables "
                                   "to zero")

    phase_two_cost = np.zeros(columns + rows)
    phase_two_cost[:columns] = objective
    outcome = _run_phase(widened, form.rhs, phase_two_cost, basis, state,
                         budget, rule, phase=2, forbidden=artificial)
    if outcome is not None:
        return outcome(state, budget)

    values = np.linalg.solve(widened[:, basis], form.rhs)
    value = float(sum(objective[j] * values[i]
                      for i, j in enumerate(basis) if j < columns))
    return state.finish(
        Status.COMPLETE, budget,
        result={"objective": float(form.value_of(value)),
                "iterations": len(records)},
    )


class _State:
    """Everything the two phases share, and the Run they eventually become."""

    def __init__(self, records: List[Dict[str, Any]], rows: int, columns: int,
                 c_max: float):
        self.records = records
        self.rows = rows
        self.columns = columns
        self.c_max = c_max
        self.used_bland = False

    def finish(self, status: Status, budget: Budget, reason: str = "",
               result: Optional[Dict[str, Any]] = None) -> Run:
        assumptions = [
            "the condition number is GLPK's kappa_1 = ||A_B||_1 ||A_B^-1||_1 "
            "divided by m and floored at one, following (4.33), not the exact "
            "condition number of the basis",
            "A_1 and A_max are the normalised lower bounds ||A_B||_1 / "
            "(d ||A_B||_max) and 1 / d, since SimplexIter runs on a normalised "
            "matrix; the raw norms are in the log under their own names",
            "d is the larger of the row and column sparsities of the basis. "
            "The thesis's list names the column maximum, but the normalisation "
            "it states in the same section needs the larger, and taking the "
            "column maximum makes A_1 and A_max over-estimates rather than the "
            "lower bounds they are declared to be. Both readings are logged, "
            "as d_columns and d_rows",
            "the iteration that establishes optimality is not logged, having "
            "no entering column; that happens once per phase, so a complete "
            "two-phase solve omits two optimality tests",
            "the first phase is logged alongside the second, but over the "
            "instance's own column count rather than the widened one it "
            "actually searches",
            "a component of the pivot direction counts as positive above "
            "{:g}; t is a marked-element count, so that threshold is part of "
            "the number".format(TOLERANCE),
        ]
        if self.used_bland:
            assumptions.append(
                "Bland's rule took over after repeated degenerate pivots, so "
                "the columns that entered after that point are not the ones "
                "steepest-edge pricing would have chosen"
            )
        return Run(
            implementation=IMPLEMENTATION,
            status=status,
            records=tuple(self.records),
            instance={"n": self.columns, "m": self.rows, "c_max": self.c_max},
            elapsed=budget.elapsed,
            budget=budget.seconds,
            result=result,
            reason=reason,
            assumptions=tuple(assumptions),
        )


def _run_phase(matrix: np.ndarray, rhs: np.ndarray, cost: np.ndarray,
               basis: List[int], state: _State, budget: Budget, rule: str,
               phase: int, forbidden: Optional[np.ndarray]):
    """Iterate until optimal, and hand back a finisher if it ended otherwise.

    Returning a callable rather than a Run keeps the budget check at the one
    place it belongs -- the top of the loop, between iterations -- while letting
    the caller stop the whole solve rather than only this phase.
    """
    degenerate = 0

    while True:
        if budget.spent:
            return lambda s, b: s.finish(
                Status.TRUNCATED if s.records else Status.FAILED, b,
                reason="the budget ran out during phase {}".format(phase),
            )

        basis_matrix = matrix[:, basis]
        try:
            inverse = np.linalg.inv(basis_matrix)
        except np.linalg.LinAlgError:
            return lambda s, b: s.finish(
                Status.TRUNCATED if s.records else Status.FAILED, b,
                reason="the basis became singular, which is a breakdown of "
                       "this implementation rather than a property of the "
                       "instance",
            )

        multipliers = cost[basis] @ inverse
        reduced = cost - multipliers @ matrix
        reduced[basis] = 0.0
        candidates = reduced < -TOLERANCE
        if forbidden is not None:
            candidates &= ~forbidden
        improving = np.flatnonzero(candidates)
        if improving.size == 0:
            return None  # optimal for this phase

        choose = _RULES["bland" if degenerate >= _BLAND_AFTER else rule]
        if degenerate >= _BLAND_AFTER:
            state.used_bland = True
        entering, direction = choose(reduced, improving, inverse, matrix)

        # The record describes the basis and the entering column as the quantum
        # subroutines would see them, so it is written before the pivot.
        record = _record(basis_matrix, inverse, direction, state.rows, phase)
        record["t_improving"] = int(improving.size)
        state.records.append(record)

        positive = direction > TOLERANCE
        if not positive.any():
            return lambda s, b: s.finish(
                Status.COMPLETE if phase == 2 else Status.FAILED, b,
                reason="the program is unbounded: column {} improves the "
                       "objective and nothing limits it".format(entering),
                result={"objective": "unbounded",
                        "iterations": len(s.records)},
            )

        current = inverse @ rhs
        ratios = np.where(positive, current / np.where(positive, direction, 1.0),
                          np.inf)
        step = float(np.min(ratios))
        if degenerate >= _BLAND_AFTER:
            leaving = int(min(
                (i for i in np.flatnonzero(ratios <= step + TOLERANCE)),
                key=lambda i: basis[i],
            ))
        else:
            leaving = int(np.argmin(ratios))

        degenerate = degenerate + 1 if step <= TOLERANCE else 0
        basis[leaving] = entering
