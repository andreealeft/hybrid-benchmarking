"""Knapsack, which needs reading rather than running.

Alone among the routes here, the tree generator's cost does not depend on how a
classical solver behaved.  It depends on the instance: each profit and weight
enters through the position of its lowest set bit, because that is where the
Fourier adder's rotations start.  So there is nothing to instrument and nothing
to time -- the file already contains every input but one, and producing that one
is arithmetic.

That missing input is the **profit bound**, which fixes the width of the profit
register and so appears inside every count in Appendix C.  Any upper bound on
the best achievable value will do, and a smaller one gives a smaller circuit, so
the choice is not neutral.  Two are available and the tighter is taken:

*the Dantzig bound*, the value of the fractional relaxation -- items by
decreasing value per unit of cost until the budget runs out, with the one that
straddles it taken in part.  It is an upper bound on the integer optimum, it
costs a sort, and floored it is an integer;

*the optimum the file states*, where the generator recorded one.  It is exact,
so it is the tightest bound there is, and the register only ever has to hold the
profit of a feasible path -- every one of which is at most the optimum.

Whichever was used is recorded on the cost, because a reader comparing two
knapsack counts needs to know whether the difference is the instances or the
bounds.

Because nothing was run, the resulting count is analytic rather than logged.
Marking it logged would hedge a number that is not hedged: these gate and cycle
counts are exact functions of the instance, and would come out identical on
anyone else's machine.

What is *not* produced here is the amplification schedule.  ``QTGSearch`` wraps
the generator in maximum finding against a rising profit threshold, and its cost
depends on the Grover powers actually drawn, which is a property of a run rather
than of an instance -- a simulation, not a reading.  So this costs the state
preparation and says so.
"""

from __future__ import annotations

from typing import List, Tuple

from ..instances import Knapsack
from .budget import Budget, Run, Status

IMPLEMENTATION = ("the instance itself, read rather than solved: the tree "
                  "generator's cost is a function of the profits and weights, "
                  "so there was no classical run to instrument")

HANDOFF = (
    "this costs the tree generator, which is a state preparation. QTGSearch -- "
    "the maximum finding around it -- needs the schedule of Grover powers the "
    "search actually draws, and that is a simulation of the instance rather "
    "than a property of it. Supply a schedule to cost QTGSearch directly."
)


def dantzig_bound(profits: Tuple[int, ...], weights: Tuple[int, ...],
                  capacity: int) -> int:
    """The fractional relaxation's value, floored.

    Items in decreasing value per unit of cost; the one that straddles the
    budget is taken in part.  Since every profit is an integer, the integer
    optimum cannot exceed the floor of this.
    """
    order = sorted(range(len(profits)),
                   key=lambda i: profits[i] / weights[i], reverse=True)
    remaining = float(capacity)
    total = 0.0
    for index in order:
        if weights[index] <= remaining:
            remaining -= weights[index]
            total += profits[index]
        else:
            total += profits[index] * remaining / weights[index]
            break
    return int(total)


def solve(instance: Knapsack, budget: Budget) -> Run:
    """Read a knapsack instance and work out the one number it does not state."""
    if not instance.profits:
        return Run(implementation=IMPLEMENTATION, status=Status.FAILED,
                   budget=budget.seconds, elapsed=budget.elapsed,
                   reason="the instance has no items")
    if min(instance.weights) > instance.capacity:
        return Run(
            implementation=IMPLEMENTATION, status=Status.FAILED,
            budget=budget.seconds, elapsed=budget.elapsed,
            reason="no item fits in the budget, so every feasible packing is "
                   "empty and there is no profit register to size",
        )

    relaxed = dantzig_bound(instance.profits, instance.weights,
                            instance.capacity)
    chosen, why = relaxed, "the Dantzig bound, from the fractional relaxation"
    if instance.optimum is not None and instance.optimum < relaxed:
        chosen = int(instance.optimum)
        why = ("the optimum the file states, which is tighter than the "
               "Dantzig bound of {}".format(relaxed))
    if chosen < 1:
        return Run(
            implementation=IMPLEMENTATION, status=Status.FAILED,
            budget=budget.seconds, elapsed=budget.elapsed,
            reason="the best achievable value is zero, so the profit register "
                   "has no width",
        )

    return Run(
        implementation=IMPLEMENTATION,
        status=Status.COMPLETE,
        derivation="ANALYTIC",
        records=(),
        instance={
            "profits": list(instance.profits),
            "weights": list(instance.weights),
            "capacity": int(instance.capacity),
            "profit_bound": chosen,
        },
        elapsed=budget.elapsed,
        budget=budget.seconds,
        result={"items": len(instance.profits), "profit_bound": chosen},
        handoff=HANDOFF,
        assumptions=(
            "the profit register is sized by {}".format(why),
            "the amplification schedule is not produced here, so this is the "
            "cost of the tree generator rather than of the search around it",
        ),
    )
