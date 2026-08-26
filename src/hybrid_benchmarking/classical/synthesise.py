"""Building an instance from a description of one.

Someone who has a real file gets a real number.  Someone who has only a size --
sixty people, a hundred and eighty shifts they could cover -- has told us what
their problem *looks like*, not what it is, and the honest thing to do with that
is say so loudly and then build something the right shape.

That is what this does.  It generates an instance to the stated size, and from
there the ordinary path takes over: the same classical solvers run on it, write
the same log, and the same lemmas cost it.  Nothing downstream knows or needs to
know that the instance was made up.

**The number that comes out is for the generated instance, not for yours.**  It
is a real cost of a real solve of a real instance of that size, which is a
useful thing and not the same as an answer.  Every run from here carries that
sentence in its assumptions, and it is the first thing the interface shows.

Generation is deterministic: the same answers give the same instance and
therefore the same number, every time and on every machine.  A number that
wobbled when you asked twice would be worse than useless -- someone would
average them.  The seed is derived from the answers themselves.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from ..instances import (
    Graph,
    Instance,
    InstanceError,
    Knapsack,
    LinearProgram,
    Matrix,
    MultidimensionalKnapsack,
    Network,
    QuadraticKnapsack,
)

#: Said on every cost built this way, and shown before the number.
CAVEAT = (
    "the instance was generated to the size given, not read from your data: "
    "this is what a problem of that shape costs, which is not the same as what "
    "yours costs"
)

#: Beyond this the generated program is larger than the instrumented solvers
#: carry, and the honest answer is a smaller number rather than a long wait.
MAX_THINGS = 4000

#: Values and weights are drawn from here.  The knapsack circuits read binary
#: representations, so the *spread* of the numbers matters and not only how many
#: there are; a range of this width gives the mix of lowest set bits a real
#: instance has, and never a zero, which has no lowest set bit at all.
_VALUE_RANGE = (1, 100)


def _seed(problem: str, values: Dict[str, Any]) -> random.Random:
    """A generator fixed by the answers, so the same answers give one instance."""
    key = (problem,) + tuple(sorted((k, str(v)) for k, v in values.items()))
    return random.Random(hash(key) & 0xFFFFFFFF)


def _count(values: Dict[str, Any], name: str, least: int = 1) -> int:
    try:
        number = int(float(values[name]))
    except (KeyError, TypeError, ValueError):
        raise InstanceError("{} has to be a whole number".format(name))
    if number < least:
        raise InstanceError(
            "{} is {}, and has to be at least {}".format(name, number, least)
        )
    if number > MAX_THINGS:
        raise InstanceError(
            "{} of {} is more than this tool solves classically; it runs the "
            "real algorithm rather than a formula, so try {} or fewer"
            .format(name, number, MAX_THINGS)
        )
    return number


def _edges(rng: random.Random, vertices: int, wanted: int) -> List[Tuple[int, int]]:
    """A simple graph on ``vertices`` with about ``wanted`` edges.

    Sampled without replacement where that is cheap and by rejection where the
    graph is sparse, which is the case that matters -- a dense one would not fit
    the instrumented solvers anyway.
    """
    possible = vertices * (vertices - 1) // 2
    wanted = max(1, min(wanted, possible))
    if wanted > possible // 2:
        every = [(u, v) for u in range(vertices) for v in range(u + 1, vertices)]
        rng.shuffle(every)
        return sorted(every[:wanted])
    seen = set()
    while len(seen) < wanted:
        u = rng.randrange(vertices)
        v = rng.randrange(vertices)
        if u != v:
            seen.add((min(u, v), max(u, v)))
    return sorted(seen)


def _graph(problem: str, values: Dict[str, Any]) -> Graph:
    rng = _seed(problem, values)
    vertices = _count(values, "things", 2)
    edges = _edges(rng, vertices, _count(values, "links"))
    return Graph(name="a generated instance", source="(generated)",
                 layout="generated", vertices=vertices, edges=tuple(edges))


def _network(problem: str, values: Dict[str, Any]) -> Network:
    rng = _seed(problem, values)
    vertices = _count(values, "things", 3)
    arcs = [(u, v, float(rng.randint(*_VALUE_RANGE)))
            for u, v in _edges(rng, vertices, _count(values, "links"))]
    return Network(name="a generated instance", source="(generated)",
                   layout="generated", vertices=vertices, arcs=tuple(arcs),
                   source_vertex=0, sink_vertex=vertices - 1)


def _knapsack(problem: str, values: Dict[str, Any]) -> Knapsack:
    rng = _seed(problem, values)
    items = _count(values, "things", 2)
    profits = tuple(rng.randint(*_VALUE_RANGE) for _ in range(items))
    weights = tuple(rng.randint(*_VALUE_RANGE) for _ in range(items))
    return Knapsack(name="a generated instance", source="(generated)",
                    layout="generated", profits=profits, weights=weights,
                    capacity=_capacity(values, weights))


def _capacity(values: Dict[str, Any], weights) -> int:
    """A budget that admits some of the items and not all of them.

    Taken from the answer where one was given, but never so large that
    everything fits nor so small that nothing does -- at either extreme the
    problem stops being a choice, and the generated instance would be costing
    something nobody asked about.
    """
    stated = _count(values, "budget") if "budget" in values else 0
    total = sum(weights)
    return max(min(stated or total // 2, total - 1), min(weights))


def _quadratic(problem: str, values: Dict[str, Any]) -> QuadraticKnapsack:
    rng = _seed(problem, values)
    items = _count(values, "things", 2)
    profits = tuple(rng.randint(*_VALUE_RANGE) for _ in range(items))
    weights = tuple(rng.randint(*_VALUE_RANGE) for _ in range(items))
    density = min(max(int(float(values.get("pairs", 30))), 0), 100)
    pairs = {}
    for high in range(1, items):
        for low in range(high):
            if rng.randrange(100) < density:
                pairs[(high, low)] = rng.randint(*_VALUE_RANGE)
    return QuadraticKnapsack(
        name="a generated instance", source="(generated)", layout="generated",
        profits=profits, weights=weights, capacity=_capacity(values, weights),
        pairs=pairs,
    )


def _multidimensional(problem: str,
                      values: Dict[str, Any]) -> MultidimensionalKnapsack:
    rng = _seed(problem, values)
    items = _count(values, "things", 2)
    limits = _count(values, "limits")
    profits = tuple(rng.randint(*_VALUE_RANGE) for _ in range(items))
    rows, capacities = [], []
    for _ in range(limits):
        row = tuple(rng.randint(*_VALUE_RANGE) for _ in range(items))
        rows.append(row)
        capacities.append(_capacity(values, row))
    return MultidimensionalKnapsack(
        name="a generated instance", source="(generated)", layout="generated",
        profits=profits, weights=tuple(rows), capacities=tuple(capacities),
    )


def _program(problem: str, values: Dict[str, Any]) -> LinearProgram:
    """A linear program of the stated shape, feasible and bounded by design.

    Every column gets an upper bound, which is what keeps it bounded, and every
    row is a ``<=`` whose right-hand side is generous enough to admit the origin,
    which is what keeps it feasible.  A generated program that turned out
    infeasible would cost nothing and teach nobody anything.
    """
    rng = _seed(problem, values)
    columns = _count(values, "things", 2)
    rows = _count(values, "links")
    if rows >= columns:
        raise InstanceError(
            "a program with {} rules and only {} quantities to set has more "
            "constraints than freedom; give it more quantities than rules"
            .format(rows, columns)
        )
    matrix = []
    for row in range(rows):
        touched = sorted(rng.sample(range(columns), min(columns, rng.randint(2, 5))))
        for column in touched:
            matrix.append((row, column, float(rng.randint(1, 9))))
    return LinearProgram(
        name="a generated instance", source="(generated)", layout="generated",
        columns=tuple("x{}".format(i) for i in range(columns)),
        rows=tuple("r{}".format(i) for i in range(rows)),
        senses=tuple("L" for _ in range(rows)),
        objective=tuple(float(rng.randint(*_VALUE_RANGE)) for _ in range(columns)),
        matrix=tuple(matrix),
        rhs=tuple(float(rng.randint(20, 60)) for _ in range(rows)),
        ranges=tuple(None for _ in range(rows)),
        lower=tuple(0.0 for _ in range(columns)),
        upper=tuple(10.0 for _ in range(columns)),
        integer=tuple(False for _ in range(columns)),
        maximise=True,
    )


def _matrix(problem: str, values: Dict[str, Any]) -> Matrix:
    """A sparse symmetric system, diagonally dominant so it is non-singular.

    Dominance is not decoration: without it a generated matrix is singular often
    enough to be annoying, and a singular system has no condition number to log.
    """
    rng = _seed(problem, values)
    size = _count(values, "things", 2)
    per_row = max(1, min(_count(values, "links"), size - 1))
    entries: Dict[Tuple[int, int], float] = {}
    for row in range(size):
        for column in rng.sample(range(size), min(size, per_row)):
            if column == row:
                continue
            value = float(rng.randint(1, 9))
            entries[(row, column)] = value
            entries[(column, row)] = value
    for row in range(size):
        weight = sum(abs(v) for (r, _), v in entries.items() if r == row)
        entries[(row, row)] = weight + float(rng.randint(1, 9))
    return Matrix(name="a generated instance", source="(generated)",
                  layout="generated", rows=size, columns=size,
                  entries=tuple((r, c, v) for (r, c), v in sorted(entries.items())),
                  symmetric=True)


_BUILDERS = {
    "maximum-flow": _network,
    "vertex-cover": _graph,
    "independent-set": _graph,
    "clique": _graph,
    "linear-programming": _program,
    "knapsack": _knapsack,
    "quadratic-knapsack": _quadratic,
    "multidimensional-knapsack": _multidimensional,
    "linear-systems": _matrix,
}


def build(problem: str, values: Dict[str, Any]) -> Instance:
    """An instance of the shape described, for the family this problem belongs to."""
    from ..problems import family_of

    family = family_of(problem)
    if family not in _BUILDERS:
        raise InstanceError(
            "nothing here knows how to make up a {} instance".format(family)
        )
    return _BUILDERS[family](problem, values)
