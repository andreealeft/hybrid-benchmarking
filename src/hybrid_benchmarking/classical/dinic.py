"""Dinic's algorithm, instrumented for the quantum breadth-first search route.

The route wants one thing per record: how many vertices landed in each layer of
each breadth-first sweep.  That is not a summary statistic anyone can guess at
-- it is the shape of the level graph at that moment in the solve -- so the only
way to get it is to run Dinic and watch.

What is logged, and why it is these numbers.  ``qBFS`` replaces the sweep that
builds the level graph: for each layer it calls quantum search repeatedly until
every vertex of that layer has been found, over a register holding one qubit per
vertex.  So the search space is the whole vertex set, whichever layer is being
built, and the marked count is that layer's occupancy.  The record is therefore
the list of occupancies, level by level, starting at the source's own layer --
which holds one vertex and costs almost nothing, but is a layer, and the
library's own worked example includes it.

Two choices worth stating because they change the total:

*Every sweep is logged, including the last one.*  Dinic stops when the sink is no
longer reachable, and it learns that by performing a breadth-first sweep that
fails.  That sweep happened and the quantum version would perform it too, so it
is a record like any other.

*The sweep is run to exhaustion, not stopped at the sink.*  An implementation
may abandon the search once the sink is levelled; ours levels everything
reachable, because the layered graph the blocking-flow step walks is defined
over all of it.  Stopping early would log smaller layers and a smaller cost.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

from ..instances import Network
from .budget import Budget, Run, Status

IMPLEMENTATION = ("Dinic's algorithm, this library's own pure-Python "
                  "implementation -- not the solver any published figure used")

#: How often the blocking-flow walk looks at the clock.  Between iteration
#: boundaries there is nothing to record, so checking every step would only cost
#: clock reads; checking never would let one long phase overrun the budget.
_CLOCK_EVERY = 4096


class _Residual:
    """The residual network, as paired forward and backward arcs.

    Arc ``e`` and arc ``e ^ 1`` are each other's reverse, which is what makes
    pushing flow a subtraction on one and an addition on the other.
    """

    def __init__(self, vertices: int):
        self.adjacent: List[List[int]] = [[] for _ in range(vertices)]
        self.head: List[int] = []
        self.capacity: List[float] = []

    def add(self, tail: int, head: int, capacity: float) -> None:
        self.adjacent[tail].append(len(self.head))
        self.head.append(head)
        self.capacity.append(float(capacity))
        self.adjacent[head].append(len(self.head))
        self.head.append(tail)
        self.capacity.append(0.0)

    def tail_of(self, arc: int) -> int:
        return self.head[arc ^ 1]


def _sweep(residual: _Residual, source: int, vertices: int) -> List[int]:
    """One breadth-first sweep: the level of every vertex, or -1 if unreached."""
    level = [-1] * vertices
    level[source] = 0
    queue = deque([source])
    while queue:
        vertex = queue.popleft()
        for arc in residual.adjacent[vertex]:
            reached = residual.head[arc]
            if residual.capacity[arc] > 0 and level[reached] < 0:
                level[reached] = level[vertex] + 1
                queue.append(reached)
    return level


def layer_sizes(level: Sequence[int]) -> List[int]:
    """Occupancy of each layer, from the source's own down to the deepest.

    Empty trailing layers cannot occur -- breadth-first search fills them in
    order -- so the list is exactly as long as the level graph is deep.
    """
    deepest = max(level)
    sizes = [0] * (deepest + 1)
    for depth in level:
        if depth >= 0:
            sizes[depth] += 1
    return sizes


def _blocking_flow(residual: _Residual, level: List[int], source: int,
                   sink: int, budget: Budget) -> Tuple[float, bool]:
    """Saturate every shortest path in the level graph.

    Written as a walk with an explicit path rather than recursively: the graphs
    people hand this are deeper than Python's stack.  Returns the flow pushed
    and whether the clock ran out partway.
    """
    forward = [0] * len(residual.adjacent)
    pushed = 0.0
    path: List[int] = []
    vertex = source
    steps = 0

    while True:
        steps += 1
        if steps % _CLOCK_EVERY == 0 and budget.spent:
            return pushed, True

        if vertex == sink:
            bottleneck = min(residual.capacity[arc] for arc in path)
            for arc in path:
                residual.capacity[arc] -= bottleneck
                residual.capacity[arc ^ 1] += bottleneck
            pushed += bottleneck
            # Retreat to the tail of the first arc this augmentation saturated;
            # everything before it still has capacity and can be reused.
            first_full = next(
                index for index, arc in enumerate(path)
                if residual.capacity[arc] <= 0
            )
            vertex = residual.tail_of(path[first_full])
            del path[first_full:]
            continue

        advanced = False
        while forward[vertex] < len(residual.adjacent[vertex]):
            arc = residual.adjacent[vertex][forward[vertex]]
            reached = residual.head[arc]
            if residual.capacity[arc] > 0 and level[reached] == level[vertex] + 1:
                path.append(arc)
                vertex = reached
                advanced = True
                break
            forward[vertex] += 1

        if not advanced:
            if vertex == source:
                return pushed, False
            # Nothing leaves this vertex inside the level graph, so no shortest
            # path uses it again this phase.
            level[vertex] = -1
            arc = path.pop()
            vertex = residual.tail_of(arc)
            forward[vertex] += 1


def solve(network: Network, budget: Budget) -> Run:
    """Run Dinic on a network, logging the layer sizes of every sweep."""
    vertices = network.vertices
    residual = _Residual(vertices)
    for tail, head, capacity in network.arcs:
        if capacity < 0:
            return Run(
                implementation=IMPLEMENTATION, status=Status.FAILED,
                budget=budget.seconds, elapsed=budget.elapsed,
                reason="arc {} -> {} has negative capacity {}".format(
                    tail, head, capacity),
            )
        residual.add(tail, head, capacity)

    if network.source_vertex == network.sink_vertex:
        return Run(
            implementation=IMPLEMENTATION, status=Status.FAILED,
            budget=budget.seconds, elapsed=budget.elapsed,
            reason="the source and the sink are the same vertex, so there is "
                   "nothing to route",
        )

    records: List[Dict[str, Any]] = []
    flow = 0.0
    status = Status.COMPLETE

    while True:
        level = _sweep(residual, network.source_vertex, vertices)
        # The sweep happened whether or not it reached the sink, so it is a
        # record either way -- including the final one, which is how Dinic
        # discovers that it is finished.
        records.append({"layers": layer_sizes(level)})

        if level[network.sink_vertex] < 0:
            break
        if budget.spent:
            status = Status.TRUNCATED
            break

        pushed, ran_out = _blocking_flow(
            residual, level, network.source_vertex, network.sink_vertex, budget
        )
        flow += pushed
        if ran_out:
            status = Status.TRUNCATED
            break

    return Run(
        implementation=IMPLEMENTATION,
        status=status,
        records=tuple(records),
        instance={"vertices": vertices},
        elapsed=budget.elapsed,
        budget=budget.seconds,
        result={"maximum_flow": flow, "sweeps": len(records)}
        if status is Status.COMPLETE else {"flow_so_far": flow},
        reason="" if status is Status.COMPLETE else
               "the blocking-flow phase was still running when the budget ran out",
        assumptions=(
            "every breadth-first sweep is logged, including the final one that "
            "fails to reach the sink and so ends the algorithm",
            "each sweep levels every reachable vertex rather than stopping at "
            "the sink, which is what the layered graph is defined over",
        ),
    )
