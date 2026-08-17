"""The two DIMACS formats: capacitated networks, and undirected edge lists.

Both come out of the first DIMACS implementation challenge and share a shape --
one line, one letter, then fields separated by whitespace -- but they describe
different objects, so they are read by different functions and produce different
instances.

**Maximum flow** (``.max``), which becomes a :class:`~. Network`::

    c a comment, legal anywhere
    p max 4 5
    n 1 s
    n 4 t
    a 1 2 3

The problem line comes first and appears once; exactly one node is designated
the source and exactly one the sink; each ``a`` line is a directed arc with a
capacity.  Capacities are integers in the specification, but generators emit
floats and a float capacity costs nothing to accept, so they are accepted.

**Undirected graphs** (``.clq``, ``.col``, ``.edges``), which become a
:class:`~. Graph`::

    p edge 4 4
    e 1 2

The same format is called ``edge``, ``edges``, ``col``, ``clq`` and ``clique``
by different collections, in either case; all of them are read here as the one
thing they are.  An ``e`` line may carry a third field, a weight, which is
dropped: :class:`~. Graph` has nowhere to put it, and the linear programs built
from these files do not use it.

Node numbers are one-based in the file and zero-based in what comes out.  That
conversion is the reason a node ``0`` is refused rather than accepted: it is not
a stale index, it is a file written in the other convention, and reading it as
written would shift every vertex by one while still producing a graph.

**Where a file disagrees with its own header.**  Collected DIMACS files do this
constantly -- generators write the counts before appending the last arc, hand
edits do not update the header, and clique files list ids above the declared
node count.  The data is the instance and the header is a claim about it, so:

* the declared node count is a *lower* bound.  The vertex count is the larger of
  it and the largest id seen, which keeps isolated vertices when the header asks
  for them and keeps arcs pointing at real vertices when it does not.  Note this
  is the direction the house tie-break rule does not prefer -- it can only raise
  a count -- but the alternative, dropping a vertex an arc refers to, is not a
  reading of the file at all;
* the declared arc or edge count sizes nothing at all: the arcs read are the
  arcs.  Edge lists that write both directions of every edge contradict it by a
  factor of two before normalisation has even started, so it is worth nothing
  as a count and is kept only as the sanity check below.

Both are believed only within a factor of :data:`_TOLERANCE`.  A header off by
one, or by a factor of two, is a stale header; a header off by an order of
magnitude is a truncated download or the wrong file, and reading it would yield
a plausible instance of the wrong size -- which is the failure this package
exists to avoid.  So a large disagreement raises, naming the problem line.

On the way out, ``source`` is the file the instance came from, and the source
node of a network is :attr:`Network.source_vertex`.  The two are different
things wearing one word throughout this format's literature, which is why the
field is named at length.

Everything else raises :class:`~. InstanceError` naming the offending line.  No
line is skipped for being unrecognised: a format this old accumulates dialects,
and silently ignoring the one line that carried the answer is how a reader
returns a graph that is subtly not the file's.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from . import Graph, InstanceError, Network

__all__ = [
    "parse_graph",
    "parse_max_flow",
    "read_graph",
    "read_max_flow",
]

#: How far a declared count may be from what the file holds before we stop
#: believing the header describes this file.  See the module docstring.
_TOLERANCE = 10

#: Spellings of the undirected edge-list problem line, lower-cased.
_EDGE_WORDS = ("edge", "edges", "col", "clq", "clique")

#: Spellings of the maximum-flow problem line, lower-cased.
_FLOW_WORDS = ("max", "flow")


# ---------------------------------------------------------------------------
# maximum flow
# ---------------------------------------------------------------------------

def parse_max_flow(text: str, name: str = "", source: str = "") -> Network:
    """Read a DIMACS maximum-flow file into a :class:`~. Network`.

    Parallel arcs and self-loops are kept exactly as written -- a maximum-flow
    solver is entitled to see the network it was given, and merging parallel
    arcs would change what an instrumented run reports about it.

    ``source`` names where the text came from; the node the flow starts at is
    :attr:`Network.source_vertex`.
    """
    declared_nodes = None  # type: Optional[int]
    declared_arcs = None  # type: Optional[int]
    problem_at = 0
    source_node = None  # type: Optional[int]
    sink_node = None  # type: Optional[int]
    source_at = 0
    sink_at = 0
    arcs = []  # type: List[Tuple[int, int, float]]
    largest = 0

    for number, raw in enumerate(text.splitlines(), 1):
        words = raw.strip().split()
        if not words:
            continue
        tag = words[0].lower()

        if tag == "c":
            continue

        if tag == "p":
            if declared_nodes is not None:
                raise InstanceError(
                    "line {}: a second problem line; a DIMACS file states "
                    "'p max <nodes> <arcs>' exactly once".format(number)
                )
            declared_nodes, declared_arcs = _problem(
                words, number, _FLOW_WORDS, "arc", "maximum-flow",
                "parse_graph",
            )
            problem_at = number
            continue

        if declared_nodes is None:
            raise InstanceError(
                "line {}: an {!r} line before the problem line; a DIMACS "
                "maximum-flow file states 'p max <nodes> <arcs>' before any "
                "node or arc".format(number, tag)
            )

        if tag == "n":
            if len(words) != 3:
                raise InstanceError(
                    "line {}: a node designator reads 'n <node> s' or "
                    "'n <node> t', but this has {} fields".format(
                        number, len(words)
                    )
                )
            node = _node(words[1], number)
            largest = max(largest, node + 1)
            role = words[2].lower()
            if role == "s":
                if source_node is not None:
                    raise InstanceError(
                        "line {}: a second source designator, the first on "
                        "line {}; the format allows exactly one".format(
                            number, source_at
                        )
                    )
                source_node, source_at = node, number
            elif role == "t":
                if sink_node is not None:
                    raise InstanceError(
                        "line {}: a second sink designator, the first on line "
                        "{}; the format allows exactly one".format(
                            number, sink_at
                        )
                    )
                sink_node, sink_at = node, number
            else:
                raise InstanceError(
                    "line {}: {!r} designates neither a source ('s') nor a "
                    "sink ('t')".format(number, words[2])
                )
            continue

        if tag == "a":
            if len(words) != 4:
                raise InstanceError(
                    "line {}: an arc reads 'a <tail> <head> <capacity>', but "
                    "this has {} fields".format(number, len(words))
                )
            tail = _node(words[1], number)
            head = _node(words[2], number)
            largest = max(largest, tail + 1, head + 1)
            arcs.append((tail, head, _capacity(words[3], number)))
            continue

        raise InstanceError(
            "line {}: {!r} starts no line a DIMACS maximum-flow file has; it "
            "holds only c, p, n and a lines".format(number, words[0])
        )

    if declared_nodes is None or declared_arcs is None:
        raise InstanceError(
            "no problem line: the file states no 'p max <nodes> <arcs>' line "
            "anywhere, so it is not a DIMACS maximum-flow file"
        )
    if source_node is None:
        raise InstanceError(
            "no source: the file designates no node 's', which every DIMACS "
            "maximum-flow instance must (the problem line is line {})".format(
                problem_at
            )
        )
    if sink_node is None:
        raise InstanceError(
            "no sink: the file designates no node 't', which every DIMACS "
            "maximum-flow instance must (the problem line is line {})".format(
                problem_at
            )
        )
    if source_node == sink_node:
        raise InstanceError(
            "line {}: node {} is designated both source and sink; there is "
            "no flow to maximise".format(sink_at, sink_node + 1)
        )

    _believe("arc", declared_arcs, len(arcs), problem_at, True)
    _believe("node", declared_nodes, largest, problem_at, True)

    return Network(
        name=name,
        source=source,
        layout="dimacs-max",
        vertices=max(declared_nodes, largest),
        arcs=tuple(arcs),
        source_vertex=source_node,
        sink_vertex=sink_node,
    )


def read_max_flow(path: Union[str, Path]) -> Network:
    """Read a maximum-flow file, naming it after the file it came from."""
    location = Path(path).expanduser()
    return parse_max_flow(_text(location), location.stem, str(location))


# ---------------------------------------------------------------------------
# undirected graphs
# ---------------------------------------------------------------------------

def parse_graph(text: str, name: str = "", source: str = "") -> Graph:
    """Read a DIMACS edge list into a :class:`~. Graph`.

    Collections disagree about whether an undirected edge is written once or in
    both directions, and some list an edge twice outright, so the edges are
    normalised -- ``u < v``, self-loops dropped, duplicates dropped, sorted --
    and a file written either way gives the same graph.  Dropping a self-loop is
    normalisation, not a line left unread: an undirected graph without weights
    has nowhere to record one, and the programs built from these files would
    gain a row that constrains a variable against itself.
    """
    declared_nodes = None  # type: Optional[int]
    declared_edges = None  # type: Optional[int]
    problem_at = 0
    written = 0
    edges = set()  # type: Set[Tuple[int, int]]
    largest = 0

    for number, raw in enumerate(text.splitlines(), 1):
        words = raw.strip().split()
        if not words:
            continue
        tag = words[0].lower()

        if tag == "c":
            continue

        if tag == "p":
            if declared_nodes is not None:
                raise InstanceError(
                    "line {}: a second problem line; a DIMACS file states "
                    "'p edge <nodes> <edges>' exactly once".format(number)
                )
            declared_nodes, declared_edges = _problem(
                words, number, _EDGE_WORDS, "edge", "edge-list",
                "parse_max_flow",
            )
            problem_at = number
            continue

        if declared_nodes is None:
            raise InstanceError(
                "line {}: an {!r} line before the problem line; a DIMACS edge "
                "list states 'p edge <nodes> <edges>' before any edge".format(
                    number, tag
                )
            )

        if tag == "e":
            if len(words) not in (3, 4):
                raise InstanceError(
                    "line {}: an edge reads 'e <u> <v>', optionally followed "
                    "by a weight, but this has {} fields".format(
                        number, len(words)
                    )
                )
            first = _node(words[1], number)
            second = _node(words[2], number)
            largest = max(largest, first + 1, second + 1)
            written += 1
            if first != second:
                edges.add((min(first, second), max(first, second)))
            continue

        raise InstanceError(
            "line {}: {!r} starts no line a DIMACS edge list has; it holds "
            "only c, p and e lines".format(number, words[0])
        )

    if declared_nodes is None or declared_edges is None:
        raise InstanceError(
            "no problem line: the file states no 'p edge <nodes> <edges>' "
            "line anywhere, so it is not a DIMACS edge list"
        )

    _believe("edge", declared_edges, written, problem_at, True)
    _believe("node", declared_nodes, largest, problem_at, True)

    return Graph(
        name=name,
        source=source,
        layout="dimacs-edge",
        vertices=max(declared_nodes, largest),
        edges=tuple(sorted(edges)),
    )


def read_graph(path: Union[str, Path]) -> Graph:
    """Read an edge-list file, naming it after the file it came from."""
    location = Path(path).expanduser()
    return parse_graph(_text(location), location.stem, str(location))


# ---------------------------------------------------------------------------
# the pieces both readers share
# ---------------------------------------------------------------------------

def _text(location: Path) -> str:
    """The file as text, or an :class:`~. InstanceError` saying why not."""
    try:
        return location.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))


def _problem(
    words: List[str],
    number: int,
    expected: Tuple[str, ...],
    noun: str,
    what: str,
    other: str,
) -> Tuple[int, int]:
    """The two counts off a problem line, or a refusal naming the line.

    A problem line spelled for the *other* DIMACS format is refused by name and
    told which reader it wanted, because that is the mistake people make and
    "expected 4 fields" would not help them.
    """
    shape = "'p {} <nodes> <{}s>'".format(expected[0], noun)
    if len(words) != 4:
        raise InstanceError(
            "line {}: a problem line reads {}, but this has {} fields".format(
                number, shape, len(words)
            )
        )
    stated = words[1].lower()
    if stated in expected:
        return (
            _count(words[2], number, "node count"),
            _count(words[3], number, "{} count".format(noun)),
        )
    if stated in (_EDGE_WORDS + _FLOW_WORDS):
        raise InstanceError(
            "line {}: 'p {}' is not a {} problem line; this file wants "
            "{}".format(number, words[1], what, other)
        )
    raise InstanceError(
        "line {}: 'p {}' names no DIMACS format this reads; a {} file states "
        "{}".format(number, words[1], what, shape)
    )


def _count(word: str, number: int, what: str) -> int:
    """A declared count: a non-negative integer, or a refusal."""
    try:
        value = int(word)
    except ValueError:
        raise InstanceError(
            "line {}: the {} {!r} is not an integer".format(number, what, word)
        )
    if value < 0:
        raise InstanceError(
            "line {}: the {} is {}, which counts nothing".format(
                number, what, value
            )
        )
    return value


def _node(word: str, number: int) -> int:
    """A node id, one-based in the file and zero-based on the way out."""
    try:
        value = int(word)
    except ValueError:
        raise InstanceError(
            "line {}: the node id {!r} is not an integer".format(number, word)
        )
    if value < 1:
        raise InstanceError(
            "line {}: node {} is out of range; DIMACS numbers nodes from 1, "
            "so a 0 here means the file uses another convention and reading "
            "it as written would shift every vertex".format(number, value)
        )
    return value - 1


def _capacity(word: str, number: int) -> float:
    """An arc capacity: non-negative, integral in the specification but taken
    as a float, since generators write both and no cost formula here cares."""
    try:
        value = float(word)
    except ValueError:
        raise InstanceError(
            "line {}: the capacity {!r} is not a number".format(number, word)
        )
    if value < 0:
        raise InstanceError(
            "line {}: the capacity is {}; a DIMACS capacity is not "
            "negative".format(number, word)
        )
    return value


def _believe(
    what: str, declared: int, seen: int, at: int, undercount_matters: bool
) -> None:
    """Refuse a header that cannot be a stale count of *this* file.

    Small disagreements are the normal state of collected DIMACS files and are
    resolved by trusting the data.  An order of magnitude is not a stale count;
    see the module docstring for why that is refused instead.

    Both directions are checked, for node counts as well as arc counts.  A
    header may legitimately declare more nodes than any arc mentions -- isolated
    vertices exist -- so this direction was once left unchecked, and a one-arc
    file declaring a billion nodes was read as a billion-vertex network.  That
    is not a wrong number so much as an unusable one: the vertex count is the
    search space of every quantum breadth-first sweep, and the classical run
    tries to allocate an adjacency list per vertex before anything is costed.
    A tenfold margin leaves room for a graph that really is mostly isolated.
    """
    if seen > declared * _TOLERANCE and seen > _TOLERANCE:
        raise InstanceError(
            "line {}: the problem line declares {} {}s but the file holds {}; "
            "that is too far apart to be a stale header, so this is either a "
            "truncated file or not the format it claims".format(
                at, declared, what, seen
            )
        )
    if undercount_matters and declared > max(seen, 1) * _TOLERANCE:
        raise InstanceError(
            "line {}: the problem line declares {} {}s but the file holds only "
            "{}; that is too far apart to be a stale header, so this is either "
            "a truncated file or not the format it claims".format(
                at, declared, what, seen
            )
        )
