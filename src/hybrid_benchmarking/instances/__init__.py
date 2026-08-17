"""Reading the files people actually have.

The library asks for condition numbers, layer sizes and improving-column counts.
Nobody has those.  What they have is a file: a DIMACS network, a knapsack
instance from Pisinger's generator, a matrix from Matrix Market, a model in MPS.
This package turns such a file into a plain description of the instance, and
:mod:`..classical` runs the classical algorithm on it to produce the log.

The readers hold no cost logic and import nothing from the rest of the library,
so a wrong count here cannot become a plausible gate count elsewhere -- it
becomes a graph with the wrong number of edges, which the tests catch.  They are
standard-library only: numpy enters one step later, with the solvers.

Every reader offers the same two entry points, ``parse(text, name, source)`` and
``read(path)``, and raises :class:`InstanceError` on anything it cannot read
rather than guessing.  Guessing is the failure mode that matters here: a
knapsack layout misread by one column still produces numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

__all__ = [
    "Graph",
    "Instance",
    "InstanceError",
    "Knapsack",
    "LinearProgram",
    "Matrix",
    "Network",
    "detect",
    "read",
]


class InstanceError(ValueError):
    """The file is not what it claims to be, or not something we can read."""


# ---------------------------------------------------------------------------
# what a file can turn out to hold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """Common to everything read from disk."""

    #: Usually the file stem, or the name the format states internally.
    name: str
    #: Where it was read from, for the log to cite.
    source: str
    #: Which reader produced it: ``dimacs-max``, ``dimacs-edge``, ``pisinger``,
    #: ``matrix-market``, ``mps``.
    layout: str

    def describe(self) -> str:
        return "{} ({})".format(self.name, self.layout)


@dataclass(frozen=True)
class Network(Instance):
    """A directed graph with arc capacities, a source and a sink.

    Vertices are numbered from zero, whatever the file did.  Parallel arcs are
    kept as they are written: a maximum-flow solver is entitled to see them.
    """

    vertices: int
    #: ``(tail, head, capacity)``, vertices zero-based.
    arcs: Tuple[Tuple[int, int, float], ...]
    #: Where the flow starts and ends, zero-based.  Named at length because
    #: :attr:`Instance.source` is the file this came from, and a dataclass field
    #: called ``source`` on both would silently be one field.
    source_vertex: int = 0
    sink_vertex: int = 0

    def describe(self) -> str:
        return "{}: {} vertices, {} arcs, {} to {}".format(
            self.name, self.vertices, len(self.arcs),
            self.source_vertex, self.sink_vertex,
        )


@dataclass(frozen=True)
class Graph(Instance):
    """An undirected graph, without weights.

    Edges are normalised -- ``u < v``, no self-loops, no duplicates -- because
    the linear programs built from them have one constraint per edge, and a
    duplicate edge would silently add a redundant row.
    """

    vertices: int
    #: ``(u, v)`` with ``u < v``, vertices zero-based, sorted.
    edges: Tuple[Tuple[int, int], ...]

    def describe(self) -> str:
        return "{}: {} vertices, {} edges".format(
            self.name, self.vertices, len(self.edges)
        )


@dataclass(frozen=True)
class Knapsack(Instance):
    """Items with a value and a cost, and a budget.

    Profits and weights are integers because the circuits that cost this
    problem read their binary representations: where the ones sit in each value
    changes the gate count, so a float here would not merely be imprecise, it
    would be meaningless.
    """

    profits: Tuple[int, ...]
    weights: Tuple[int, ...]
    capacity: int
    #: The optimal value, where the file states one.  Some generators record it.
    optimum: Optional[int] = None

    def describe(self) -> str:
        return "{}: {} items, capacity {}".format(
            self.name, len(self.profits), self.capacity
        )


@dataclass(frozen=True)
class Matrix(Instance):
    """A sparse matrix, as Matrix Market states it.

    Symmetric files store one triangle; :attr:`entries` holds both, so that
    anything counting non-zeros per row gets the right answer without having to
    remember what the header said.
    """

    rows: int
    columns: int
    #: ``(row, column, value)``, zero-based, both triangles present.
    entries: Tuple[Tuple[int, int, float], ...]
    symmetric: bool = False

    def describe(self) -> str:
        return "{}: {} by {}, {} non-zeros{}".format(
            self.name, self.rows, self.columns, len(self.entries),
            ", symmetric" if self.symmetric else "",
        )


@dataclass(frozen=True)
class LinearProgram(Instance):
    """A linear program in the shape an MPS file states it.

    Deliberately not the standard form: the translation to ``Ax = b, x >= 0``
    adds columns, and doing it here would lose the distinction between a bound
    the file stated and a slack the translation invented.  :mod:`..classical.lp`
    does that step, and is tested against the shapes :mod:`..problems` predicts.
    """

    #: Column names, in file order.
    columns: Tuple[str, ...]
    #: Constraint row names, objective excluded, in file order.
    rows: Tuple[str, ...]
    #: ``'L'``, ``'G'`` or ``'E'`` per row.
    senses: Tuple[str, ...]
    #: Objective coefficients, per column.
    objective: Tuple[float, ...]
    #: ``(row index, column index, value)``, zero-based on :attr:`rows` and
    #: :attr:`columns`.
    matrix: Tuple[Tuple[int, int, float], ...]
    #: Right-hand side, per row; zero where the file gives none.
    rhs: Tuple[float, ...]
    #: Per row: the RANGES entry, or ``None``.  A range turns one row into two
    #: bounds on the same expression.
    ranges: Tuple[Optional[float], ...] = ()
    #: Lower bound per column, zero by default.
    lower: Tuple[float, ...] = ()
    #: Upper bound per column, ``None`` for infinite.
    upper: Tuple[Optional[float], ...] = ()
    #: True where an INTORG/INTEND marker declared the column integer.  The
    #: relaxation is what gets solved; the flag is kept so the log can say so.
    integer: Tuple[bool, ...] = ()
    maximise: bool = False
    objective_name: str = ""
    objective_constant: float = 0.0

    def describe(self) -> str:
        return "{}: {} columns, {} rows{}".format(
            self.name, len(self.columns), len(self.rows),
            ", relaxed from an integer program" if any(self.integer) else "",
        )


# ---------------------------------------------------------------------------
# working out what a file is
# ---------------------------------------------------------------------------

#: Extension to layout, where the extension is decisive.
_BY_SUFFIX = {
    ".max": "dimacs-max",
    ".clq": "dimacs-edge",
    ".col": "dimacs-edge",
    ".edges": "dimacs-edge",
    ".mtx": "matrix-market",
    ".mps": "mps",
    ".kp": "pisinger",
}


def detect(path: Union[str, Path]) -> str:
    """Which reader this file wants, from its name and its first few lines.

    The extension decides where it can; otherwise the header does.  Where
    neither does, this raises rather than picking the most likely one -- a
    knapsack file read as an edge list produces a graph, not an error.
    """
    location = Path(path).expanduser()
    suffix = location.suffix.lower()
    if suffix in _BY_SUFFIX:
        return _BY_SUFFIX[suffix]

    try:
        head = location.open("r", errors="replace").read(4096)
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))

    lowered = head.lstrip().lower()
    if lowered.startswith("%%matrixmarket"):
        return "matrix-market"

    for line in head.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        head_word = words[0].upper()
        if head_word in ("NAME", "OBJSENSE", "ROWS") and len(words) <= 3:
            return "mps"
        if head_word == "P" and len(words) >= 2:
            what = words[1].lower()
            if what in ("max", "flow"):
                return "dimacs-max"
            if what in ("edge", "edges", "col", "clique"):
                return "dimacs-edge"
        if head_word == "N" and len(words) == 2:
            return "pisinger"
        if stripped.startswith("c ") or stripped.startswith("*"):
            continue  # a comment in several of these formats; keep looking

    raise InstanceError(
        "cannot tell what {} is; name it .max, .clq, .mtx, .mps or .kp, or "
        "pass the format explicitly".format(location.name)
    )


def read(path: Union[str, Path], layout: str = "") -> Instance:
    """Read an instance file, choosing the reader by :func:`detect`."""
    location = Path(path).expanduser()
    if not location.exists():
        raise InstanceError("no such file: {}".format(location))
    chosen = layout or detect(location)

    if chosen == "dimacs-max":
        from .dimacs import read_max_flow

        return read_max_flow(location)
    if chosen == "dimacs-edge":
        from .dimacs import read_graph

        return read_graph(location)
    if chosen == "pisinger":
        from .knapsack import read as read_knapsack

        return read_knapsack(location)
    if chosen == "matrix-market":
        from .matrixmarket import read as read_matrix

        return read_matrix(location)
    if chosen == "mps":
        from .mps import read as read_mps

        return read_mps(location)
    raise InstanceError("no reader for {!r}".format(chosen))
