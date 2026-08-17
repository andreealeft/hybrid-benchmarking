"""Linear programs, and the standard form the simplex actually runs on.

The simplex works on ``min c'x`` subject to ``Ax = b, x >= 0``.  Almost nothing
arrives in that shape: a vertex cover relaxation has inequalities and upper
bounds, an MPS model has ranges and free variables and bounds of every kind.
Getting from one to the other adds columns, and how many it adds is not
bookkeeping -- it is what :mod:`..problems` predicts when it says a vertex cover
on ``V`` vertices and ``E`` edges becomes a program with ``2V + E`` columns and
``V + E`` rows.  Those predictions are what the route feeds the lemmas, so if
the translation here disagreed with them the logged condition numbers would
belong to one program and the column counts to another.  There is a test that
they agree.

Everything goes through one converter.  A model states its rows with senses and
its columns with bounds, and :func:`standard_form` does the same six things to
all of them whatever they came from:

*a finite lower bound is a shift*, ``x = y + l``, which moves the right-hand
side and leaves the column where it was;

*a finite upper bound is a row*, ``y + s = u - l``, which is where the slack per
bound in the predicted shapes comes from;

*a column bounded only above* becomes ``x = u - y``, reflecting the column
rather than adding one;

*a free column splits in two*, ``x = y⁺ - y⁻``, which is the one case that costs
a column without a row;

*a fixed column leaves altogether*, folded into the right-hand side, because a
variable that cannot move is not a variable the simplex should be searching
over;

*an inequality takes a slack or a surplus*, and then any row with a negative
right-hand side is negated, so that the artificial basis of the first phase is
the identity and feasible.

The matrix is dense.  These are instrumented runs on instances small enough to
watch, not a production solver, and the instrumentation itself -- an explicit
basis inverse every iteration -- is what fixes the scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..instances import Graph, LinearProgram, Network

#: Refuse rather than ask the machine for a matrix this big.  A dense standard
#: form of this many entries is already a third of a gigabyte, and the basis
#: inverse taken every iteration would be far past any budget anyway -- so
#: failing here with a reason beats swapping for five minutes and failing there.
MAX_ENTRIES = 25_000_000

INFINITE = float("inf")


class ModelError(ValueError):
    """The model cannot be put in standard form, and here is why."""


@dataclass
class Model:
    """A linear program as it is written, before any translation.

    Rows are sparse -- ``(column indices, values, sense, right-hand side)`` --
    because a vertex cover row touches two columns out of thousands.
    """

    columns: int
    objective: np.ndarray
    rows: List[Tuple[Sequence[int], Sequence[float], str, float]] = field(
        default_factory=list)
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    maximise: bool = False
    name: str = ""

    def __post_init__(self):
        self.objective = np.asarray(self.objective, dtype=float)
        if self.lower is None:
            self.lower = np.zeros(self.columns)
        if self.upper is None:
            self.upper = np.full(self.columns, INFINITE)

    def add_row(self, indices: Sequence[int], values: Sequence[float],
                sense: str, rhs: float) -> None:
        self.rows.append((indices, values, sense, rhs))


@dataclass(frozen=True)
class StandardForm:
    """``min c'x`` subject to ``Ax = b``, ``x >= 0``, ``b >= 0``."""

    matrix: np.ndarray
    rhs: np.ndarray
    objective: np.ndarray
    name: str = ""
    #: Added to the objective value to undo the shifts and reflections above.
    constant: float = 0.0
    #: What each column of :attr:`matrix` was, for a log to be readable.
    origin: Tuple[str, ...] = ()
    #: ``-1`` where the model was a maximisation, which the simplex only ever
    #: sees as a negated objective.  Applied when the value is reported, so
    #: that "objective 2.5" means what the person who wrote the model meant.
    sign: float = 1.0

    def value_of(self, minimised: float) -> float:
        """The objective in the units the model was written in."""
        return self.sign * (minimised + self.constant)

    @property
    def rows(self) -> int:
        return self.matrix.shape[0]

    @property
    def columns(self) -> int:
        return self.matrix.shape[1]

    @property
    def shape(self) -> Dict[str, int]:
        """``n`` and ``m`` as the lemmas name them."""
        return {"n": self.columns, "m": self.rows}


# ---------------------------------------------------------------------------
# the converter
# ---------------------------------------------------------------------------

def standard_form(model: Model) -> StandardForm:
    """Translate a model into the shape the simplex runs on."""
    lower, upper = np.asarray(model.lower, float), np.asarray(model.upper, float)
    objective = np.asarray(model.objective, float)
    if model.maximise:
        objective = -objective

    # Each original column becomes zero, one or two standard-form columns.  The
    # map records how, so rows can be rewritten in one pass afterwards.
    replacement: List[Tuple[str, Tuple[int, ...], Tuple[float, ...], float]] = []
    cost: List[float] = []
    origin: List[str] = []
    constant = 0.0
    bounded: List[Tuple[int, float]] = []  # column, width, for the extra rows

    for j in range(model.columns):
        low, high = float(lower[j]), float(upper[j])
        if low > high:
            raise ModelError(
                "column {} has a lower bound of {} above its upper bound of {}"
                .format(j, low, high)
            )
        if low == high and low != INFINITE and low != -INFINITE:
            replacement.append(("fixed", (), (), low))
            constant += objective[j] * low
            continue
        if low > -INFINITE:
            index = len(cost)
            replacement.append(("shift", (index,), (1.0,), low))
            cost.append(objective[j])
            origin.append("x{}".format(j))
            constant += objective[j] * low
            if high < INFINITE:
                bounded.append((index, high - low))
        elif high < INFINITE:
            index = len(cost)
            replacement.append(("reflect", (index,), (-1.0,), high))
            cost.append(-objective[j])
            origin.append("-x{}".format(j))
            constant += objective[j] * high
        else:
            plus, minus = len(cost), len(cost) + 1
            replacement.append(("split", (plus, minus), (1.0, -1.0), 0.0))
            cost.extend((objective[j], -objective[j]))
            origin.extend(("x{}+".format(j), "x{}-".format(j)))

    rows: List[Tuple[Dict[int, float], str, float]] = []
    for indices, values, sense, rhs in model.rows:
        if sense not in ("L", "G", "E"):
            raise ModelError("a row cannot have sense {!r}".format(sense))
        coefficients: Dict[int, float] = {}
        shifted = float(rhs)
        for column, value in zip(indices, values):
            kind, targets, signs, offset = replacement[column]
            shifted -= value * offset
            for target, sign in zip(targets, signs):
                coefficients[target] = coefficients.get(target, 0.0) + value * sign
        rows.append((coefficients, sense, shifted))

    # An upper bound is a row, and it is added after the model's own rows so
    # that the predicted shapes -- one constraint per edge, then one bound per
    # vertex -- come out in that order.
    for index, width in bounded:
        rows.append(({index: 1.0}, "L", width))

    total_columns = len(cost) + sum(1 for _, sense, _ in rows if sense != "E")
    if len(rows) * total_columns > MAX_ENTRIES:
        raise ModelError(
            "the standard form is {} by {}, which is larger than this tool "
            "carries a dense basis for; a longer budget will not help"
            .format(len(rows), total_columns)
        )

    matrix = np.zeros((len(rows), total_columns))
    rhs = np.zeros(len(rows))
    objective_row = np.zeros(total_columns)
    objective_row[:len(cost)] = cost
    names = list(origin)

    next_slack = len(cost)
    for row, (coefficients, sense, value) in enumerate(rows):
        for column, coefficient in coefficients.items():
            matrix[row, column] = coefficient
        rhs[row] = value
        if sense != "E":
            matrix[row, next_slack] = 1.0 if sense == "L" else -1.0
            names.append("s{}".format(row))
            next_slack += 1

    # The first phase starts from an artificial identity basis, which is only
    # feasible when every right-hand side is non-negative.
    negative = rhs < 0
    matrix[negative] *= -1.0
    rhs[negative] *= -1.0

    return StandardForm(
        matrix=matrix, rhs=rhs, objective=objective_row, name=model.name,
        constant=constant, origin=tuple(names),
        sign=-1.0 if model.maximise else 1.0,
    )


# ---------------------------------------------------------------------------
# the models the problems ask for
# ---------------------------------------------------------------------------

def _relaxation(graph: Graph, sense: str, maximise: bool) -> Model:
    """One row per edge, one unit upper bound per vertex.

    Vertex cover and independent set differ only in the direction of the edge
    constraint and of the objective; writing them once is what makes the
    identical predicted shape in :mod:`..problems` an observation rather than a
    coincidence.
    """
    model = Model(
        columns=graph.vertices,
        objective=np.ones(graph.vertices),
        upper=np.ones(graph.vertices),
        maximise=maximise,
        name=graph.name,
    )
    for u, v in graph.edges:
        model.add_row((u, v), (1.0, 1.0), sense, 1.0)
    return model


def vertex_cover(graph: Graph) -> Model:
    """``min sum x_v`` subject to ``x_u + x_v >= 1`` on every edge."""
    return _relaxation(graph, "G", maximise=False)


def independent_set(graph: Graph) -> Model:
    """``max sum x_v`` subject to ``x_u + x_v <= 1`` on every edge."""
    return _relaxation(graph, "L", maximise=True)


def clique(graph: Graph) -> Model:
    """``max sum x_v`` subject to ``x_u + x_v <= 1`` on every *non*-edge.

    Sparse graphs give dense programs here: the row count grows with the square
    of the vertex count, which is why clique is the awkward member of the
    family and why an instance that reads fine can still be refused for size.
    """
    present = set(graph.edges)
    model = Model(
        columns=graph.vertices,
        objective=np.ones(graph.vertices),
        upper=np.ones(graph.vertices),
        maximise=True,
        name=graph.name,
    )
    for u in range(graph.vertices):
        for v in range(u + 1, graph.vertices):
            if (u, v) not in present:
                model.add_row((u, v), (1.0, 1.0), "L", 1.0)
    return model


def maximum_flow(network: Network) -> Model:
    """``max F`` subject to conservation everywhere and capacity on every arc.

    The flow value is a column of its own, entering the source's conservation
    row and the sink's with opposite signs.  Without it, conservation at every
    vertex over the arc variables alone forces the net flow out of the source
    to zero and the program can only express circulations -- see
    :func:`~..problems.flow_shape`, which says the same thing from the other
    side.

    Capacities are written as rows rather than as bounds on the columns.  The
    two are the same program, but a zero-capacity arc -- which DIMACS permits
    and files contain -- has equal bounds, and the standard-form converter quite
    correctly folds a column that cannot move into the right-hand side and drops
    it.  That would make the program one column and one row smaller than the
    shape the route predicts, on some networks and not others.
    """
    if network.source_vertex == network.sink_vertex:
        raise ModelError(
            "the source and the sink are the same vertex, so the flow value is "
            "unconstrained and the program is unbounded"
        )

    arcs = network.arcs
    flow = len(arcs)  # the flow-value column sits after the arc columns
    model = Model(
        columns=len(arcs) + 1,
        objective=np.eye(1, len(arcs) + 1, flow)[0],
        maximise=True,
        name=network.name,
    )

    # One conservation row per vertex, in vertex order.
    for vertex in range(network.vertices):
        indices, values = [], []
        for index, (tail, head, _) in enumerate(arcs):
            weight = (1.0 if tail == vertex else 0.0) - (1.0 if head == vertex
                                                         else 0.0)
            if weight:
                indices.append(index)
                values.append(weight)
        if vertex == network.source_vertex:
            indices.append(flow)
            values.append(-1.0)
        elif vertex == network.sink_vertex:
            indices.append(flow)
            values.append(1.0)
        model.add_row(indices, values, "E", 0.0)

    for index, (_, _, capacity) in enumerate(arcs):
        model.add_row((index,), (1.0,), "L", float(capacity))
    return model


def from_linear_program(program: LinearProgram) -> Model:
    """An MPS model, with its ranges turned into bounded slacks.

    A range makes a row two-sided.  Rather than splitting it into two rows --
    which would change the row count the file states, and so the ``m`` every
    logged condition number is divided by -- the row becomes an equality and
    the two-sidedness moves into a bounded column, which the converter above
    then handles like any other bound.
    """
    columns = len(program.columns)
    objective = np.array(program.objective, dtype=float)
    lower = np.array(
        [-INFINITE if value is None else float(value) for value in program.lower]
        if program.lower else [0.0] * columns, dtype=float)
    upper = np.array(
        [INFINITE if value is None else float(value) for value in program.upper]
        if program.upper else [INFINITE] * columns, dtype=float)

    model = Model(columns=columns, objective=objective, lower=lower,
                  upper=upper, maximise=program.maximise, name=program.name)

    by_row: Dict[int, Tuple[List[int], List[float]]] = {}
    for row, column, value in program.matrix:
        indices, values = by_row.setdefault(row, ([], []))
        indices.append(column)
        values.append(value)

    extra: List[Tuple[int, float, float]] = []  # row, low, high for a range
    for row, sense in enumerate(program.senses):
        indices, values = by_row.get(row, ([], []))
        rhs = float(program.rhs[row]) if program.rhs else 0.0
        span = program.ranges[row] if program.ranges else None
        if span is None:
            model.add_row(indices, values, sense, rhs)
            continue
        low, high = _range_bounds(sense, rhs, float(span))
        model.add_row(indices, values, "E", low)
        extra.append((row, 0.0, high - low))

    # Each ranged row gains one column, bounded by the width of its range.
    if extra:
        widened = np.zeros(columns + len(extra))
        widened[:columns] = model.objective
        model.objective = widened
        model.lower = np.concatenate([model.lower, np.zeros(len(extra))])
        model.upper = np.concatenate(
            [model.upper, np.array([high for _, _, high in extra])])
        model.columns = columns + len(extra)
        for offset, (row, _, _) in enumerate(extra):
            indices, values, sense, rhs = model.rows[row]
            model.rows[row] = (
                list(indices) + [columns + offset], list(values) + [-1.0],
                sense, rhs,
            )
    return model


def _range_bounds(sense: str, rhs: float, span: float) -> Tuple[float, float]:
    """What a RANGES entry means, which depends on the row's own sense.

    The conventions are not symmetric and every solver implements the same
    asymmetry: on a ``G`` row the range extends upwards, on an ``L`` row
    downwards, and on an equality the sign of the range says which way.
    """
    width = abs(span)
    if sense == "G":
        return rhs, rhs + width
    if sense == "L":
        return rhs - width, rhs
    if span >= 0:
        return rhs, rhs + width
    return rhs - width, rhs


def model_for(instance, problem: str) -> Model:
    """The model a problem makes of an instance."""
    builders = {
        "vertex-cover": vertex_cover,
        "independent-set": independent_set,
        "clique": clique,
        "linear-programming": from_linear_program,
        "maximum-flow": maximum_flow,
    }
    if problem not in builders or builders[problem] is None:
        raise ModelError(
            "{} is not a problem this builds a linear program for; it builds "
            "{}".format(problem, ", ".join(
                key for key, value in builders.items() if value))
        )
    return builders[problem](instance)
