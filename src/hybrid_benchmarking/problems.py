"""Problems, and the routes through them.

Someone with a real problem does not have a subroutine; they have a network, or
a set of items and a budget.  This module is the layer between the two: it says
what problems the library can cost, which classical algorithm solves each, which
step of that algorithm a quantum routine replaces, and what data has to be
logged to cost it.

The list is not a taxonomy anyone invented.  A problem belongs here only if
there is a chain of lemmas that can cost it, so it is read off the datasets the
underlying work actually benchmarks -- which is also why maximum cut is absent
and maximum flow appears three times.

**Routes are not comparable with each other.**  The quantum-search route through
Dinic counts a fixed schedule; the simplex route is a lower bound with
non-leading terms dropped; the interior point route is a cycle bound resting on
a chain of deliberately benevolent assumptions.  Three numbers in a row would
imply a comparison none of the three sources supports, so every route carries
its provenance and the interface is expected to show it rather than tabulate
bare values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from .provenance import Unit

Shape = Callable[[Dict[str, float]], Dict[str, float]]


@dataclass(frozen=True)
class Field:
    """One quantity someone has to supply, named as they would name it."""

    name: str
    label: str
    help: str
    example: str = ""


@dataclass(frozen=True)
class Route:
    """One way of attacking a problem, and what it costs."""

    key: str
    label: str
    classical: str
    replaces: str
    target: str
    unit: Unit
    #: Logged once per iteration, phase or system -- the output of the
    #: instrumented classical run.
    per_record: Tuple[Field, ...] = ()
    #: True of the instance as a whole.
    per_instance: Tuple[Field, ...] = ()
    #: Chosen by whoever is asking, not measured.
    chosen: Tuple[Field, ...] = ()
    #: Turns problem shape into the shape of the linear program it becomes.
    shape: Optional[Shape] = None
    #: Some costs take a whole run at once rather than one record at a time.
    #: ``("layers", "phases")`` means: gather the ``layers`` field of every
    #: record into a list and pass it as ``phases``.
    collects: Optional[Tuple[str, str]] = None
    #: Log field name to cost parameter name, where the two differ.  Someone
    #: logging a graph writes "vertices"; the lemma calls it the length of the
    #: list being searched.  Neither should have to yield to the other.
    renames: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    def describe(self) -> str:
        return "{} solves this classically; the quantum version replaces {}." \
            .format(self.classical, self.replaces)


@dataclass(frozen=True)
class Problem:
    """A real task, under the name someone would recognise."""

    key: str
    label: str
    technical: str
    blurb: str
    routes: Tuple[Route, ...]


# ---------------------------------------------------------------------------
# shapes: problem data to linear-program data
# ---------------------------------------------------------------------------

def _pairs(vertices: float) -> float:
    return vertices * (vertices - 1) / 2


# The simplex works on the standard form ``Ax = b, x >= 0``, not on the
# formulation as written.  Getting from one to the other adds variables: a
# surplus or slack for every inequality, and one more for every variable with an
# upper bound.  It is not bookkeeping -- written naively, vertex cover has one
# variable per vertex and one constraint per edge, so any graph with more edges
# than vertices would ask the simplex for a basis larger than the number of
# columns, which is not a program at all.  The library refuses that, which is
# how the omission shows up rather than quietly producing a number.


def cover_shape(data: Dict[str, float]) -> Dict[str, float]:
    """Vertex cover and independent set: one constraint per edge.

    ``x_u + x_v >= 1`` for every edge and ``x_v <= 1`` for every vertex.  In
    standard form that is one surplus per edge and one slack per bound, so the
    program has ``2V + E`` columns and ``V + E`` rows.
    """
    vertices, edges = data["vertices"], data["edges"]
    return {"n": 2 * vertices + edges, "m": vertices + edges}


def clique_shape(data: Dict[str, float]) -> Dict[str, float]:
    """Maximum clique: one constraint per *non*-edge.

    Sparse graphs give dense programs -- the constraint count grows with the
    square of the vertex count -- which is why clique instances are the awkward
    ones in this family.
    """
    vertices, edges = data["vertices"], data["edges"]
    absent = _pairs(vertices) - edges
    return {"n": 2 * vertices + absent, "m": vertices + absent}


def flow_shape(data: Dict[str, float]) -> Dict[str, float]:
    """Maximum flow: one variable per edge, and one more for the flow itself.

    Conservation is an equality at every vertex and needs no slack; each
    capacity is an inequality and takes one.  That gives ``V + E`` rows and, on
    the face of it, ``2E`` columns.

    It is ``2E + 1``.  Conservation at *every* vertex, over the edge variables
    alone, forces the net flow out of the source to zero -- such a program can
    express circulations and nothing else, so its maximum flow is always zero
    whatever the network.  What makes it a maximum-flow program is one further
    column: the flow value, entering the source's row and the sink's row with
    opposite signs and carrying the whole objective.  Writing it instead as an
    uncapacitated return arc from sink to source gives the same count, since
    that arc takes a column and, being uncapacitated, no row and no slack.

    Both readings agree, so this is a correction rather than a choice.  It moves
    ``n`` by one, and so moves the ``n - m`` non-basic count that the optimality
    test and the pivoting rule search over.
    """
    vertices, edges = data["vertices"], data["edges"]
    return {"n": 2 * edges + 1, "m": vertices + edges}


# ---------------------------------------------------------------------------
# the fields that recur
# ---------------------------------------------------------------------------

GRAPH = (
    Field("vertices", "Vertices", "How many nodes the network has", "1000"),
    Field("edges", "Edges", "How many connections between them", "5000"),
)

PRECISION = (
    Field("epsilon", "Precision", "How exactly each step must be resolved. "
          "Costs scale inversely, so this choice matters more than it looks",
          "1e-3"),
    Field("delta", "Tolerance", "Tolerance of the surrounding classical "
          "algorithm", "1e-3"),
)

LP_SHAPE = (
    Field("n", "Variables", "Columns in the linear program", "200"),
    Field("m", "Constraints", "Rows in the linear program", "50"),
)

SIMPLEX_RECORD = (
    Field("kappa", "Condition number", "Of the basis at this iteration", "10"),
    Field("d", "Sparsity", "Most non-zeros in any row or column", "4"),
    Field("A_1", "1-norm", "Largest absolute column sum of the basis", "3"),
    Field("A_max", "Largest entry", "Largest absolute entry of the basis", "1"),
    # Lemma 10 defines this one, and defines it as the number of positive
    # components of u = A_B^-1 A_k.  It is not the improving-column count: that
    # marks Lemma 24's search over the n - m non-basic columns, it routinely
    # exceeds m, and here it would mark more elements than the list being
    # searched holds -- which the validity domain refuses outright.  Under the
    # steepest-edge rule this route targets, nothing else consumes t at all.
    Field("t", "Positive pivot components",
          "How many components of the pivot direction are positive. Lemma 10 "
          "searches the basis for one of them, so this is a count out of m -- "
          "not the number of columns that could enter", "5"),
    Field("c_max", "Largest cost", "Largest absolute objective coefficient", "2"),
    Field("u_norm", "Pivot norm",
          "Norm of the pivot direction of the entering column, ||A_B^-1 A_k||",
          "1.5"),
)

# Not every solver reads every one of these.  Chebyshev reaches the matrix
# through a quantum walk and the singular value transformation through a block
# encoding, so neither ever sees the largest entry; only the two that simulate
# the matrix -- HHL and Fourier, both through qubitization -- pay for it.  Asking
# for it anyway would refuse a log that is complete for the route it was written
# for, which is the opposite of what checking a log is for.
SOLVER_RECORD = (
    Field("kappa", "Condition number", "Of this system", "10"),
    Field("d", "Sparsity", "Most non-zeros in any row", "4"),
    Field("x_norm", "Solution norm", "Norm of the solution vector", "1"),
)

SIMULATED_ENTRY = (
    Field("A_max", "Largest entry", "Largest absolute entry. Needed only by "
          "the solvers that simulate the matrix, which is where it enters", "1"),
)

NEWTON_RECORD = (
    Field("N", "Dimension", "Unknowns in this Newton system", "1000"),
    Field("d", "Sparsity", "Most non-zeros in any row", "50"),
    Field("kappa", "Condition number", "Of this Newton system", "100"),
)


# ---------------------------------------------------------------------------
# routes shared by every problem that becomes a linear program
# ---------------------------------------------------------------------------

def _simplex_route(shape: Shape, rule: str = "steepest-edge") -> Route:
    return Route(
        key="quantum-simplex",
        label="Quantum simplex",
        classical="The simplex method",
        replaces="the pivoting step -- choosing which variable enters the "
                 "basis -- with quantum search over the candidate columns",
        target="SimplexIter/" + rule,
        unit=Unit.GATES,
        per_record=SIMPLEX_RECORD,
        per_instance=GRAPH if shape else LP_SHAPE,
        chosen=PRECISION,
        shape=shape,
        note="A lower bound: non-leading terms are dropped and every "
             "assumption favours the quantum side, so a real run costs more.",
    )


def _ipm_route() -> Route:
    # No problem shape here, unlike the simplex route beside it.  A Newton
    # system states its own dimension, which is what the cost reads; the
    # program's column and row counts never reach it.  Declaring them anyway
    # would make every log carry a vertex count for nothing, and would have the
    # route predict an ``m`` that is not the dimension solved -- an interior
    # point method presolves redundant rows away, and a maximum-flow model has
    # one by construction.
    return Route(
        key="quantum-interior-point",
        label="Quantum interior point",
        classical="A primal-dual interior point method",
        replaces="the Newton system solve at each step with a quantum linear "
                 "solver, then reads the answer back out",
        target="IPM/mnes",
        unit=Unit.CYCLES,
        per_record=NEWTON_RECORD,
        chosen=(PRECISION[0],),
        note="A lower bound under deliberately generous assumptions: one "
             "oracle call per cycle, no amplification overhead, convergence in "
             "a single step. The readout dominates.",
    )


def _lp_routes(shape: Shape) -> Tuple[Route, ...]:
    return (_simplex_route(shape), _ipm_route())


# ---------------------------------------------------------------------------
# the problems
# ---------------------------------------------------------------------------

PROBLEMS: Tuple[Problem, ...] = (
    Problem(
        key="maximum-flow",
        label="Routing as much as possible through a network",
        technical="Maximum flow",
        blurb="Pipes, cables, roads or shipping lanes with limited capacity: "
              "how much can travel from a source to a destination at once? "
              "It also turns up inside scheduling and matching problems.",
        routes=(
            Route(
                key="quantum-bfs",
                label="Quantum breadth-first search",
                classical="Dinic's algorithm",
                replaces="the breadth-first sweep that layers the network, "
                         "with quantum search for each layer's vertices",
                target="Dinic",
                unit=Unit.GATES,
                collects=("layers", "phases"),
                renames={"vertices": "X"},
                per_instance=(GRAPH[0],),
                per_record=(
                    Field("layers", "Layer sizes",
                          "Vertices in each layer of this sweep, as a list",
                          "[1, 3, 5, 2]"),
                ),
                note="The circuit is explicit enough to schedule, so the "
                     "derivation counts cycles; the gate count follows by "
                     "assuming one cycle costs one gate. That assumption is "
                     "generous -- a cycle usually holds many gates.",
            ),
            _simplex_route(flow_shape),
            _ipm_route(),
        ),
    ),
    Problem(
        key="vertex-cover",
        label="Covering a network with as few sites as possible",
        technical="Minimum vertex cover",
        blurb="Placing cameras, sensors or guards so that every connection is "
              "watched by at least one of them, using as few as you can.",
        routes=_lp_routes(cover_shape),
    ),
    Problem(
        key="independent-set",
        label="Choosing as many mutually compatible items as possible",
        technical="Maximum independent set",
        blurb="Picking the largest group of things that do not conflict -- "
              "non-interfering transmitters, non-clashing bookings, "
              "compatible machines.",
        routes=_lp_routes(cover_shape),
    ),
    Problem(
        key="clique",
        label="Finding the largest fully connected group",
        technical="Maximum clique",
        blurb="The biggest set of items that are all pairwise related: a "
              "tightly knit community, a set of mutually compatible parts.",
        routes=_lp_routes(clique_shape),
    ),
    Problem(
        key="linear-programming",
        label="Allocating limited resources",
        technical="Linear programming",
        blurb="Production planning, blending, logistics, staffing -- anything "
              "where a linear objective meets linear constraints. If you "
              "already have a model in MPS or LP format, this is your entry.",
        routes=_lp_routes(None),
    ),
    Problem(
        key="knapsack",
        label="Packing the most value under a budget",
        technical="0-1 knapsack",
        blurb="Choosing which items to take when each has a value and a cost "
              "and the total cost is capped: cargo loading, project "
              "selection, capital budgeting.",
        routes=(
            Route(
                key="tree-generator",
                label="Quantum tree generator search",
                classical="COMBO, the state-of-the-art dynamic programming "
                          "solver",
                replaces="the whole search, with a quantum state that "
                         "superposes feasible packings weighted towards "
                         "profitable ones, amplified towards the best",
                target="QTG",
                unit=Unit.CYCLES,
                per_instance=(
                    Field("profits", "Values",
                          "What each item is worth, as a list", "[6, 2, 1, 2]"),
                    Field("weights", "Costs",
                          "What each item costs, as a list", "[2, 2, 1, 5]"),
                    Field("capacity", "Budget",
                          "Total cost you can afford", "7"),
                    Field("profit_bound", "Value ceiling",
                          "Any upper bound on the best achievable value", "11"),
                ),
                note="The only route here whose cost depends on the actual "
                     "numbers rather than on problem size: where the ones sit "
                     "in each value's binary representation changes the "
                     "circuit.",
            ),
        ),
    ),
    Problem(
        key="linear-systems",
        label="Solving a large system of equations, or a physical field",
        technical="Linear systems and discretised PDEs",
        blurb="Steady-state temperature, electrostatic potential, structural "
              "load -- anything that discretises into a large sparse system. "
              "Also the step every method above outsources to a solver.",
        routes=tuple(
            Route(
                key=key,
                label=label,
                classical="A direct factorisation, or an iterative method "
                          "such as conjugate gradient",
                replaces="the whole solve, preparing the solution as a "
                         "quantum state rather than as numbers",
                target=target,
                unit=Unit.QUERIES,
                per_record=SOLVER_RECORD + (SIMULATED_ENTRY if simulates else ()),
                chosen=(PRECISION[0],),
                note=note,
            )
            for key, label, target, simulates, note in (
                ("qsvt", "Singular value transformation", "QLS-QSVT", False,
                 "Lowest query count across every dataset in the comparison."),
                ("chebyshev", "Chebyshev polynomials", "QLS-Chebyshev", False,
                 "Close behind, and the one the interior point work adopts."),
                ("fourier", "Fourier series", "QLS-Fourier/via-qubitization",
                 True,
                 "About an order of magnitude dearer, but simpler to build."),
                ("hhl", "HHL", "HHL", True,
                 "The famous one, and orders of magnitude more expensive: its "
                 "cost scales as one over the precision where the others "
                 "scale as its logarithm."),
            )
        ),
    ),
)


_BY_KEY = {problem.key: problem for problem in PROBLEMS}


def get_problem(key: str) -> Problem:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError("no problem {!r}; known: {}".format(
            key, ", ".join(sorted(_BY_KEY))
        ))


def get_route(problem_key: str, route_key: str) -> Route:
    problem = get_problem(problem_key)
    for route in problem.routes:
        if route.key == route_key:
            return route
    raise KeyError("{} has no route {!r}; it has {}".format(
        problem_key, route_key, ", ".join(r.key for r in problem.routes)
    ))
