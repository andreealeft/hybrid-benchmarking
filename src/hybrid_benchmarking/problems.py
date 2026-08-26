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
    """A real task, under the name someone would recognise.

    Many of these are the same problem wearing different clothes.  Siting
    satellites and siting charging stations are both a quadratic knapsack, and
    someone who has one of them does not want to be told that before they can
    get a number.  So the *family* is what owns the routes, the shapes and the
    fields, and a problem is a name and a story attached to one -- which is why
    there are seventy of these and nine families.
    """

    key: str
    label: str
    technical: str
    blurb: str
    routes: Tuple[Route, ...]
    #: The problem underneath, which decides how it is solved and costed.
    family: str = ""


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
    # These four are not the raw quantities their names suggest, and a log
    # filled in with the raw ones is costed under a different normalisation
    # without anything failing. The conventions are the thesis's own; see
    # classical/simplex.py, which writes exactly these.
    Field("kappa", "Condition number",
          "Of the basis. GLPK's kappa_1 = ||A_B||_1 ||A_B^-1||_1 divided by m "
          "and floored at one, per (4.33): not the exact condition number",
          "10"),
    Field("d", "Sparsity",
          "Largest number of non-zeros in any row or column of the basis, "
          "whichever is larger", "4"),
    Field("A_1", "1-norm",
          "Of the normalised basis: ||A_B||_1 / (d ||A_B||_max), not the raw "
          "column sum", "1"),
    Field("A_max", "Largest entry",
          "Of the normalised basis: 1/d, not the raw largest entry", "0.25"),
    # Lemma 10 defines this one, and defines it as the number of positive
    # components of u = A_B^-1 A_k.  It is not the improving-column count: that
    # marks Lemma 24's search over the n - m non-basic columns, it routinely
    # exceeds m, and here it would mark more elements than the list being
    # searched holds -- which the validity domain refuses outright.  Under the
    # steepest-edge rule this route targets, nothing else consumes t at all.
    Field("t", "Positive pivot components",
          "How many components of the pivot direction are positive. Lemma 10 "
          "searches the basis for one of them, so this is a count out of m: "
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
    Field("d", "Sparsity",
          "Most non-zeros in any row or column, whichever is larger: the "
          "sparsity of the Hermitian matrix a quantum solver acts on", "4"),
    Field("x_norm", "Solution norm",
          "Norm of the solution, for a matrix scaled to unit spectral norm and "
          "a right-hand side of unit norm, so it lies between 1 and kappa", "1"),
)

SIMULATED_ENTRY = (
    Field("A_max", "Largest entry", "Largest absolute entry. Needed only by "
          "the solvers that simulate the matrix, which is where it enters", "1"),
)

NEWTON_RECORD = (
    Field("N", "Dimension", "Unknowns in this Newton system", "1000"),
    Field("d", "Sparsity",
          "Most non-zeros in any row or column of the system matrix", "50"),
    Field("kappa", "Condition number", "Of this Newton system", "100"),
)

#: The interior point analysis fixes its own precision rather than taking the
#: caller's.  Its Section IV-A assumes iterative refinement turns a solve at
#: 10^-1 into a high-precision answer in one step, and then puts 10^-1 into
#: equation (10) -- so that is the number the published bound is computed at,
#: and a tighter one here would not be the paper's figure.
IPM_PRECISION = (
    Field("epsilon", "Precision",
          "What each Newton system is solved to. The paper uses 1e-1 and "
          "assumes one step of iterative refinement does the rest; the readout "
          "cost scales as one over its square", "1e-1"),
)


KNAPSACK_INSTANCE = (
    Field("profits", "Values", "What each item is worth, as a list",
          "[6, 2, 1, 2]"),
    Field("weights", "Costs", "What each item costs, as a list", "[2, 2, 1, 5]"),
    Field("capacity", "Budget", "Total cost you can afford", "7"),
    Field("profit_bound", "Value ceiling",
          "Any upper bound on the best achievable value", "11"),
)

QUADRATIC_INSTANCE = KNAPSACK_INSTANCE + (
    Field("pair_profits", "Pair bonuses",
          "What each pair earns *in addition* when both are chosen, as "
          "{\"(i, j)\": value}. Bonuses only: a pair that costs you "
          "something when both are chosen is not something this circuit has a "
          "gate for", '{"(0, 1)": 6}'),
)

MULTIDIMENSIONAL_INSTANCE = (
    Field("profits", "Values", "What each item is worth, as a list",
          "[6, 2, 1, 2]"),
    Field("weights", "Costs",
          "What each item costs in each dimension: one list per dimension",
          "[[2, 2, 1, 5], [3, 1, 4, 2]]"),
    Field("capacities", "Budgets", "One budget per dimension, as a list",
          "[7, 6]"),
    Field("profit_bound", "Value ceiling",
          "Any upper bound on the best achievable value", "11"),
)


def knapsack_shape(data: Dict[str, float]) -> Dict[str, float]:
    """How many items there are, which nobody should have to count."""
    profits = data.get("profits") or []
    return {"items": len(profits)}


def multidimensional_shape(data: Dict[str, float]) -> Dict[str, float]:
    """Items and dimensions, both read off the lists that were supplied."""
    profits = data.get("profits") or []
    weights = data.get("weights") or []
    return {"items": len(profits), "dimensions": len(weights)}


# ---------------------------------------------------------------------------
# routes shared by every problem that becomes a linear program
# ---------------------------------------------------------------------------

def _simplex_route(shape: Shape, rule: str = "steepest-edge") -> Route:
    return Route(
        key="quantum-simplex",
        label="Quantum simplex",
        classical="The simplex method",
        replaces="the pivoting step, choosing which variable enters the "
                 "basis, with quantum search over the candidate columns",
        target="SimplexIter/" + rule,
        unit=Unit.GATES,
        per_record=SIMPLEX_RECORD,
        per_instance=GRAPH if shape else LP_SHAPE,
        chosen=PRECISION,
        shape=shape,
        note="A lower bound: non-leading terms are dropped and every "
             "assumption favours the quantum side, so a real run costs more.",
    )


def _ipm_route(system: str = "mnes") -> Route:
    # No problem shape here, unlike the simplex route beside it.  A Newton
    # system states its own dimension, which is what the cost reads; the
    # program's column and row counts never reach it.  Declaring them anyway
    # would make every log carry a vertex count for nothing, and would have the
    # route predict an ``m`` that is not the dimension solved -- an interior
    # point method presolves redundant rows away, and a maximum-flow model has
    # one by construction.
    return Route(
        key="quantum-interior-point" + ("" if system == "mnes" else "-oss"),
        label="Quantum interior point" + (
            "" if system == "mnes" else ", orthogonal subspace system"),
        classical="A primal-dual interior point method",
        replaces="the Newton system solve at each step with a quantum linear "
                 "solver, then reads the answer back out",
        target="IPM/" + system,
        unit=Unit.CYCLES,
        per_record=NEWTON_RECORD,
        chosen=IPM_PRECISION,
        note="A lower bound under deliberately generous assumptions: one "
             "oracle call per cycle, no amplification overhead, convergence in "
             "a single step. The readout dominates.",
    )


def _lp_routes(shape: Shape) -> Tuple[Route, ...]:
    return (_simplex_route(shape), _ipm_route("mnes"),
            _ipm_route("oss"))


# ---------------------------------------------------------------------------
# the problems
# ---------------------------------------------------------------------------

def _quadratic_route() -> Route:
    return Route(
        key="tree-generator",
        label="Quantum tree generator based search",
        classical="An exact solver for the quadratic knapsack",
        replaces="the whole search, with a quantum state that superposes "
                 "feasible choices weighted towards profitable ones, amplified "
                 "towards the best",
        target="QTG-quadratic",
        unit=Unit.CYCLES,
        per_instance=QUADRATIC_INSTANCE,
        shape=knapsack_shape,
        note="The pair bonuses must be positive. The circuit has a gate for a "
             "pair that earns something together and none for a pair that "
             "costs something, so a story about interference or cannibalisation "
             "is not one this counts.",
    )


def _multidimensional_route() -> Route:
    return Route(
        key="tree-generator",
        label="Quantum tree generator based search",
        classical="An exact solver for the multidimensional knapsack",
        replaces="the whole search, with a quantum state that superposes "
                 "feasible choices weighted towards profitable ones, amplified "
                 "towards the best",
        target="QTG-multidimensional",
        unit=Unit.CYCLES,
        per_instance=MULTIDIMENSIONAL_INSTANCE,
        shape=multidimensional_shape,
        note="Every budget is checked at once: a choice survives only where it "
             "fits in all of them. The dimensions act on separate registers, "
             "so they cost gates in proportion to their number but far fewer "
             "cycles.",
    )


def _knapsack_route() -> Route:
    return Route(
        key="tree-generator",
        label="Quantum tree generator based search",
        classical="COMBO, the state-of-the-art dynamic programming solver",
        replaces="the whole search, with a quantum state that superposes "
                 "feasible packings weighted towards profitable ones, "
                 "amplified towards the best",
        target="QTG",
        unit=Unit.CYCLES,
        per_instance=KNAPSACK_INSTANCE,
        note="The only route here whose cost depends on the actual numbers "
             "rather than on problem size: where the ones sit in each value's "
             "binary representation changes the circuit.",
    )


def _solver_routes() -> Tuple[Route, ...]:
    return tuple(
        Route(
            key=key, label=label,
            classical="A direct factorisation, or an iterative method such as "
                      "conjugate gradient",
            replaces="the whole solve, preparing the solution as a quantum "
                     "state rather than as numbers",
            target=target, unit=Unit.QUERIES,
            per_record=SOLVER_RECORD + (SIMULATED_ENTRY if simulates else ()),
            chosen=(PRECISION[0],), note=note,
        )
        for key, label, target, simulates, note in (
            ("qsvt", "Singular value transformation", "QLS-QSVT", False,
             "Lowest query count across every dataset in the comparison."),
            ("chebyshev", "Chebyshev polynomials", "QLS-Chebyshev", False,
             "Close behind, and the one the interior point work adopts."),
            ("fourier", "Fourier series", "QLS-Fourier/via-qubitization", True,
             "Twenty to sixty times dearer, but simpler to build."),
            ("hhl", "HHL", "HHL", True,
             "The famous one, and orders of magnitude more expensive: its cost "
             "scales as one over the precision where the others scale as its "
             "logarithm."),
        )
    )


#: What each family is, underneath: the routes it offers and the technical
#: problem it really is.  Everything a route needs -- the fields, the shape, the
#: units -- belongs here, so that adding a seventieth friendly name costs a line
#: of prose rather than a copy of a route.
FAMILIES: Dict[str, Tuple[str, Tuple[Route, ...]]] = {
    "maximum-flow": ("Maximum flow", (
        Route(
            key="quantum-bfs", label="Quantum breadth-first search",
            classical="Dinic's algorithm",
            replaces="the breadth-first sweep that layers the network, with "
                     "quantum search for each layer's vertices",
            target="Dinic", unit=Unit.GATES,
            collects=("layers", "phases"), renames={"vertices": "X"},
            per_instance=(GRAPH[0],),
            per_record=(Field("layers", "Layer sizes",
                              "Vertices in each layer of this sweep, as a list",
                              "[1, 3, 5, 2]"),),
            note="The circuit is explicit enough to schedule, so the derivation "
                 "counts cycles; the gate count follows by assuming one cycle "
                 "costs one gate. That assumption is generous: a cycle "
                 "usually holds many gates.",
        ),
    ) + _lp_routes(flow_shape)),
    "vertex-cover": ("Minimum vertex cover", _lp_routes(cover_shape)),
    "independent-set": ("Maximum independent set", _lp_routes(cover_shape)),
    "clique": ("Maximum clique", _lp_routes(clique_shape)),
    "linear-programming": ("Linear programming", _lp_routes(None)),
    "knapsack": ("0-1 knapsack", (_knapsack_route(),)),
    "quadratic-knapsack": ("0-1 quadratic knapsack", (_quadratic_route(),)),
    "multidimensional-knapsack": ("0-1 multidimensional knapsack",
                                  (_multidimensional_route(),)),
    "linear-systems": ("Linear systems and discretised PDEs", _solver_routes()),
}


#: The names people actually use, and the story each one tells.  Several dozen
#: of these are the same nine problems: siting satellites and siting charging
#: stations are both a quadratic knapsack, and someone arriving with one of them
#: should not have to know that to get a number.
#:
#: ``(key, family, label, blurb)``.
_CATALOGUE: Tuple[Tuple[str, str, str, str], ...] = (
    # ---- maximum flow ----------------------------------------------------
    ("maximum-flow", "maximum-flow",
     "Routing as much as possible through a network",
     "Pipes, cables, roads or shipping lanes with limited capacity: how much "
     "can travel from a source to a destination at once?"),
    ("evacuation", "maximum-flow", "Getting everyone out in time",
     "How many people an escape network can move away from a danger zone "
     "before its corridors and stairwells saturate."),
    ("supply-chain", "maximum-flow", "How much a supply chain can carry",
     "Factories, depots and lanes with limited throughput: the most that can "
     "reach the customer per week, and which link stops it going higher."),
    ("shift-assignment", "maximum-flow", "Assigning people to shifts",
     "Every worker can cover some shifts and not others. Filling as many "
     "shifts as possible is a matching, and a matching is a flow."),
    ("school-places", "maximum-flow", "Matching applicants to places",
     "Students to schools, doctors to hospitals, applicants to slots: as "
     "many good pairings as the preferences and capacities allow."),
    ("job-machines", "maximum-flow", "Putting jobs on the machines that fit",
     "Each job runs on some machines and not others. How many can run at once."),
    ("image-separation", "maximum-flow", "Separating an object from its background",
     "Cutting an image into foreground and background as cheaply as possible "
     "is a minimum cut, and a minimum cut is a maximum flow."),
    ("project-prerequisites", "maximum-flow",
     "Choosing projects when some need others first",
     "Profitable projects with prerequisites: picking the best set that is "
     "closed under what it depends on."),
    ("weakest-link", "maximum-flow", "Finding a network's weakest link",
     "Which few connections, if they failed, would cut the network in two: "
     "and how much capacity that costs."),
    ("elimination", "maximum-flow", "Who can still win the league",
     "Given the games left to play, whether a team is already out."),

    # ---- 0-1 knapsack ----------------------------------------------------
    ("knapsack", "knapsack", "Packing the most value under a budget",
     "Choosing which items to take when each has a value and a cost and the "
     "total cost is capped: cargo loading, project selection, capital "
     "budgeting."),
    ("capital-budgeting", "knapsack", "Which investments to fund",
     "A fixed pot of money and more proposals than it covers. Which set "
     "returns the most."),
    ("release-planning", "knapsack", "What goes in the next release",
     "Features with a value and an engineering cost, and a fixed number of "
     "weeks before the ship date."),
    ("ad-budget", "knapsack", "Where to spend an advertising budget",
     "Placements with a price and an expected return, and a budget that will "
     "not cover them all."),
    ("cache-contents", "knapsack", "What to keep in a cache",
     "Fixed storage, items of different sizes and different hit rates: what "
     "earns its space."),
    ("observing-night", "knapsack", "Which targets to observe tonight",
     "One night of telescope time, each target needing a slot and returning "
     "some scientific value."),
    ("maintenance-backlog", "knapsack", "Which repairs to fund this year",
     "A maintenance budget and a backlog longer than it: the set that "
     "prevents the most failure."),

    # ---- quadratic knapsack ----------------------------------------------
    ("quadratic-knapsack", "quadratic-knapsack",
     "Choosing things that are worth more together",
     "Items with their own value and a bonus for each pair chosen together, "
     "under a single budget. Coverage, synergy and reinforcement problems all "
     "have this shape."),
    ("satellite-siting", "quadratic-knapsack", "Which orbital slots to fill",
     "Each satellite covers some ground on its own, and a pair covers more "
     "between them than either does alone. Launch budget is finite."),
    ("charging-stations", "quadratic-knapsack", "Where to put charging points",
     "Each site serves its own neighbourhood; two sites on the same route "
     "serve journeys neither could serve alone."),
    ("sensor-overlap", "quadratic-knapsack", "Placing sensors that see more together",
     "Cameras or detectors whose fields overlap, so a well-chosen pair covers "
     "ground that neither alone would."),
    ("cell-towers", "quadratic-knapsack", "Siting masts for continuous coverage",
     "A mast covers its own cell; two adjacent masts give handover, so a call "
     "survives the journey between them."),
    ("depot-siting", "quadratic-knapsack", "Where to open depots",
     "Each depot serves its own area, and two depots together serve routes "
     "that run between them."),
    ("store-network", "quadratic-knapsack", "Which shops to open",
     "Each location draws its own custom, and nearby pairs reinforce a "
     "catchment rather than splitting it."),
    ("team-selection", "quadratic-knapsack", "Picking a team that works well together",
     "Each person brings something; some pairs bring more than the sum of the "
     "two. Headcount is capped."),
    ("research-portfolio", "quadratic-knapsack", "Which projects to run together",
     "Projects worth funding on their own, and worth more when they share "
     "equipment, data or people."),
    ("product-bundles", "quadratic-knapsack", "Which products to stock together",
     "Each line sells; some pairs sell better side by side than apart."),
    ("committee", "quadratic-knapsack", "Forming a committee",
     "Members chosen for what they know, and for whose expertise combines "
     "with whose."),

    # ---- multidimensional knapsack ---------------------------------------
    ("multidimensional-knapsack", "multidimensional-knapsack",
     "Choosing under several limits at once",
     "Every choice consumes some of each of several fixed resources, and all "
     "of them must hold. Weight and volume, money and people, power and space."),
    ("container-loading", "multidimensional-knapsack",
     "Loading a container by weight and volume",
     "Cargo that must fit both limits at once: heavy and small, light and "
     "bulky, and the mix that carries the most value."),
    ("cloud-packing", "multidimensional-knapsack", "Packing workloads onto a machine",
     "Each service wants processor, memory and disk, and the host has a fixed "
     "amount of each."),
    ("multi-budget-portfolio", "multidimensional-knapsack",
     "Choosing projects under money, people and time",
     "Approving a slate when the constraint is not only the budget but also "
     "the engineers and the calendar."),
    ("menu-planning", "multidimensional-knapsack", "Planning meals within several limits",
     "Calories, protein, salt and cost: every one of them capped, and the "
     "menu has to satisfy all of them together."),
    ("production-run", "multidimensional-knapsack", "What to manufacture this quarter",
     "Each product consumes several raw materials, and every stockpile is "
     "finite."),
    ("grant-allocation", "multidimensional-knapsack", "Which grants to award",
     "Funds, supervisors and bench space are separate limits, and a proposal "
     "needs all three."),
    ("payload-selection", "multidimensional-knapsack", "What to put on the spacecraft",
     "Instruments competing for mass, power and data rate at the same time."),
    ("rack-filling", "multidimensional-knapsack", "Filling a data-centre rack",
     "Power, cooling and rack units, each of which runs out on its own "
     "schedule."),
    ("commissioning-slate", "multidimensional-knapsack", "What to commission this season",
     "Budget, studio days and the availability of people, all binding at once."),

    # ---- vertex cover ----------------------------------------------------
    ("vertex-cover", "vertex-cover", "Covering a network with as few sites as possible",
     "Placing cameras, sensors or guards so that every connection is watched "
     "by at least one of them, using as few as you can."),
    ("camera-placement", "vertex-cover", "Watching every corridor",
     "Cameras at junctions, so that every corridor between them is overlooked "
     "by at least one."),
    ("network-monitors", "vertex-cover", "Monitoring every link",
     "Probes on routers so that every link in the network has a probe at one "
     "end of it."),
    ("checkpoints", "vertex-cover", "Where to put the checkpoints",
     "Every route in or out has to pass at least one, and each one costs "
     "money and delay."),
    ("pipeline-inspection", "vertex-cover", "Inspecting every pipe",
     "Inspection points at the junctions, chosen so that no length of pipe "
     "goes unvisited."),
    ("patch-priority", "vertex-cover", "Which machines to patch first",
     "Securing every connection in a network by hardening as few of its "
     "machines as possible."),

    # ---- independent set -------------------------------------------------
    ("independent-set", "independent-set",
     "Choosing as many mutually compatible items as possible",
     "Picking the largest group of things that do not conflict: "
     "non-interfering transmitters, non-clashing bookings, compatible "
     "machines."),
    ("transmitters", "independent-set", "Turning on as many transmitters as possible",
     "Two transmitters too close on the same frequency interfere. How many "
     "can run at once."),
    ("booking-clashes", "independent-set", "Fitting in as many bookings as possible",
     "Requests that overlap cannot both be granted. The most that can."),
    ("compatible-machines", "independent-set", "Running as many machines as possible",
     "Some pairs cannot run at the same time: shared power, shared "
     "extraction, shared floor. The largest set that can."),
    ("spaced-seating", "independent-set", "Seating people apart",
     "Seats too close together cannot both be used. How many people fit."),
    ("ad-slots", "independent-set", "Placing as many adverts as possible",
     "Placements that would clash with each other, and the most that can run "
     "side by side."),

    # ---- clique ----------------------------------------------------------
    ("clique", "clique", "Finding the largest fully connected group",
     "The biggest set of items that are all pairwise related: a tightly knit "
     "community, a set of mutually compatible parts."),
    ("community", "clique", "Finding a tightly knit group",
     "The largest set of people in a network who all know one another."),
    ("compatible-parts", "clique", "A set of parts that all fit together",
     "Components that are pairwise compatible, and the biggest assembly you "
     "can make from them."),
    ("trading-group", "clique", "A group that all trade with each other",
     "The largest set of parties with an existing relationship between every "
     "pair."),
    ("fragment-matching", "clique", "Molecular fragments that all fit",
     "Pieces that are pairwise compatible, and the largest combination of "
     "them."),

    # ---- linear programming ----------------------------------------------
    ("linear-programming", "linear-programming", "Allocating limited resources",
     "Production planning, blending, logistics, staffing: anything where a "
     "linear objective meets linear constraints. If you already have a model "
     "in MPS format, this is your entry."),
    ("production-planning", "linear-programming", "How much of each thing to make",
     "Lines, materials and hours that all run out, and a mix that earns the "
     "most within them."),
    ("blending", "linear-programming", "Mixing to a specification at least cost",
     "Fuels, feeds and alloys: the cheapest blend that still meets every "
     "requirement."),
    ("logistics", "linear-programming", "Moving goods for the least cost",
     "Sources, destinations and freight rates, and the cheapest way to meet "
     "every demand."),
    ("rostering", "linear-programming", "Covering the rota",
     "Enough people on every shift, respecting rest and contracts, at the "
     "lowest cost."),
    ("energy-dispatch", "linear-programming", "Which generators to run",
     "Meeting demand each hour from plants with different costs and limits."),
    ("diet-problem", "linear-programming", "The cheapest adequate diet",
     "The original linear program: nutritional requirements met for the least "
     "money."),
    ("revenue-management", "linear-programming", "How much to sell at each price",
     "Fixed capacity, several fare classes, and the allocation that earns the "
     "most."),

    # ---- linear systems --------------------------------------------------
    ("linear-systems", "linear-systems",
     "Solving a large system of equations, or a physical field",
     "Steady-state temperature, electrostatic potential, structural load: "
     "anything that discretises into a large sparse system. Also the step "
     "every method above outsources to a solver."),
    ("heat-distribution", "linear-systems", "Where the heat settles",
     "The steady-state temperature across a component once it has stopped "
     "changing."),
    ("electrostatics", "linear-systems", "The field around a charged object",
     "Potential across a region, given what sits on its boundary."),
    ("structural-load", "linear-systems", "How a structure carries its load",
     "Displacement and stress through a frame or a part under load."),
    ("circuit-analysis", "linear-systems", "Voltages around a circuit",
     "Node potentials in a network of resistors and sources."),
    ("groundwater", "linear-systems", "How water moves underground",
     "Pressure through porous rock, which is the same equation as the heat "
     "one wearing different units."),
    ("deblurring", "linear-systems", "Recovering a sharp image",
     "Undoing a known blur, which is a large and badly conditioned system."),
    ("least-squares", "linear-systems", "Fitting a model to data",
     "The best fit in the least-squares sense, through the normal equations."),
)


PROBLEMS: Tuple[Problem, ...] = tuple(
    Problem(key=key, label=label, technical=FAMILIES[family][0], blurb=blurb,
            routes=FAMILIES[family][1], family=family)
    for key, family, label, blurb in _CATALOGUE
)


@dataclass(frozen=True)
class Ask:
    """One plain-language question, in the words of whoever is being asked.

    The generator underneath needs a vertex count or an item count; nobody
    arrives with one of those.  They arrive with people, shifts, sites, budgets.
    So a family declares *what* it needs to know and the sentence it needs it in,
    and each problem supplies the nouns for its own story.
    """

    key: str
    frame: str
    example: str
    help: str = ""


#: What each family has to be told before it can build an instance, and the
#: sentence it asks in.  ``{a}`` and ``{b}`` are filled from the problem's own
#: nouns, so one frame serves every name in the family.
BEGINNER: Dict[str, Tuple[Ask, ...]] = {
    "maximum-flow": (
        Ask("things", "How many {a} are there?", "60",
            "Every place the flow can pass through, counting where it starts "
            "and where it ends up."),
        Ask("links", "How many {b} connect them?", "180",
            "More connections means more ways through, and a bigger circuit."),
    ),
    "vertex-cover": (
        Ask("things", "How many {a} are there?", "60", ""),
        Ask("links", "How many {b} have to be covered?", "180", ""),
    ),
    "independent-set": (
        Ask("things", "How many {a} are there?", "60", ""),
        Ask("links", "How many {b} are there?", "180",
            "Each one rules out having both of the things it joins."),
    ),
    "clique": (
        Ask("things", "How many {a} are there?", "40", ""),
        Ask("links", "How many {b} are there?", "500",
            "A sparse set of these makes a *denser* problem, which is the "
            "awkward thing about this one."),
    ),
    "linear-programming": (
        Ask("things", "How many {a} can you set?", "200", ""),
        Ask("links", "How many {b} must hold?", "50", ""),
    ),
    "knapsack": (
        Ask("things", "How many {a} are there to choose from?", "40", ""),
        Ask("budget", "How much {b} do you have?", "100",
            "Roughly. It sets how many of them you can afford at once."),
    ),
    "quadratic-knapsack": (
        Ask("things", "How many {a} are there to choose from?", "30", ""),
        Ask("budget", "How much {b} do you have?", "100", ""),
        Ask("pairs", "Out of every hundred pairs, how many help each other?",
            "30", "Pairs that are worth more together than apart. Only "
                  "bonuses: a pair that gets in its own way is not something "
                  "this counts."),
    ),
    "multidimensional-knapsack": (
        Ask("things", "How many {a} are there to choose from?", "40", ""),
        Ask("limits", "How many separate {b} are there?", "3",
            "Every one of them has to hold at once, which is what makes this "
            "harder than a single budget."),
        Ask("budget", "How much of each is available?", "100", ""),
    ),
    "linear-systems": (
        Ask("things", "How many {a} are there?", "500", ""),
        Ask("links", "How many others does each {b} touch?", "5",
            "Most large systems are sparse: each unknown depends on only a few "
            "of its neighbours."),
    ),
}


#: The nouns each problem uses for itself.  ``a`` is the thing being counted or
#: chosen, ``b`` is whatever the family's second question is about.
NOUNS: Dict[str, Tuple[str, str]] = {
    # maximum flow
    "maximum-flow": ("places in the network", "pipes or cables"),
    "evacuation": ("rooms and exits", "corridors between them"),
    "supply-chain": ("factories, depots and shops", "shipping lanes"),
    "shift-assignment": ("people and shifts", "possible pairings"),
    "school-places": ("applicants and places", "acceptable pairings"),
    "job-machines": ("jobs and machines", "possible pairings"),
    "image-separation": ("pixels", "neighbouring pairs"),
    "project-prerequisites": ("projects", "prerequisites between them"),
    "weakest-link": ("places in the network", "connections between them"),
    "elimination": ("teams", "remaining games"),
    # 0-1 knapsack
    "knapsack": ("items", "budget"),
    "capital-budgeting": ("proposals", "money"),
    "release-planning": ("features", "engineering time"),
    "ad-budget": ("placements", "budget"),
    "cache-contents": ("items", "storage"),
    "observing-night": ("targets", "observing time"),
    "maintenance-backlog": ("repairs", "maintenance budget"),
    # quadratic knapsack
    "quadratic-knapsack": ("items", "budget"),
    "satellite-siting": ("orbital slots", "launch budget"),
    "charging-stations": ("candidate sites", "budget"),
    "sensor-overlap": ("possible sensor positions", "budget"),
    "cell-towers": ("candidate mast sites", "budget"),
    "depot-siting": ("candidate depot sites", "budget"),
    "store-network": ("candidate locations", "budget"),
    "team-selection": ("candidates", "room on the team"),
    "research-portfolio": ("projects", "funding"),
    "product-bundles": ("product lines", "shelf space"),
    "committee": ("candidates", "room on the committee"),
    # multidimensional knapsack
    "multidimensional-knapsack": ("items", "limits"),
    "container-loading": ("crates", "limits, such as weight and volume,"),
    "cloud-packing": ("services", "resources, such as processor and memory,"),
    "multi-budget-portfolio": ("projects", "budgets, such as money and people,"),
    "menu-planning": ("dishes", "nutritional limits"),
    "production-run": ("products", "raw materials"),
    "grant-allocation": ("proposals", "resources, such as funds and supervisors,"),
    "payload-selection": ("instruments", "limits, such as mass and power,"),
    "rack-filling": ("machines", "limits, such as power and cooling,"),
    "commissioning-slate": ("productions", "budgets, such as money and studio days,"),
    # vertex cover
    "vertex-cover": ("possible sites", "connections"),
    "camera-placement": ("junctions where a camera could go", "corridors"),
    "network-monitors": ("routers", "links"),
    "checkpoints": ("places a checkpoint could stand", "routes"),
    "pipeline-inspection": ("junctions", "lengths of pipe"),
    "patch-priority": ("machines", "connections between them"),
    # independent set
    "independent-set": ("candidates", "conflicts between them"),
    "transmitters": ("transmitters", "pairs close enough to interfere"),
    "booking-clashes": ("requests", "pairs that overlap in time"),
    "compatible-machines": ("machines", "pairs that cannot run together"),
    "spaced-seating": ("seats", "pairs too close together"),
    "ad-slots": ("placements", "pairs that would clash"),
    # clique
    "clique": ("members", "relationships between them"),
    "community": ("people", "who knows whom"),
    "compatible-parts": ("components", "compatible pairs"),
    "trading-group": ("parties", "existing relationships"),
    "fragment-matching": ("fragments", "compatible pairs"),
    # linear programming
    "linear-programming": ("quantities", "rules"),
    "production-planning": ("products you could make", "capacity limits"),
    "blending": ("ingredients", "specifications"),
    "logistics": ("routes you could use", "supply and demand rules"),
    "rostering": ("shift patterns", "coverage rules"),
    "energy-dispatch": ("generators", "demand and capacity rules"),
    "diet-problem": ("foods", "nutritional requirements"),
    "revenue-management": ("fare classes", "capacity rules"),
    # linear systems
    "linear-systems": ("unknowns", "unknown"),
    "heat-distribution": ("points on the component", "point"),
    "electrostatics": ("points in the region", "point"),
    "structural-load": ("points on the structure", "point"),
    "circuit-analysis": ("junctions in the circuit", "junction"),
    "groundwater": ("points in the rock", "point"),
    "deblurring": ("pixels", "pixel"),
    "least-squares": ("parameters to fit", "parameter"),
}


#: How the menu is grouped.  Deliberately **not** by family: putting siting a
#: satellite next to siting a charging point would announce that they are one
#: problem, which is the thing the catalogue exists to spare people.  These cut
#: across families instead, by the kind of task someone thinks they have --
#: which is also how they will look for it.
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("choosing", "Choosing what to fund or build"),
    ("placing", "Deciding where to put things"),
    ("limits", "Fitting within several limits at once"),
    ("groups", "Picking people and groups"),
    ("clashes", "Avoiding clashes"),
    ("flow", "Getting things through a network"),
    ("matching", "Matching things up"),
    ("planning", "Planning and allocating"),
    ("physical", "Solving a physical system"),
)

_CATEGORY_OF: Dict[str, str] = {
    "choosing": ("knapsack capital-budgeting release-planning ad-budget "
                 "cache-contents observing-night maintenance-backlog "
                 "quadratic-knapsack"),
    "placing": ("satellite-siting charging-stations sensor-overlap cell-towers "
                "depot-siting store-network vertex-cover camera-placement "
                "network-monitors checkpoints pipeline-inspection "
                "patch-priority"),
    "limits": ("multidimensional-knapsack container-loading cloud-packing "
               "multi-budget-portfolio menu-planning production-run "
               "grant-allocation payload-selection rack-filling "
               "commissioning-slate"),
    "groups": ("team-selection committee research-portfolio product-bundles "
               "clique community compatible-parts trading-group "
               "fragment-matching"),
    "clashes": ("independent-set transmitters booking-clashes "
                "compatible-machines spaced-seating ad-slots"),
    "flow": ("maximum-flow evacuation supply-chain weakest-link "
             "image-separation"),
    "matching": ("shift-assignment school-places job-machines elimination "
                 "project-prerequisites"),
    "planning": ("linear-programming production-planning blending logistics "
                 "rostering energy-dispatch diet-problem revenue-management"),
    "physical": ("linear-systems heat-distribution electrostatics "
                 "structural-load circuit-analysis groundwater deblurring "
                 "least-squares"),
}

#: Problem key to category key, expanded once from the readable form above.
CATEGORY: Dict[str, str] = {
    key: category
    for category, keys in _CATEGORY_OF.items() for key in keys.split()
}


def category_of(problem_key: str) -> str:
    """Which heading a problem sits under, for the menu."""
    return CATEGORY.get(problem_key, "planning")


def beginner_asks(problem_key: str) -> Tuple[Field, ...]:
    """The questions this problem puts to someone who has no file.

    Each comes back as an ordinary :class:`Field`, so the interface draws it the
    same way it draws everything else -- but labelled in the words of the person
    being asked rather than in the words of the lemma underneath.
    """
    problem = get_problem(problem_key)
    first, second = NOUNS.get(problem_key, ("items", "connections"))
    return tuple(
        Field(name=ask.key,
              label=ask.frame.format(a=first, b=second),
              help=ask.help, example=ask.example)
        for ask in BEGINNER[problem.family]
    )


def family_of(problem_key: str) -> str:
    """The problem underneath the name, which is what decides how it is solved."""
    return get_problem(problem_key).family


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
