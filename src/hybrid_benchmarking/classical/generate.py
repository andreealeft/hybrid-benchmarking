"""From a file on disk to a cost, with the log visible in between.

One function does the whole loop -- read the instance, pick the route, run the
classical algorithm under a budget, write the log, cost it -- and every stage of
it is separately available, because the log is the point.  Someone who wants to
see what was recorded before believing what it cost should be able to, and
someone who already has a log of their own should never come through here at
all.

Which route an instance gets is a default, not a decision: a DIMACS network is
offered Dinic because that is the study the network format comes from, and a
graph is offered the simplex because that is what a vertex cover relaxation is
solved by.  Any other route the problem catalogue lists can be asked for
instead, and the ones that cannot be produced say why rather than failing
obscurely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..dataset import Dataset
from ..dataset import run as cost_log
from ..dataset import write as write_log
from ..instances import Graph, Instance, InstanceError, Knapsack, LinearProgram
from ..instances import Matrix, MultidimensionalKnapsack, Network
from ..instances import QuadraticKnapsack
from ..instances import read as read_instance
from ..problems import Route, family_of, get_problem, get_route
from .budget import DEFAULT_SECONDS, Budget, Run, Status


class GenerationError(ValueError):
    """This instance cannot be run down this route, and here is why."""


#: What an instance of each layout is offered when nobody says otherwise.
DEFAULT_ROUTE: Dict[str, Tuple[str, str]] = {
    "dimacs-max": ("maximum-flow", "quantum-bfs"),
    "dimacs-edge": ("vertex-cover", "quantum-simplex"),
    "pisinger": ("knapsack", "tree-generator"),
    "quadratic-knapsack": ("quadratic-knapsack", "tree-generator"),
    "multidimensional-knapsack": ("multidimensional-knapsack",
                                  "tree-generator"),
    "matrix-market": ("linear-systems", "qsvt"),
    "mps": ("linear-programming", "quantum-simplex"),
}


@dataclass(frozen=True)
class Generated:
    """An instance, the log we made from it, and how the run that made it went."""

    instance: Instance
    route: Route
    run: Run
    data: Dataset

    @property
    def status(self) -> Status:
        return self.run.status

    def save(self, path: str) -> str:
        """Write the log where the user can read it."""
        return write_log(self.data, path, self.route)


# ---------------------------------------------------------------------------
# what each route needs from a classical run
# ---------------------------------------------------------------------------

Runner = Callable[[Instance, Route, Budget, Dict[str, Any]], Run]
_RUNNERS: Dict[Tuple[str, str], Runner] = {}
#: Which kind of instance each route's runner needs.  Kept beside the runner so
#: an interface can say what to hand over rather than leaving someone to find
#: out by handing over the wrong thing.
_ACCEPTS: Dict[Tuple[str, str], type] = {}

#: How to describe each kind of instance to someone about to look for one on
#: their disk: what it is, and what it is usually called.
_DESCRIBED: Dict[str, Tuple[str, str]] = {
    "Network": ("a DIMACS maximum-flow network", "instance.max"),
    "Graph": ("a DIMACS graph or edge list", "instance.clq"),
    "Knapsack": ("a knapsack instance, Pisinger or Martello-Toth layout",
                 "instance.kp"),
    "QuadraticKnapsack": ("a quadratic knapsack instance, Billionnet-Soutif "
                          "layout", "instance.qkp"),
    "MultidimensionalKnapsack": ("a multidimensional knapsack instance, "
                                 "OR-Library layout", "instance.mdkp"),
    "Matrix": ("a Matrix Market matrix", "instance.mtx"),
    "LinearProgram": ("an MPS model, fixed or free format", "model.mps"),
}


def accepts(problem: str, route: str) -> Tuple[str, str]:
    """What to hand this route, and what such a file is usually called."""
    try:
        problem = family_of(problem)
    except KeyError:
        pass
    wanted = _ACCEPTS.get((problem, route))
    if wanted is None:
        return "", ""
    return _DESCRIBED.get(wanted.__name__, ("an instance file", "instance"))


def _runner(problem: str, route: str, wants: type) -> Callable[[Callable], Callable]:
    """Register a classical run against a route, and the instance type it needs."""
    def register(function):
        def checked(instance, chosen_route, budget, options):
            if not isinstance(instance, wants):
                raise GenerationError(
                    "{} needs {}, and {} is {}".format(
                        chosen_route.label, wants.__name__.lower(),
                        instance.name, type(instance).__name__.lower(),
                    )
                )
            return function(instance, chosen_route, budget, options)

        _RUNNERS[(problem, route)] = checked
        _ACCEPTS[(problem, route)] = wants
        return function
    return register


@_runner("maximum-flow", "quantum-bfs", Network)
def _run_dinic(instance: Network, route: Route, budget: Budget,
               options: Dict[str, Any]) -> Run:
    from .dinic import solve

    return solve(instance, budget)


@_runner("knapsack", "tree-generator", Knapsack)
def _run_knapsack(instance: Knapsack, route: Route, budget: Budget,
                  options: Dict[str, Any]) -> Run:
    from .knapsack import solve

    return solve(instance, budget)


@_runner("quadratic-knapsack", "tree-generator", QuadraticKnapsack)
def _run_quadratic(instance: QuadraticKnapsack, route: Route, budget: Budget,
                   options: Dict[str, Any]) -> Run:
    from .knapsack import solve_quadratic

    return solve_quadratic(instance, budget)


@_runner("multidimensional-knapsack", "tree-generator",
         MultidimensionalKnapsack)
def _run_multidimensional(instance: MultidimensionalKnapsack, route: Route,
                          budget: Budget, options: Dict[str, Any]) -> Run:
    from .knapsack import solve_multidimensional

    return solve_multidimensional(instance, budget)


def _run_linear_system(instance: Matrix, route: Route, budget: Budget,
                       options: Dict[str, Any]) -> Run:
    """One system, solved once; which solver is costed changes nothing here.

    The four quantum solvers differ in how they reach the matrix, not in what
    they need to know about it, so the same log serves all four -- which is
    what makes comparing them on one instance a comparison rather than four
    separate measurements.
    """
    from .linsystem import solve

    right = options.get("rhs") or None
    if isinstance(right, str):
        from ..instances.matrixmarket import read_vector

        right = read_vector(right)
    return solve(instance, budget, right)


for _key in ("qsvt", "chebyshev", "fourier", "hhl"):
    _runner("linear-systems", _key, Matrix)(_run_linear_system)


def _run_simplex(problem: str, instance, route: Route, budget: Budget,
                 options: Dict[str, Any]) -> Run:
    """Build the linear program this problem makes of the instance, then pivot.

    The graph problems predict their own shape -- a vertex cover on ``V``
    vertices and ``E`` edges is a program of ``2V + E`` columns and ``V + E``
    rows -- and the route feeds that prediction to the lemmas rather than
    anything this run reports.  So the two have to agree, and this is where
    that is checked: a disagreement means the model built here is not the model
    the cost is being computed for, which is precisely the kind of mismatch
    that produces a plausible number.
    """
    from .lp import ModelError, model_for, standard_form
    from .simplex import solve

    try:
        form = standard_form(model_for(instance, problem))
    except ModelError as error:
        raise GenerationError(str(error))

    # The pivoting rule is part of the cost, not a preference: the route names
    # SimplexIter/steepest-edge, and running a different rule would log the
    # iterations of one algorithm and price them with another's formula. It is
    # honoured only where the target agrees with it.
    rule = options.get("rule", "steepest-edge")
    if not route.target.endswith("/" + rule):
        raise GenerationError(
            "this route costs {}, so a run under the {} rule would be logging "
            "one algorithm and pricing it as another".format(route.target, rule)
        )

    run = solve(form, budget, rule)
    if route.shape is not None:
        predicted = route.shape(_instance_shape(instance))
        for name in ("n", "m"):
            if int(predicted[name]) != form.shape[name]:
                raise GenerationError(
                    "{} predicts a program of {} = {:.0f} but the one built "
                    "here has {} = {}; the log and the cost would be about "
                    "different programs".format(
                        route.label, name, predicted[name], name,
                        form.shape[name])
                )
    run.instance.update(_instance_shape(instance))
    return run


def _instance_shape(instance) -> Dict[str, Any]:
    """The instance's own size, under the names the route's shape expects."""
    if isinstance(instance, Graph):
        return {"vertices": instance.vertices, "edges": len(instance.edges)}
    if isinstance(instance, Network):
        return {"vertices": instance.vertices, "edges": len(instance.arcs)}
    return {}


def _run_ipm(problem: str, instance, route: Route, budget: Budget,
             options: Dict[str, Any]) -> Run:
    """The same linear program, attacked from the inside instead.

    Shares the standard form with the simplex route deliberately: the two routes
    through a problem are meant to be two ways at one program, and building the
    program twice would be two ways at two programs.
    """
    from .ipm import solve
    from .lp import ModelError, model_for, standard_form

    try:
        form = standard_form(model_for(instance, problem))
    except ModelError as error:
        raise GenerationError(str(error))

    run = solve(form, budget, options.get("system") or
                ("oss" if route.target.endswith("/oss") else "mnes"))
    run.instance.update(_instance_shape(instance))
    return run


for _problem, _type in (("vertex-cover", Graph), ("independent-set", Graph),
                        ("clique", Graph), ("maximum-flow", Network),
                        ("linear-programming", LinearProgram)):
    _runner(_problem, "quantum-simplex", _type)(
        (lambda key: lambda instance, route, budget, options:
            _run_simplex(key, instance, route, budget, options))(_problem)
    )
    for _which in ("quantum-interior-point", "quantum-interior-point-oss"):
        _runner(_problem, _which, _type)(
            (lambda key: lambda instance, route, budget, options:
                _run_ipm(key, instance, route, budget, options))(_problem)
        )


def supported() -> Tuple[Tuple[str, str], ...]:
    """Every problem and route this package can produce a log for."""
    return tuple(sorted(_RUNNERS))


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def generate(instance: Instance, problem: str = "", route: str = "",
             budget: Optional[Budget] = None, **options: Any) -> Generated:
    """Run the classical algorithm on an instance and hand back the log."""
    problem_key, route_key = _choose(instance, problem, route)
    chosen = get_route(problem_key, route_key)

    runner = _RUNNERS.get((family_of(problem_key), route_key))
    if runner is None:
        raise GenerationError(
            "no instrumented classical run for {}/{}; this tool can generate "
            "logs for {}".format(
                problem_key, route_key,
                ", ".join("{}/{}".format(*pair) for pair in supported()),
            )
        )

    allowance = budget if budget is not None else Budget(DEFAULT_SECONDS)
    run = runner(instance, chosen, allowance, dict(options))
    data = Dataset(
        records=tuple(run.records),
        instance=dict(run.instance),
        source=instance.source or instance.name,
        generated=run.stated(),
    )
    return Generated(instance=instance, route=chosen, run=run, data=data)


def generate_from_file(path: str, problem: str = "", route: str = "",
                       budget: Optional[Budget] = None, layout: str = "",
                       **options: Any) -> Generated:
    """Read an instance file and run it, in one step.

    ``layout`` names the reader outright, for the files whose extension does not
    give them away.  Several of the benchmark sets ship as ``.txt``, which says
    nothing about what is inside, and guessing between formats is the one thing
    the readers are built not to do.
    """
    return generate(read_instance(path, layout), problem, route, budget,
                    **options)


def generate_from_parameters(problem: str, values: Dict[str, Any],
                             route: str = "",
                             budget: Optional[Budget] = None,
                             **options: Any) -> Generated:
    """Make up an instance of the size described, then run it like any other.

    The caveat that it was made up travels on the run, so every cost built this
    way carries it and the interface shows it before the number.
    """
    from ..instances import InstanceError
    from .synthesise import CAVEAT, build

    try:
        instance = build(problem, values)
    except InstanceError as error:
        raise GenerationError(str(error))
    made = generate(instance, problem, route, budget, **options)
    made.run.assumptions = (CAVEAT,) + tuple(made.run.assumptions)
    made.data.generated["assumptions"] = list(made.run.stated().get(
        "assumptions", ()))
    return made


def compare(problem: str, values: Dict[str, Any],
            chosen: Optional[Dict[str, Any]] = None,
            budget: Optional[Budget] = None) -> Dict[str, Any]:
    """Every route through a problem, on one generated instance.

    One instance for all of them, because two routes costed on two instances
    would differ for a reason nobody asked about.

    The results are put side by side and **not** added, ranked or reconciled.
    They are in different units and rest on different assumptions -- the
    quantum-search route counts a fixed schedule, the simplex route is a lower
    bound with terms dropped, the interior point route is a cycle bound resting
    on deliberately generous assumptions -- so the columns are a description of
    what each analysis says, not a race between them.  Every one carries its own
    unit, bound and provenance for exactly that reason.
    """
    from ..problems import get_problem

    results = []
    for route in get_problem(problem).routes:
        entry: Dict[str, Any] = {"route": route.key, "label": route.label,
                                 "classical": route.classical,
                                 "note": route.note}
        try:
            made = generate_from_parameters(problem, values, route.key, budget)
            report = cost(made, _for(route, chosen))
            entry.update(total=report["total"], unit=report["unit"],
                         unit_label=report["unit_label"],
                         bound=report["bound"],
                         derivation=report["derivation"],
                         provenance=report["provenance"],
                         assumptions=report["assumptions"],
                         records=report["logged_records"],
                         instance=made.instance.describe(),
                         result=made.run.result or {})
            entry["timing"] = timing(report["total"], report["unit"],
                                     made.run)
        except (GenerationError, ValueError) as error:
            entry["error"] = str(error)
        results.append(entry)

    units = sorted({r["unit_label"] for r in results if "unit_label" in r})
    return {
        "problem": problem,
        "routes": results,
        "units": units,
        "comparable": False,
        "why": "These are not four measurements of one thing. They count "
               "different units and rest on different assumptions, so they sit "
               "side by side and are never added or ranked against each other.",
    }


def timing(total: float, unit_name: str, run) -> Dict[str, Any]:
    """Put a costed route beside the classical solver that produced its log.

    The comparison is between *whole algorithms*, which is not the same as
    between a count and a stopwatch.  A quantum routine that replaces part of a
    classical solve leaves the rest of it standing: Dinic's layering sweeps
    become quantum searches and the blocking-flow phases do not, so the honest
    quantum column is the part it replaces, costed, plus the part it does not,
    measured.  ``Run.replaced_seconds`` says which part; where it says nothing,
    the routine replaces the run entire.

    The required rate divides by the replaced part for the same reason.  Asking
    what gate time would make a count of sweeps match a whole max-flow solve
    would flatter the quantum side by however much of the solve it never
    touched.

    Returns the projection, the hybrid total, the required rate and why -- or,
    for a unit with no duration, only why not.  It never raises: a route that
    cannot be timed belongs in the same table as one that can, and the reason
    is more useful than a gap.
    """
    from ..provenance import Unit
    from ..runtime import NoClock, RECORD_SECONDS, humanise, project

    classical_seconds = float(getattr(run, "elapsed", run) or 0.0)
    replaced = getattr(run, "replaced_seconds", None)
    replaced = classical_seconds if replaced is None else float(replaced)
    retained = max(0.0, classical_seconds - replaced)

    # Unrounded: these are arithmetic, and the hybrid total and the required
    # rate are derived from them.  The humanised strings beside them are what
    # anyone reads.
    block: Dict[str, Any] = {"classical_seconds": classical_seconds,
                             "classical": humanise(classical_seconds),
                             "replaced_seconds": replaced,
                             "replaced": humanise(replaced),
                             "retained_seconds": retained,
                             "retained": humanise(retained),
                             "partial": retained > 0}
    # The report names a unit as ``GATES``; the enum answers to that and to
    # ``gates``, and which one arrives depends on who called.
    try:
        unit = Unit[unit_name] if unit_name in Unit.__members__ else Unit(unit_name)
    except (KeyError, ValueError):
        block["why_not"] = "unknown unit {}".format(unit_name)
        return block
    try:
        projection = project(total, unit, replaced)
    except NoClock as error:
        block["why_not"] = str(error)
        return block

    hybrid = projection.seconds + retained
    block.update(
        seconds=projection.seconds,
        quantum=humanise(projection.seconds),
        hybrid_seconds=hybrid,
        hybrid=humanise(hybrid),
        rate=projection.rate,
        rate_named=projection.rate_named,
        required=projection.required,
        required_named=humanise(projection.required or 0.0),
        shortfall=projection.shortfall,
        summary=projection.describe(),
        assumptions=list(projection.assumptions),
        reference=RECORD_SECONDS,
    )
    if retained > 0:
        block["note"] = (
            "the quantum routine replaces {} of a {} solve; the remaining {} "
            "stays classical and is counted in the total".format(
                humanise(replaced), humanise(classical_seconds),
                humanise(retained))
        )
    return block


def _for(route, chosen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill a route's chosen parameters from its own defaults, then overrides."""
    from ..web import _coerce

    values: Dict[str, Any] = {}
    for field in route.chosen:
        if field.example:
            values[field.name] = _coerce(field.example)
    for name, raw in (chosen or {}).items():
        if str(raw).strip():
            values[name] = _coerce(raw)
    return values


def cost(generated: Generated,
         chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cost a generated log, exactly as a hand-written one is costed.

    Nothing here knows the log was generated; the extra caveats travel on the
    log itself, which is what makes the two paths the same path.
    """
    if not generated.run.usable:
        raise GenerationError(
            "{} produced no usable log: {}".format(
                generated.instance.name, generated.run.advice()
            )
        )
    report = cost_log(generated.route, generated.data, chosen)
    report["instance"] = generated.instance.name
    report["advice"] = generated.run.advice()
    # The classical run that produced this log was timed, so the count can be
    # put beside it without assuming anything the log does not already carry.
    report["timing"] = timing(report["total"], report["unit"],
                              generated.run)
    return report


def _choose(instance: Instance, problem: str, route: str) -> Tuple[str, str]:
    """Fill in whichever of problem and route was not asked for."""
    fallback = DEFAULT_ROUTE.get(instance.layout)
    if not problem:
        if fallback is None:
            raise GenerationError(
                "nothing is registered as the natural problem for a {} file; "
                "name one".format(instance.layout)
            )
        problem = fallback[0]
    if not route:
        if fallback is not None and fallback[0] == problem:
            route = fallback[1]
        else:
            routes = get_problem(problem).routes
            route = routes[0].key
    get_route(problem, route)  # raises by name if either is unknown
    return problem, route
