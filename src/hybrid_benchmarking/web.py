"""A local interface: the registry, in a browser, with nothing installed.

This is a client of the public API and holds no logic of its own.  Every number
it shows comes from :func:`~hybrid_benchmarking.registry.get` and
:meth:`~hybrid_benchmarking.cost.Cost.evaluate`, and every page it serves can be
reproduced by the snippet it prints alongside the answer.

The server is ``http.server`` from the standard library and the page has no
external resources at all.  That is deliberate: the promise is that a Python
installation is the only prerequisite, and a web framework or a font from a
content delivery network would quietly break it -- the second one only for
people working offline, which is the worst kind of breakage.

Nothing is uploaded anywhere.  The server binds to the loopback interface and
serves one browser on the same machine.
"""

from __future__ import annotations

import json
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compose import build, code, describe, entries, fillings_for
from .cost import Cost, UnitMismatch, ValidityWarning
from .dataset import Dataset, FormatError
from .dataset import check as check_log
from .dataset import load as load_log
from .dataset import run as run_log
from .dataset import template as log_template
from .problems import PROBLEMS, Route, get_problem, get_route
from .provenance import Unit
from .registry import Implementation, Routine, all_routines, get

_STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# turning registry objects into plain data
# ---------------------------------------------------------------------------

def _unit_name(unit: Unit) -> str:
    return unit.name


def _cost_data(cost: Cost) -> Dict[str, Any]:
    return {
        "unit": _unit_name(cost.unit),
        "unit_label": str(cost.unit),
        "parameters": list(cost.parameters),
        "formula": cost.formula(),
        "latex": cost.latex(),
        "bound": str(cost.provenance.bound),
        "derivation": str(cost.provenance.derivation),
        "sources": list(cost.provenance.sources),
        "assumptions": list(cost.provenance.assumptions),
        "conditions": [
            {"message": condition.message, "hard": condition.hard}
            for condition in cost.validity.conditions
        ],
        "slots": {name: unit.name for name, unit in cost.slots.items()},
    }


def _implementation_data(impl: Implementation) -> Dict[str, Any]:
    return {
        "name": impl.name,
        "path": impl.path,
        "summary": impl.summary,
        "citation": impl.citation,
        "built_from": list(impl.built_from),
        "units": [_unit_name(u) for u in impl.units],
        "costs": {_unit_name(u): _cost_data(c) for u, c in impl.costs.items()},
    }


def _routine_data(routine: Routine) -> Dict[str, Any]:
    return {
        "name": routine.name,
        "summary": routine.summary,
        "units": [_unit_name(u) for u in routine.units],
        "parameters": list(routine.parameters),
        "implementations": [
            _implementation_data(i) for i in routine.implementations
        ],
    }


def catalogue() -> List[Dict[str, Any]]:
    """Every routine, with just enough to draw the list and the matrix."""
    return [
        {
            "name": r.name,
            "summary": r.summary,
            "units": [_unit_name(u) for u in r.units],
            "constructions": [i.name for i in r.implementations],
        }
        for r in all_routines()
    ]


def _field_data(f) -> Dict[str, Any]:
    return {"name": f.name, "label": f.label, "help": f.help,
            "example": f.example}


def _generatable() -> frozenset:
    """Which routes this machine can produce a log for, rather than only cost.

    Imported here rather than at the top because the classical solvers pull in
    numpy, and the promise that the registry and this interface work with a bare
    Python and sympy is one there is a test for.
    """
    from .classical import supported

    return frozenset(supported())


def _family(problem_key: str) -> str:
    from .problems import family_of

    try:
        return family_of(problem_key)
    except KeyError:
        return problem_key


def _route_data(route: Route, problem_key: str = "") -> Dict[str, Any]:
    from .classical.generate import accepts

    wanted, example = accepts(problem_key, route.key)
    return {
        "key": route.key,
        #: True when pointing at an instance file is enough -- the classical
        #: run happens here and the log is produced rather than asked for.
        "generated": (_family(problem_key), route.key) in _generatable(),
        #: What sort of file, and what one is usually called.  "Give it an
        #: instance file" is only an instruction if it says which.
        "accepts": wanted,
        "example_file": example,
        "label": route.label,
        "classical": route.classical,
        "replaces": route.replaces,
        "sentence": route.describe(),
        "target": route.target,
        "unit": route.unit.name,
        "unit_label": str(route.unit),
        "note": route.note,
        "per_record": [_field_data(f) for f in route.per_record],
        "per_instance": [_field_data(f) for f in route.per_instance],
        "chosen": [_field_data(f) for f in route.chosen],
        "whole_run": bool(route.collects) or not route.per_record,
    }


def figures() -> List[Dict[str, Any]]:
    """The published results, redrawn, for the introduction."""
    from .figures import all_figures

    return all_figures()


def running_version() -> Dict[str, Any]:
    """Which build is answering, so that a reader can tell.

    Nothing on screen said this, and that is what let a stale copy pass for a
    current one: the tool updates itself in the background at an address that
    never changes, so there was no moment at which anybody could have noticed
    they were looking at last week's numbers.  A version somebody can read out
    is what makes "are you on the latest?" a question with an answer.
    """
    from .update import installed, stamp

    built = stamp()
    return {"version": installed(), "built_from": built[:7] if built else None}


def problems() -> List[Dict[str, Any]]:
    """Every problem the library can cost, under the name people use.

    Ordered by the menu's headings, so the page can group by runs of equal
    ``category`` without knowing what the headings are or what order they go in.
    Seventy-one names in one alphabetical column is a list nobody reads to the
    end of; the same names under nine headings is a list you scan.
    """
    from .problems import CATEGORIES, category_of

    order = {key: index for index, (key, _) in enumerate(CATEGORIES)}
    heading = dict(CATEGORIES)
    return [
        {
            "key": p.key,
            "label": p.label,
            "technical": p.technical,
            "blurb": p.blurb,
            "category": category_of(p.key),
            "category_label": heading[category_of(p.key)],
            "routes": [_route_data(r, p.key) for r in p.routes],
        }
        for p in sorted(PROBLEMS, key=lambda p: order[category_of(p.key)])
    ]


def problem_detail(key: str) -> Dict[str, Any]:
    from .illustrations import picture, story

    problem = get_problem(key)
    return {
        "key": problem.key,
        "label": problem.label,
        "technical": problem.technical,
        "blurb": problem.blurb,
        # What the situation is, in the reader's nouns, and what it looks like.
        "story": story(problem.key),
        "picture": picture(problem.key),
        "routes": [_route_data(r, problem.key) for r in problem.routes],
        "incomparable": len(problem.routes) > 1,
    }


def cost_from_log(problem: str, route_key: str, chosen: Dict[str, Any],
                  path: str = "", text: str = "") -> Dict[str, Any]:
    """Cost a logged classical run.

    The file is read where it sits.  Nothing is uploaded -- the server and the
    file are on the same machine, which is the whole point of running this
    locally.
    """
    route = get_route(problem, route_key)
    if path:
        data = load_log(path)
    elif text.strip():
        from .dataset import _strip_comments

        suffix = ".json" if _strip_comments(text).lstrip().startswith("{") \
            else ".csv"
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
            tmp.write(text)
            temporary = tmp.name
        data = load_log(temporary)
    else:
        raise FormatError("give a path to a log, or paste one")

    values = {name: _coerce(raw) for name, raw in (chosen or {}).items()
              if str(raw).strip()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ValidityWarning)
        result = run_log(route, data, values)
    result["warnings"] = [str(w.message) for w in caught]
    result["source"] = data.source
    result["snippet"] = (
        "import hybrid_benchmarking as hb\n\n"
        "route = hb.get_route({!r}, {!r})\n"
        "data = hb.load({!r})\n"
        "hb.run(route, data, {!r})".format(
            problem, route_key, path or "your-log.csv", values
        )
    )
    return result


def cost_from_instance(problem: str, route_key: str, path: str,
                       chosen: Dict[str, Any],
                       budget: Optional[float] = None) -> Dict[str, Any]:
    """Run the classical algorithm here, then cost what it wrote down.

    The other half of :func:`cost_from_log`, and the half that makes the panel
    usable by someone who has a network rather than a condition number.  It
    hands back the log as well as the count, because the log is the artefact and
    skipping past it would turn this into an oracle.

    Like everything else here, the file is read where it sits and the run
    happens on this machine.
    """
    from .classical import Budget, GenerationError, generate_from_file
    from .classical.budget import DEFAULT_SECONDS
    from .classical import cost as cost_generated
    from .dataset import render
    from .instances import InstanceError

    try:
        generated = generate_from_file(
            path, problem, route_key, Budget(budget or DEFAULT_SECONDS)
        )
    except (InstanceError, GenerationError) as error:
        raise FormatError(str(error))

    payload: Dict[str, Any] = {
        "instance": generated.instance.describe(),
        "layout": generated.instance.layout,
        "implementation": generated.run.implementation,
        "status": str(generated.run.status),
        "elapsed": round(generated.run.elapsed, 3),
        "result": generated.run.result or {},
        "advice": generated.run.advice(),
        "handoff": generated.run.handoff,
        "log": render(generated.data, generated.route),
    }
    if not generated.run.usable:
        payload["error"] = generated.run.reason or generated.run.advice()
        return payload

    values = {name: _coerce(raw) for name, raw in (chosen or {}).items()
              if str(raw).strip()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ValidityWarning)
        report = cost_generated(generated, values)
    report["warnings"] = [str(w.message) for w in caught]
    report["snippet"] = (
        "import hybrid_benchmarking as hb\n\n"
        "generated = hb.generate_from_file({!r}, {!r}, {!r})\n"
        "print(hb.render(generated.data, generated.route))\n"
        "hb.classical.cost(generated, {!r})".format(
            path, problem, route_key, values
        )
    )
    payload.update(report)
    return payload


def beginner_form(problem_key: str) -> Dict[str, Any]:
    """The questions to put to someone who has a problem but not a file."""
    from .problems import beginner_asks

    problem = get_problem(problem_key)
    return {
        "key": problem.key,
        "label": problem.label,
        "blurb": problem.blurb,
        "asks": [_field_data(f) for f in beginner_asks(problem_key)],
        "routes": [{"key": r.key, "label": r.label, "unit_label": str(r.unit),
                    "classical": r.classical} for r in problem.routes],
    }


def compare_from_parameters(problem_key: str, values: Dict[str, Any],
                            chosen: Optional[Dict[str, Any]] = None,
                            budget: Optional[float] = None) -> Dict[str, Any]:
    """Make an instance of the size described and cost every route on it.

    The caveat comes back first and the interface leads with it: the number is
    what a problem of that shape costs, which is not what theirs costs.
    """
    from .classical import Budget, compare
    from .classical.budget import DEFAULT_SECONDS
    from .classical.synthesise import CAVEAT

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ValidityWarning)
        result = compare(problem_key, values or {}, chosen,
                         Budget(budget or DEFAULT_SECONDS))
    result["caveat"] = CAVEAT
    result["warnings"] = sorted({str(w.message) for w in caught})
    result["snippet"] = (
        "import hybrid_benchmarking as hb\n\n"
        "hb.classical.compare({!r}, {!r})".format(problem_key, dict(values))
    )
    return result


def why_not(routine: Routine, unit: Unit) -> str:
    """Explain a unit a routine does not offer, rather than greying it out.

    A missing gate count is not an oversight; it means no gate formula exists
    until an oracle implementation is fixed.  Saying so teaches the reason in
    one line, which is the whole difference between a disabled control and an
    informative one.
    """
    if not routine.units:
        return ("this is an oracle or an input model: it is the unit other "
                "costs are counted in, and has no cost of its own")
    if unit is Unit.GATES:
        return ("no gate count: the oracle implementation is not fixed, so "
                "there is nothing to count gates for")
    if unit is Unit.CYCLES:
        if Unit.GATES in routine.units:
            return ("no cycle count: these gate counts are analytic bounds "
                    "with terms dropped, and a bound has no schedule to "
                    "parallelise")
        return "no cycle count: cycles refine gates, and there is no gate count"
    if unit is Unit.QUERIES:
        return ("no query count: this routine is costed against a fixed gate "
                "model rather than relative to an oracle")
    return "not derived for this routine"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _coerce(raw: Any) -> Any:
    """Read a form value.

    Numbers arrive as numbers; vectors and schedules arrive as JSON, because
    some costs depend on the instance rather than on summary statistics.
    """
    if isinstance(raw, (int, float, list)):
        return raw
    text = str(raw).strip()
    if not text:
        raise ValueError("empty value")
    try:
        return float(text) if any(c in text for c in ".eE") else int(text)
    except ValueError:
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("{!r} is neither a number nor valid JSON".format(text))


def snippet(path: str, unit: Optional[str], values: Dict[str, Any]) -> str:
    """The library call that reproduces what the interface just did.

    Anything clicked here becomes a line someone can paste into a script, which
    is also the answer to "this instance is too large for the packaged tool".
    """
    arguments = ["hb.Unit." + unit] if unit else []
    arguments += [
        "{}={}".format(name, json.dumps(value))
        for name, value in sorted(values.items())
    ]
    return (
        'import hybrid_benchmarking as hb\n\n'
        'hb.get({!r}).evaluate({})'.format(path, ", ".join(arguments))
    )


def _run(cost: Cost, coerced: Dict[str, Any]) -> Tuple[Cost, List[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ValidityWarning)
        evaluated = cost.evaluate(**coerced)
    return evaluated, [str(w.message) for w in caught]


def evaluate_composition(spec: Dict[str, Any],
                         values: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate an assembled cost, the same way a single one is evaluated."""
    coerced = {name: _coerce(raw) for name, raw in values.items()}
    cost, caught = _run(build(spec), coerced)
    return {
        "value": cost.value,
        "unit": _unit_name(cost.unit),
        "unit_label": str(cost.unit),
        "provenance": cost.provenance.describe(),
        "bound": str(cost.provenance.bound),
        "assumptions": list(cost.provenance.assumptions),
        "warnings": caught,
        "snippet": code(spec, coerced),
    }


def evaluate(path: str, unit: Optional[str],
             values: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one cost, reporting warnings rather than swallowing them."""
    target = get(path)
    chosen = Unit[unit] if unit else None
    coerced = {name: _coerce(raw) for name, raw in values.items()}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ValidityWarning)
        cost = target.evaluate(chosen, **coerced)

    return {
        "value": cost.value,
        "unit": _unit_name(cost.unit),
        "unit_label": str(cost.unit),
        "provenance": cost.provenance.describe(),
        "bound": str(cost.provenance.bound),
        "assumptions": list(cost.provenance.assumptions),
        "warnings": [str(w.message) for w in caught],
        "snippet": snippet(path, unit, coerced),
    }


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "hybrid-benchmarking"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # a local tool should not chatter into the terminal

    # -- helpers ------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The package underneath this server is replaced in place by the
        # updater, at an address that never changes.  A browser holding the
        # old page would then show the old interface over the new code, and
        # the reader would have no way of telling: no version is on screen,
        # and reloading is not something one thinks to do with an app that
        # was opened from a desktop icon.  Nothing here crosses a network, so
        # there is no bandwidth to weigh against saying so plainly.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _fail(self, error: Exception, status: int = 400) -> None:
        self._json({"error": str(error)}, status)

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            page = (_STATIC / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path == "/api/routines":
            return self._json(catalogue())
        if self.path == "/api/problems":
            return self._json(problems())
        if self.path == "/api/figures":
            return self._json(figures())
        if self.path == "/api/version":
            return self._json(running_version())
        if self.path.startswith("/api/beginner/"):
            try:
                return self._json(beginner_form(self.path.rsplit("/", 1)[-1]))
            except KeyError as error:
                return self._fail(error, 404)
        if self.path.startswith("/api/problem/"):
            try:
                return self._json(problem_detail(self.path.rsplit("/", 1)[-1]))
            except KeyError as error:
                return self._fail(error, 404)
        if self.path.startswith("/api/template/"):
            rest = self.path[len("/api/template/"):]
            spec, _, fmt = rest.partition("?")
            problem_key, _, route_key = spec.partition("/")
            try:
                route = get_route(problem_key, route_key)
            except KeyError as error:
                return self._fail(error, 404)
            kind = "json" if "json" in fmt else "csv"
            return self._send(200, log_template(route, kind).encode("utf-8"),
                              "text/plain; charset=utf-8")
        if self.path == "/api/entries":
            return self._json(entries())
        if self.path.startswith("/api/fillings/"):
            try:
                return self._json(fillings_for(self.path.rsplit("/", 1)[-1]))
            except KeyError as error:
                return self._fail(error, 404)
        if self.path.startswith("/api/routine/"):
            name = self.path[len("/api/routine/"):]
            try:
                target = get(name.replace("%2F", "/"))
            except (KeyError, ValueError) as error:
                return self._fail(error, 404)
            if isinstance(target, Implementation):
                return self._json(_implementation_data(target))
            payload = _routine_data(target)
            payload["missing"] = {
                _unit_name(u): why_not(target, u)
                for u in (Unit.GATES, Unit.QUERIES, Unit.CYCLES)
                if u not in target.units
            }
            return self._json(payload)
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/quit":
            # An update installed under a server that is already running leaves
            # the old code serving until the machine restarts, which on this
            # tool could be weeks.  So the launcher asks the old one to stand
            # down and starts the new one.  Shutting down from inside a handler
            # would deadlock, hence the thread.
            import threading

            self._json({"stopping": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if self.path not in ("/api/evaluate", "/api/compose",
                             "/api/compose/evaluate", "/api/problem/run",
                             "/api/problem/generate", "/api/problem/compare"):
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            return self._fail(error)
        try:
            if self.path == "/api/evaluate":
                return self._json(evaluate(
                    request["path"], request.get("unit"),
                    request.get("values", {}),
                ))
            if self.path == "/api/problem/compare":
                return self._json(compare_from_parameters(
                    request["problem"], request.get("values", {}),
                    request.get("chosen", {}), request.get("budget"),
                ))
            if self.path == "/api/problem/generate":
                return self._json(cost_from_instance(
                    request["problem"], request["route"],
                    request.get("instance", ""), request.get("chosen", {}),
                    request.get("budget"),
                ))
            if self.path == "/api/problem/run":
                return self._json(cost_from_log(
                    request["problem"], request["route"],
                    request.get("chosen", {}), request.get("path", ""),
                    request.get("text", ""),
                ))
            if self.path == "/api/compose":
                return self._json(describe(request["spec"]))
            return self._json(evaluate_composition(
                request["spec"], request.get("values", {})
            ))
        except (KeyError, ValueError, UnitMismatch) as error:
            return self._fail(error)


def serve(host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> Tuple[str, ThreadingHTTPServer]:
    """Start the interface and hand back its address."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = "http://{}:{}/".format(host, httpd.server_address[1])
    if open_browser:
        webbrowser.open(url)
    return url, httpd
