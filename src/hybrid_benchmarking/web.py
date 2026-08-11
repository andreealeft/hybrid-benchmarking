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

from .cost import Cost, ValidityWarning
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


def why_not(routine: Routine, unit: Unit) -> str:
    """Explain a unit a routine does not offer, rather than greying it out.

    A missing gate count is not an oversight; it means no gate formula exists
    until an oracle implementation is fixed.  Saying so teaches the reason in
    one line, which is the whole difference between a disabled control and an
    informative one.
    """
    if not routine.units:
        return ("this is an oracle or an input model -- it is the unit other "
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
        if self.path != "/api/evaluate":
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            return self._fail(error)
        try:
            return self._json(evaluate(
                request["path"], request.get("unit"), request.get("values", {})
            ))
        except (KeyError, ValueError) as error:
            return self._fail(error)


def serve(host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> Tuple[str, ThreadingHTTPServer]:
    """Start the interface and hand back its address."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = "http://{}:{}/".format(host, httpd.server_address[1])
    if open_browser:
        webbrowser.open(url)
    return url, httpd
