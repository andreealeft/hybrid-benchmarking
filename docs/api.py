"""The same answers as the local server, without the server.

The interface talks to a handful of ``/api`` addresses.  On a machine where the
tool is installed those are served by ``web.py`` over the loopback interface; in
a browser there is no loopback and no server, so this dispatcher answers the
same addresses by calling the same functions directly.  It exists on the Pages
branch alone and the package does not know about it: the point of this build is
that somebody who has never opened a terminal can still get a number, not that
the library should grow a second entry point.

What is missing here, and honestly so: the routes that read a file from a path.
A browser tab has no path to read.  The file the user picks is written into
Pyodide's own filesystem first and the ordinary route is then given that name,
so the same code reads it.
"""

from __future__ import annotations

import json

from hybrid_benchmarking import web
from hybrid_benchmarking.problems import get_route


def _template(rest):
    spec, _, fmt = rest.partition("?")
    problem_key, _, route_key = spec.partition("/")
    route = get_route(problem_key, route_key)
    return web.log_template(route, "json" if "json" in fmt else "csv")


def handle(path: str, method: str = "GET", body: str = "") -> str:
    """Answer one request, and hand back what the server would have sent.

    Errors come back the way the page already expects them, as an object with
    an ``error`` field, so the interface needs no special case for running
    without a server.
    """
    request = {}
    if body:
        try:
            request = json.loads(body)
        except json.JSONDecodeError as error:
            return json.dumps({"error": "malformed request: {}".format(error)})

    try:
        if method == "GET":
            if path == "/api/routines":
                return json.dumps(web.catalogue())
            if path == "/api/problems":
                return json.dumps(web.problems())
            if path == "/api/figures":
                return json.dumps(web.figures())
            if path == "/api/entries":
                return json.dumps(web.entries())
            if path.startswith("/api/beginner/"):
                return json.dumps(web.beginner_form(path.rsplit("/", 1)[-1]))
            if path.startswith("/api/problem/"):
                return json.dumps(web.problem_detail(path.rsplit("/", 1)[-1]))
            if path.startswith("/api/fillings/"):
                return json.dumps(web.fillings_for(path.rsplit("/", 1)[-1]))
            if path.startswith("/api/template/"):
                return _template(path[len("/api/template/"):])
            if path.startswith("/api/routine/"):
                name = path[len("/api/routine/"):].replace("%2F", "/")
                target = web.get(name)
                if isinstance(target, web.Implementation):
                    return json.dumps(web._implementation_data(target))
                payload = web._routine_data(target)
                payload["missing"] = {
                    web._unit_name(unit): web.why_not(target, unit)
                    for unit in (web.Unit.GATES, web.Unit.QUERIES,
                                 web.Unit.CYCLES)
                    if unit not in target.units
                }
                return json.dumps(payload)

        if method == "POST":
            if path == "/api/problem/compare":
                return json.dumps(web.compare_from_parameters(
                    request["problem"], request.get("values", {}),
                    request.get("chosen", {}), request.get("budget")))
            if path == "/api/problem/generate":
                return json.dumps(web.cost_from_instance(
                    request["problem"], request["route"],
                    request.get("instance", ""), request.get("chosen", {}),
                    request.get("budget")))
            if path == "/api/problem/run":
                return json.dumps(web.cost_from_log(
                    request["problem"], request["route"],
                    request.get("chosen", {}), request.get("path", ""),
                    request.get("text", "")))
            if path == "/api/evaluate":
                return json.dumps(web.evaluate(
                    request["path"], request.get("unit"),
                    request.get("values", {})))
            if path == "/api/compose":
                return json.dumps(web.compose_describe(request))
            if path == "/api/compose/evaluate":
                return json.dumps(web.compose_evaluate(request))

    except Exception as error:            # the page shows the reason
        return json.dumps({"error": str(error)})

    return json.dumps({"error": "no such address: {}".format(path)})
