"""Command line entry point.

``hybrid-benchmarking`` with no arguments starts the local interface, since
that is what someone who has just installed it wants.  The other subcommands
are the same operations without a browser, so anything the interface can do is
scriptable -- which is what makes "too large for the packaged tool" a command
rather than an email.

``log``, ``run`` and ``batch`` are the ones that start from a file rather than
from a number.  They read an instance, run the classical algorithm on it here,
and show the log before costing it: ``log`` stops after the log, ``run`` goes on
to the count, and ``batch`` does the same for a directory and tabulates.  The
log is never skipped over, because it is the thing anyone checking this work has
to be able to see.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .provenance import Unit
from .registry import Implementation, all_implementations, capability_table, get
from .web import evaluate, serve, snippet, why_not


def _parse_assignments(pairs: Sequence[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(
                "parameters are given as name=value, not {!r}".format(pair)
            )
        name, _, value = pair.partition("=")
        values[name.strip()] = value.strip()
    return values


def _command_serve(args: argparse.Namespace) -> int:
    url, httpd = serve(args.host, args.port, open_browser=not args.no_browser)
    print("hybrid-benchmarking is running at {}".format(url))
    print("nothing leaves this machine; press Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def _command_list(args: argparse.Namespace) -> int:
    print(capability_table())
    return 0


def _command_show(args: argparse.Namespace) -> int:
    target = get(args.routine)
    implementations = [target] if isinstance(target, Implementation) \
        else list(target.implementations)

    print(target.name if not isinstance(target, Implementation) else target.path)
    print("  " + target.summary.replace("\n", "\n  "))
    for impl in implementations:
        label = impl.path if impl.name != "default" else impl.routine
        print("\n  {}".format(label))
        if impl.citation:
            print("    source      {}".format(impl.citation))
        if impl.built_from:
            print("    built from  {}".format(", ".join(impl.built_from)))
        if not impl.costs:
            print("    no cost of its own")
        for unit, cost in impl.costs.items():
            print("    {:<11} {} ({})".format(
                str(unit), ", ".join(cost.parameters) or "no parameters",
                cost.provenance.bound,
            ))
            for assumption in cost.provenance.assumptions:
                print("      assuming  {}".format(assumption))
            for condition in cost.validity.conditions:
                print("      {}  {}".format(
                    "requires " if condition.hard else "valid if", condition.message
                ))

    if not isinstance(target, Implementation):
        for unit in (Unit.GATES, Unit.QUERIES, Unit.CYCLES):
            if unit not in target.units:
                print("\n  no {} -- {}".format(unit, why_not(target, unit)))
    return 0


def _command_cost(args: argparse.Namespace) -> int:
    try:
        result = evaluate(
            args.routine, args.unit, _parse_assignments(args.parameter)
        )
    except (KeyError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    for warning in result["warnings"]:
        print("warning: {}".format(warning), file=sys.stderr)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("{:.6g} {}".format(result["value"], result["unit_label"]))
        print(result["provenance"])
    return 0


def _command_formula(args: argparse.Namespace) -> int:
    target = get(args.routine)
    unit = Unit[args.unit] if args.unit else None
    cost = target.cost(unit)
    print(cost.formula())
    return 0


# ---------------------------------------------------------------------------
# starting from a file rather than from a number
# ---------------------------------------------------------------------------

#: How many rows of a log to show before saying how many more there are.  Enough
#: to see the shape of it and notice a column of zeros; short enough that the
#: cost underneath is still on the screen.
_PREVIEW = 6


def _chosen_for(route, overrides: Dict[str, str]) -> Dict[str, Any]:
    """The parameters nobody measures, filled in from the route's own defaults.

    Precision and tolerance are choices, not observations, so they cannot come
    out of a classical run.  The defaults are the ones the thesis used for its
    own figures; they are printed with the answer, because at these scales the
    cost is linear in both and the choice moves the number more than it looks.
    """
    from .web import _coerce

    values: Dict[str, Any] = {}
    for field in route.chosen:
        raw = overrides.get(field.name, field.example)
        if raw in (None, ""):
            continue
        values[field.name] = _coerce(raw)
    for name, raw in overrides.items():
        if name not in values:
            values[name] = _coerce(raw)
    return values


def _show_instance(instance) -> None:
    print(instance.describe())


def _show_run(generated) -> None:
    run = generated.run
    print("{} -- {} in {:.1f}s, {} record{}".format(
        generated.route.classical, run.status, run.elapsed,
        len(run.records), "" if len(run.records) == 1 else "s",
    ))
    if run.result:
        print("  " + ", ".join(
            "{} {}".format(key.replace("_", " "), value)
            for key, value in run.result.items()
        ))
    if run.advice():
        print("  " + run.advice())


def _log_preview(generated) -> List[str]:
    """The first few records, in whichever shape the log is written.

    A JSON log indents, so its first six lines are its header rather than its
    data; showing the records one to a line instead is what makes a preview a
    preview.  This is presentation only -- the file itself is what
    :func:`~.dataset.render` writes.
    """
    from .dataset import is_flat, render

    if is_flat(generated.data):
        # The header is provenance, and it is already printed under the cost;
        # what a preview is for is seeing the shape of the data.
        return [line for line in render(generated.data, generated.route)
                .splitlines() if not line.startswith("#")]
    lines = ["instance: " + json.dumps(generated.data.instance)]
    lines += [json.dumps(record) for record in generated.data.records]
    return lines


def _show_log(generated, saved: str = "") -> None:
    lines = _log_preview(generated)
    print("\nthe log this produced:")
    for line in lines[:_PREVIEW]:
        print("  " + line)
    if len(lines) > _PREVIEW:
        print("  ... {} more".format(len(lines) - _PREVIEW))
    print("  " + ("written to " + saved if saved else
                  "-o PATH to keep it; it is the input everything below is from"))


def _show_cost(report: Dict[str, Any], chosen: Dict[str, Any]) -> None:
    print("\n{:.6g} {}".format(report["total"], report["unit_label"]))
    logged = report.get("logged_records", report["records"])
    print("  over {} logged record{}{}".format(
        logged, "" if logged == 1 else "s",
        ", costed as one whole run" if report["whole_run"] else "",
    ))
    if chosen:
        print("  at " + ", ".join(
            "{} = {:g}".format(name, value) if isinstance(value, float)
            else "{} = {}".format(name, value)
            for name, value in sorted(chosen.items())
        ))
    print("  " + report["provenance"])
    for warning in report.get("warnings", ()):
        print("  warning: {}".format(warning), file=sys.stderr)


def _generate(args: argparse.Namespace, path: str):
    from .classical import Budget, generate_from_file

    return generate_from_file(
        path, args.problem or "", args.route or "", Budget(args.budget),
    )


def _command_log(args: argparse.Namespace) -> int:
    from .classical import GenerationError
    from .instances import InstanceError

    try:
        generated = _generate(args, args.instance)
    except (InstanceError, GenerationError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    _show_instance(generated.instance)
    _show_run(generated)
    saved = generated.save(args.output) if args.output else ""
    if args.output:
        print("\nlog written to {}".format(saved))
    else:
        from .dataset import render

        print()
        print(render(generated.data, generated.route), end="")
    return 0 if generated.run.usable else 1


def _command_run(args: argparse.Namespace) -> int:
    import warnings

    from .classical import GenerationError, cost
    from .cost import ValidityWarning
    from .instances import InstanceError

    try:
        generated = _generate(args, args.instance)
    except (InstanceError, GenerationError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    _show_instance(generated.instance)
    _show_run(generated)
    saved = generated.save(args.output) if args.output else ""
    _show_log(generated, saved)

    chosen = _chosen_for(generated.route, _parse_assignments(args.parameter))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ValidityWarning)
            report = cost(generated, chosen)
    except (GenerationError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    report["warnings"] = [str(w.message) for w in caught]

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    _show_cost(report, chosen)
    if generated.route.note:
        print("\n" + generated.route.note)
    return 0


def _instances_in(directory: str) -> List[Path]:
    from .instances import InstanceError, detect

    found = []
    for path in sorted(Path(directory).expanduser().iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            detect(path)
        except InstanceError:
            continue
        found.append(path)
    return found


def _command_batch(args: argparse.Namespace) -> int:
    import warnings

    from .classical import GenerationError, cost
    from .cost import ValidityWarning
    from .instances import InstanceError

    paths = _instances_in(args.directory)
    if not paths:
        print("nothing in {} looks like an instance file".format(args.directory),
              file=sys.stderr)
        return 1

    overrides = _parse_assignments(args.parameter)
    rows: List[Tuple[str, str, int, Optional[float], str]] = []
    #: Totals are kept per unit, never across them, as a count and a sum.  A
    #: directory holding both a network and a knapsack produces gates and
    #: cycles, and adding those is the one thing this library exists to refuse.
    totals: Dict[str, List[float]] = {}
    truncations = 0

    for path in paths:
        try:
            generated = _generate(args, str(path))
        except InstanceError as error:
            rows.append((path.name, "unreadable", 0, None, str(error)))
            continue
        except GenerationError as error:
            rows.append((path.name, "no route", 0, None, str(error)))
            continue
        if not generated.run.usable:
            rows.append((path.name, str(generated.run.status), 0, None,
                         generated.run.reason or generated.run.advice()))
            continue
        chosen = _chosen_for(generated.route, overrides)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ValidityWarning)
                report = cost(generated, chosen)
        except (GenerationError, ValueError) as error:
            rows.append((path.name, "not costed", len(generated.run.records),
                         None, str(error)))
            continue
        unit = report["unit_label"]
        seen, summed = totals.get(unit, [0.0, 0.0])
        totals[unit] = [seen + 1, summed + report["total"]]
        if generated.run.truncated:
            truncations += 1
        rows.append((path.name, report["status"] or "complete",
                     report["logged_records"], report["total"], unit))
        if args.output:
            generated.save(str(Path(args.output) / (path.stem + ".log")))

    width = max(len(row[0]) for row in rows)
    print("{:<{w}}  {:<10} {:>7}  {}".format(
        "instance", "status", "records", "cost", w=width))
    for name, status, records, total, note in rows:
        value = "{:.6g} {}".format(total, note) if total is not None else note
        print("{:<{w}}  {:<10} {:>7}  {}".format(
            name, status, records, value, w=width))

    print()
    if not totals:
        print("nothing here could be costed")
        return 1
    for unit, (seen, total) in sorted(totals.items()):
        print("{:.6g} {} over {:.0f} instance{}".format(
            total, unit, seen, "" if seen == 1 else "s"))
    if len(totals) > 1:
        print("kept apart on purpose: these units do not add")
    if truncations:
        print("{} run{} cut off by the {:.0f}s budget, so {} total is an "
              "understatement as well as a bound -- rerun with --budget"
              .format(truncations, "" if truncations == 1 else "s",
                      args.budget, "that" if len(totals) == 1 else "each"))
    return 0


def _command_template(args: argparse.Namespace) -> int:
    from .dataset import template
    from .problems import get_route

    problem, _, route = args.route.partition("/")
    if not route:
        raise SystemExit("name a route as problem/route, not {!r}".format(
            args.route))
    print(template(get_route(problem, route), "json" if args.json else "csv"))
    return 0


def _add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    from .classical.budget import DEFAULT_SECONDS

    parser.add_argument("--problem", help="which problem to treat it as")
    parser.add_argument("--route", help="which route through that problem")
    parser.add_argument("--budget", type=float, default=DEFAULT_SECONDS,
                        help="seconds the classical run may take per instance "
                             "(default %(default).0f)")
    parser.add_argument("-o", "--output", help="where to keep the log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-benchmarking",
        description="Resource analysis of fault-tolerant quantum algorithms.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("serve", help="start the local interface (default)")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--no-browser", action="store_true",
                     help="do not open a browser window")
    run.set_defaults(handler=_command_serve)

    listing = sub.add_parser("list", help="what can be counted, and in what")
    listing.set_defaults(handler=_command_list)

    show = sub.add_parser("show", help="one routine in detail")
    show.add_argument("routine")
    show.set_defaults(handler=_command_show)

    cost = sub.add_parser("cost", help="evaluate one routine")
    cost.add_argument("routine")
    cost.add_argument("-u", "--unit", help="GATES, QUERIES, CYCLES, ...")
    cost.add_argument("-p", "--parameter", action="append", default=[],
                      metavar="NAME=VALUE")
    cost.add_argument("--json", action="store_true")
    cost.set_defaults(handler=_command_cost)

    formula = sub.add_parser("formula", help="print a cost expression")
    formula.add_argument("routine")
    formula.add_argument("-u", "--unit")
    formula.set_defaults(handler=_command_formula)

    make = sub.add_parser(
        "log", help="run the classical algorithm on an instance file and "
                    "write the log it produces")
    make.add_argument("instance", help="a DIMACS, knapsack, Matrix Market or "
                                       "MPS file")
    _add_instance_arguments(make)
    make.set_defaults(handler=_command_log)

    whole = sub.add_parser(
        "run", help="the whole loop: instance file, classical run, log, cost")
    whole.add_argument("instance")
    _add_instance_arguments(whole)
    whole.add_argument("-p", "--parameter", action="append", default=[],
                       metavar="NAME=VALUE",
                       help="epsilon, delta and anything else not measured")
    whole.add_argument("--json", action="store_true")
    whole.set_defaults(handler=_command_run)

    many = sub.add_parser(
        "batch", help="the same over a directory of instances, tabulated")
    many.add_argument("directory")
    _add_instance_arguments(many)
    many.add_argument("-p", "--parameter", action="append", default=[],
                      metavar="NAME=VALUE")
    many.set_defaults(handler=_command_batch)

    blank = sub.add_parser(
        "template", help="the columns a route needs, for logging it yourself")
    blank.add_argument("route", metavar="PROBLEM/ROUTE")
    blank.add_argument("--json", action="store_true")
    blank.set_defaults(handler=_command_template)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        args = parser.parse_args((argv or []) + ["serve"])
    try:
        return args.handler(args)
    except (KeyError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
