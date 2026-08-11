"""Command line entry point.

``hybrid-benchmarking`` with no arguments starts the local interface, since
that is what someone who has just installed it wants.  The other subcommands
are the same operations without a browser, so anything the interface can do is
scriptable -- which is what makes "too large for the packaged tool" a command
rather than an email.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence

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
