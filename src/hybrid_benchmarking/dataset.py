"""The log format: what an instrumented classical run has to write down.

Hybrid benchmarking is not a calculator.  The method is to run the classical
algorithm on a real instance, record what it did, and feed those recordings into
the analytic cost formulas.  So the unit of input here is a **record** -- one
simplex iteration, one breadth-first sweep, one Newton system -- carrying the
quantities that iteration's cost depends on.

Two shapes, because two shapes are what people naturally produce:

**CSV**, one row per record, columns named after the quantities.  This is what a
logging callback inside a solver emits, so instance-wide values simply repeat
down their column and nobody has to think about it::

    kappa,d,A_1,A_max,t,c_max,u_norm,n,m
    3.1,4,3.0,1.0,12,1.0,1.4,200,50
    5.7,4,3.2,1.0,9,1.0,1.6,200,50

**JSON**, where a record holds something that is not a number -- a list of layer
sizes, a vector of item values::

    {"instance": {"vertices": 300},
     "records": [{"layers": [1, 3, 5, 2]}, {"layers": [1, 2, 4]}]}

:func:`template` prints the exact columns a given route needs, so the format is
never something to guess at.  Nothing here is uploaded: the interface runs on
the same machine as the file and reads it in place.

A log this library generated itself carries one thing more: :attr:`Dataset
.generated`, saying which classical implementation produced it and whether that
run finished.  It travels in the file -- a ``generated`` block in JSON, a
``# generated:`` header line in CSV -- so a log that was cut off after fifty
iterations still says so a week later, when the number it produced is being read
next to one that was not.  A log written by hand simply has none, and everything
here behaves exactly as it did before.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .compose import build
from .problems import Field as Descriptor
from .problems import Route
from .provenance import Bound, Derivation, Provenance

#: Prefix of the CSV comment line carrying :attr:`Dataset.generated`.
_GENERATED = "# generated:"


@dataclass(frozen=True)
class Dataset:
    """What a classical run wrote down."""

    records: Tuple[Dict[str, Any], ...] = ()
    instance: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    #: Present only when this library generated the log: the classical
    #: implementation that ran, whether it finished, and what it cost to find
    #: out.  See :mod:`.classical`.
    generated: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)


class FormatError(ValueError):
    """The file does not carry what the chosen route needs."""


# ---------------------------------------------------------------------------
# describing the format
# ---------------------------------------------------------------------------

def required(route: Route) -> Tuple[Descriptor, ...]:
    """Everything a route needs, wherever it comes from."""
    return route.per_instance + route.per_record + route.chosen


def template(route: Route, fmt: str = "csv") -> str:
    """A blank log in the right shape, with every column named and explained.

    Handing someone this is the whole answer to "what format?"
    """
    per_row = [f.name for f in route.per_record]
    constant = [f.name for f in route.per_instance]

    if fmt == "json":
        skeleton = {
            "instance": {f.name: _example(f) for f in route.per_instance},
            "records": [{f.name: _example(f) for f in route.per_record}],
        }
        body = json.dumps(skeleton, indent=2)
    else:
        columns = per_row + constant
        body = ",".join(columns) + "\n" + ",".join(
            str(_example(f)) for f in route.per_record + route.per_instance
        )

    notes = ["# {} -- {}".format(f.name, f.help)
             for f in route.per_record + route.per_instance]
    if route.chosen:
        notes.append("# supplied when you run it, not logged: "
                     + ", ".join(f.name for f in route.chosen))
    return "\n".join(notes + ["", body])


def _example(descriptor: Descriptor) -> Any:
    text = descriptor.example or "0"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load(path: str) -> Dataset:
    """Read a log, in whichever of the two shapes it is written."""
    location = Path(path).expanduser()
    if not location.exists():
        raise FormatError("no such file: {}".format(location))
    text = location.read_text()
    if location.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        return _load_json(text, str(location))
    return _load_csv(text, str(location))


def _load_json(text: str, source: str) -> Dataset:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise FormatError("not valid JSON: {}".format(error))
    if not isinstance(payload, dict):
        raise FormatError("expected an object with 'records', found a list")
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise FormatError("'records' must be a list")
    return Dataset(tuple(records), dict(payload.get("instance", {})), source,
                   dict(payload.get("generated", {})))


def _generated_from_comments(text: str) -> Dict[str, Any]:
    """The ``# generated:`` header of a log this library wrote, if there is one."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        if stripped.startswith(_GENERATED):
            try:
                payload = json.loads(stripped[len(_GENERATED):].strip())
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}
    return {}


def _load_csv(text: str, source: str) -> Dataset:
    generated = _generated_from_comments(text)
    rows = list(csv.DictReader(
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))
    if not rows:
        raise FormatError("no rows found")
    records = [
        {name: _number(value) for name, value in row.items()
         if name and value not in (None, "")}
        for row in rows
    ]
    # A column that never changes is a property of the instance, not of any one
    # iteration.  Splitting them out is only presentational -- both end up in
    # the same call -- but it makes a log self-describing.
    constant = {
        name for name in records[0]
        if all(record.get(name) == records[0][name] for record in records)
    }
    instance = {name: records[0][name] for name in constant}
    varying = tuple(
        {name: value for name, value in record.items() if name not in constant}
        for record in records
    )
    return Dataset(varying, instance, source, generated)


def summarise(generated: Dict[str, Any]) -> str:
    """One sentence saying who wrote this log and whether they finished."""
    return "{} -- {}, {} record{} in {}s".format(
        generated.get("implementation", "an instrumented classical run"),
        generated.get("status", "complete"),
        generated.get("records", 0),
        "" if generated.get("records") == 1 else "s",
        generated.get("elapsed_seconds", 0),
    )


def is_flat(data: Dataset) -> bool:
    """Whether every value is a scalar, and so whether CSV can hold this log."""
    return not any(
        isinstance(value, (list, dict))
        for record in tuple(data.records) + (data.instance,)
        for value in record.values()
    )


def render(data: Dataset, route: Optional[Route] = None) -> str:
    """A generated log, as the text that goes in the file.

    CSV where every value is a scalar, JSON where one of them is a list -- which
    is not a preference but a fact about the route: Dinic logs the sizes of a
    sweep's layers, and that is not a column.  Either way the header carries
    :attr:`Dataset.generated`, so the file remembers how the run that produced
    it went.
    """
    if is_flat(data):
        return _render_csv(data, route)
    return _render_json(data)


def _render_json(data: Dataset) -> str:
    payload: Dict[str, Any] = {}
    if data.generated:
        payload["generated"] = data.generated
    payload["instance"] = dict(data.instance)
    payload["records"] = [dict(record) for record in data.records]
    return json.dumps(payload, indent=2) + "\n"


def _render_csv(data: Dataset, route: Optional[Route]) -> str:
    columns: List[str] = []
    for record in data.records:
        for name in record:
            if name not in columns:
                columns.append(name)
    for name in data.instance:
        if name not in columns:
            columns.append(name)

    lines = []
    if data.generated:
        lines.append("# " + summarise(data.generated))
    if route is not None:
        explained = {f.name: f.help for f in route.per_record + route.per_instance}
        lines += ["# {} -- {}".format(name, explained[name])
                  for name in columns if name in explained]
    if data.generated:
        # Last, and on one line: this is the machine-readable twin of the
        # sentence at the top, and it is long enough to bury everything else.
        lines.append("{} {}".format(_GENERATED,
                                    json.dumps(data.generated, sort_keys=True)))
    lines.append(",".join(columns))
    for record in data.records:
        values = dict(data.instance)
        values.update(record)
        lines.append(",".join(_text(values.get(name, "")) for name in columns))
    return "\n".join(lines) + "\n"


def _text(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write(data: Dataset, path: str, route: Optional[Route] = None) -> str:
    """Write a generated log where the user can see it, and hand back the path."""
    location = Path(path).expanduser()
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(render(data, route))
    return str(location)


def _number(text: str) -> Any:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ---------------------------------------------------------------------------
# checking and running
# ---------------------------------------------------------------------------

def check(route: Route, data: Dataset,
          chosen: Optional[Dict[str, Any]] = None) -> List[str]:
    """Everything the route needs and this log does not have, by name."""
    chosen = chosen or {}
    supplied = set(data.instance) | set(chosen)
    if data.records:
        supplied |= set(data.records[0])

    missing = [
        "{} ({})".format(f.name, f.help)
        for f in required(route) if f.name not in supplied
    ]
    if route.per_record and not data.records:
        missing.append("at least one record -- this route costs a run "
                       "iteration by iteration")
    return missing


def parameters_for(route: Route, record: Dict[str, Any],
                   data: Dataset, chosen: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble one call's worth of parameters.

    Problem shape becomes program shape here: a graph with V vertices and E
    edges becomes a linear program with as many columns and rows as the
    formulation implies, rather than the caller having to work that out.
    """
    values: Dict[str, Any] = {}
    values.update(data.instance)
    values.update(chosen)
    values.update(record)
    if route.shape:
        values.update(route.shape(values))
    for logged, parameter in route.renames.items():
        if logged in values:
            values[parameter] = values[logged]
    return values


def origin(data: Dataset) -> Optional[Provenance]:
    """What the log itself says about where its numbers came from.

    A hand-written log says nothing, and this returns ``None`` -- the cost keeps
    exactly the provenance its lemmas give it.  A log this library generated says
    which classical implementation ran and whether it finished, and both belong
    on the answer: the numbers are :class:`~.provenance.Derivation.LOGGED` rather
    than analytic, and a run that was cut off is a lower bound for a reason that
    has nothing to do with the lemmas being lower bounds.
    """
    stated = data.generated
    if not stated:
        return None

    status = str(stated.get("status", "complete"))
    truncated = status == "truncated"
    assumptions = []
    if truncated:
        assumptions.append(
            "the classical run was cut off after {} of an unfinished solve, so "
            "this counts only what was logged and understates the whole solve "
            "-- a lower bound for a second reason, unrelated to the lemmas"
            .format(_iterations(stated))
        )
    for note in stated.get("assumptions", ()):
        assumptions.append(str(note))

    # Not every generated log is a measurement.  The knapsack circuits read the
    # binary representations of an instance's own profits and weights, and
    # nothing was run to find those, so calling them logged would hedge a number
    # that is not hedged.  The run says which it is.
    try:
        derivation = Derivation[str(stated.get("derivation", "LOGGED")).upper()]
    except KeyError:
        derivation = Derivation.LOGGED

    return Provenance.of(
        bound=Bound.LOWER if truncated else Bound.EXACT,
        derivation=derivation,
        source=str(stated.get("implementation", "an instrumented classical run")),
        assumptions=assumptions,
    )


def _iterations(stated: Dict[str, Any]) -> str:
    count = stated.get("records")
    if count is None:
        return "part"
    return "{} iteration{}".format(count, "" if count == 1 else "s")


def run(route: Route, data: Dataset,
        chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cost a logged run.

    Most routes cost one record at a time and the run is their sum -- that is
    what a whole solve costs.  A few take the run as a whole, because their
    formula does: Dinic's cost is stated over every sweep at once.
    """
    chosen = dict(chosen or {})
    missing = check(route, data, chosen)
    if missing:
        raise FormatError("the log is missing: " + "; ".join(missing))

    cost = build({"routine": route.target, "unit": route.unit.name})
    wanted = set(cost.parameters)
    source = origin(data)

    def call(values: Dict[str, Any]) -> Any:
        return cost.evaluate(**{k: v for k, v in values.items() if k in wanted})

    if route.collects:
        source_field, parameter = route.collects
        values = parameters_for(route, {}, data, chosen)
        values[parameter] = [record[source_field] for record in data.records]
        evaluated = call(values)
        return _report(route, [evaluated.value], evaluated, True, data, source)

    if not route.per_record:
        evaluated = call(parameters_for(route, {}, data, chosen))
        return _report(route, [evaluated.value], evaluated, True, data, source)

    per_record: List[float] = []
    last = None
    for record in data.records:
        last = call(parameters_for(route, record, data, chosen))
        per_record.append(last.value)
    return _report(route, per_record, last, False, data, source)


def _report(route: Route, values: Sequence[float], sample, whole_run: bool,
            data: Dataset, source: Optional[Provenance]) -> Dict[str, Any]:
    total = float(sum(values))
    provenance = sample.provenance
    if source is not None:
        provenance = provenance.combine(source)
    return {
        "route": route.key,
        "unit": route.unit.name,
        "unit_label": str(route.unit),
        "total": total,
        "records": len(values),
        #: What the classical run wrote down, which is not the same as the
        #: number of values summed: a route that costs the run as a whole
        #: evaluates once over every record at once.
        "logged_records": len(data.records),
        "largest": max(values) if values else 0.0,
        "mean": total / len(values) if values else 0.0,
        "per_record": [float(v) for v in values],
        "whole_run": whole_run,
        "bound": str(provenance.bound),
        "derivation": str(provenance.derivation),
        "provenance": provenance.describe(),
        "assumptions": list(provenance.assumptions),
        "generated": dict(data.generated),
        "status": str(data.generated.get("status", "")) if data.generated else "",
        "note": route.note,
    }
