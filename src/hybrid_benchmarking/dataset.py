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


@dataclass(frozen=True)
class Dataset:
    """What a classical run wrote down."""

    records: Tuple[Dict[str, Any], ...] = ()
    instance: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

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
    return Dataset(tuple(records), dict(payload.get("instance", {})), source)


def _load_csv(text: str, source: str) -> Dataset:
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
    return Dataset(varying, instance, source)


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

    def call(values: Dict[str, Any]) -> Any:
        return cost.evaluate(**{k: v for k, v in values.items() if k in wanted})

    if route.collects:
        source_field, parameter = route.collects
        values = parameters_for(route, {}, data, chosen)
        values[parameter] = [record[source_field] for record in data.records]
        evaluated = call(values)
        return _report(route, [evaluated.value], evaluated, whole_run=True)

    if not route.per_record:
        evaluated = call(parameters_for(route, {}, data, chosen))
        return _report(route, [evaluated.value], evaluated, whole_run=True)

    per_record: List[float] = []
    last = None
    for record in data.records:
        last = call(parameters_for(route, record, data, chosen))
        per_record.append(last.value)
    return _report(route, per_record, last, whole_run=False)


def _report(route: Route, values: Sequence[float], sample,
            whole_run: bool) -> Dict[str, Any]:
    total = float(sum(values))
    return {
        "route": route.key,
        "unit": route.unit.name,
        "unit_label": str(route.unit),
        "total": total,
        "records": len(values),
        "largest": max(values) if values else 0.0,
        "mean": total / len(values) if values else 0.0,
        "per_record": [float(v) for v in values],
        "whole_run": whole_run,
        "bound": str(sample.provenance.bound),
        "provenance": sample.provenance.describe(),
        "assumptions": list(sample.provenance.assumptions),
        "note": route.note,
    }
