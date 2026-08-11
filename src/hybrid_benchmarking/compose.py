"""Building an algorithm out of routines.

The published analyses in this library are compositions -- the quantum simplex
is a search wrapped around a sign estimate wrapped around a linear solve -- and
there is nothing special about them.  Anyone can assemble the same way, from a
description rather than from code.

Three moves, which is all the cost algebra offers and all it should:

*fill a slot*  A wrapper such as amplitude estimation costs a prefactor times
whatever it wraps.  Left alone it reports how many times it calls an unnamed
routine; fill the slot and it reports the composite.

*add*  Two counts of the same thing, run one after the other.

*multiply*  A count of repetitions times a count of what is repeated.

The composite carries the union of the parameters, the weaker of the bound
directions, every assumption either side made, and both validity domains.  That
is the part worth having: an assembled cost is exactly as hedged as its weakest
ingredient, and says so without anyone having to remember.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .cost import Cost, UnitMismatch
from .provenance import Unit
from .registry import get

Spec = Dict[str, Any]


def build(spec: Spec) -> Cost:
    """Turn a description into a cost.

    A leaf names a routine and a unit, optionally filling its slots::

        {"routine": "QAE", "unit": "GATES",
         "bind": {"oracle_gates": {"routine": "CanEnterNFP", "unit": "GATES"}}}

    A branch combines several::

        {"op": "+", "terms": [leaf, leaf]}
    """
    if "op" in spec:
        operator = spec["op"]
        terms = [build(term) for term in spec.get("terms", [])]
        if not terms:
            raise ValueError("a combination needs at least one term")
        total = terms[0]
        for term in terms[1:]:
            total = total + term if operator == "+" else total * term
        return total

    name = spec.get("routine")
    if not name:
        raise ValueError("every step names a routine")
    target = get(name)
    unit = Unit[spec["unit"]] if spec.get("unit") else None
    cost = target.cost(unit)

    fillings = {
        slot: build(inner) for slot, inner in (spec.get("bind") or {}).items()
    }
    return cost.bind(**fillings) if fillings else cost


def code(spec: Spec, values: Optional[Dict[str, Any]] = None) -> str:
    """The library call that reproduces an assembled cost."""
    lines = ["import hybrid_benchmarking as hb", ""]
    lines.append("cost = " + _expression(spec))
    if values is not None:
        arguments = ", ".join(
            "{}={}".format(name, json.dumps(value))
            for name, value in sorted(values.items())
        )
        lines.append("cost.evaluate({})".format(arguments))
    return "\n".join(lines)


def _expression(spec: Spec) -> str:
    if "op" in spec:
        joiner = " + " if spec["op"] == "+" else " * "
        return joiner.join(
            "(" + _expression(term) + ")" for term in spec.get("terms", [])
        )
    call = "hb.get({!r}).cost(hb.Unit.{})".format(spec["routine"], spec["unit"])
    fillings = spec.get("bind") or {}
    if fillings:
        inner = ", ".join(
            "{}={}".format(slot, _expression(sub))
            for slot, sub in sorted(fillings.items())
        )
        call += ".bind({})".format(inner)
    return call


def describe(spec: Spec) -> Dict[str, Any]:
    """Everything the interface needs to draw an assembled cost."""
    cost = build(spec)
    return {
        "unit": cost.unit.name,
        "unit_label": str(cost.unit),
        "parameters": list(cost.parameters),
        "formula": cost.formula(),
        "bound": str(cost.provenance.bound),
        "derivation": str(cost.provenance.derivation),
        "sources": list(cost.provenance.sources),
        "assumptions": list(cost.provenance.assumptions),
        "conditions": [
            {"message": c.message, "hard": c.hard}
            for c in cost.validity.conditions
        ],
        "open_slots": {name: unit.name for name, unit in cost.slots.items()},
        "snippet": code(spec),
    }


def entries() -> List[Dict[str, Any]]:
    """Every addressable thing with a cost, for a picker to offer."""
    from .registry import all_implementations

    return [
        {
            "path": impl.path if impl.name != "default" else impl.routine,
            "units": [u.name for u in impl.units],
        }
        for impl in all_implementations() if impl.units
    ]


def fillings_for(unit_name: str) -> List[str]:
    """Which routines can go into a slot counted in this unit."""
    from .registry import all_implementations

    unit = Unit[unit_name]
    return sorted({
        impl.path if impl.name != "default" else impl.routine
        for impl in all_implementations() if unit in impl.costs
    })
