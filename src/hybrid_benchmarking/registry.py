"""The registry.

Subroutines are the product; the published analyses are compositions of them.
A registry entry knows what it costs, in which units it can be costed at all,
what it is built from, and where it comes from in the literature.

Which units an entry offers is not a hand-maintained table -- it is whatever
cost formulas the entry actually has.  A linear solver has no gate count
because no gate formula exists for it until an oracle implementation is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

from .cost import Cost
from .provenance import Unit


@dataclass(frozen=True)
class Routine:
    """One quantum subroutine, oracle, or composition of them."""

    name: str
    summary: str
    costs: Dict[Unit, Cost]
    citation: str = ""
    built_from: Tuple[str, ...] = ()

    @property
    def units(self) -> Tuple[Unit, ...]:
        return tuple(self.costs)

    @property
    def parameters(self) -> Tuple[str, ...]:
        seen = []
        for cost in self.costs.values():
            for name in cost.parameters:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def cost(self, unit: Optional[Unit] = None) -> Cost:
        if unit is None:
            if len(self.costs) != 1:
                raise ValueError(
                    "{} can be costed in {}; say which".format(
                        self.name, ", ".join(str(u) for u in self.units)
                    )
                )
            return next(iter(self.costs.values()))
        if unit not in self.costs:
            raise ValueError(
                "{} has no {} count -- available: {}".format(
                    self.name, unit, ", ".join(str(u) for u in self.units) or "none"
                )
            )
        return self.costs[unit]

    def evaluate(self, unit: Optional[Unit] = None, strict: bool = False,
                 **values: float) -> Cost:
        return self.cost(unit).evaluate(strict=strict, **values)

    def __repr__(self) -> str:
        return "<Routine {} [{}]>".format(
            self.name, ", ".join(str(u) for u in self.units)
        )


_REGISTRY: Dict[str, Routine] = {}


def register(routine: Routine) -> Routine:
    if routine.name in _REGISTRY:
        raise ValueError("{} is already registered".format(routine.name))
    _REGISTRY[routine.name] = routine
    return routine


def get(name: str) -> Routine:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            "no routine {!r}; known: {}".format(name, ", ".join(sorted(_REGISTRY)))
        )


def names() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def all_routines() -> Iterator[Routine]:
    for name in names():
        yield _REGISTRY[name]


def capability_table() -> str:
    """What can be counted, derived from what formulas exist."""
    units = (Unit.GATES, Unit.QUERIES, Unit.CYCLES, Unit.ITERATIONS,
             Unit.REPETITIONS)
    width = max([len(n) for n in names()] + [8])
    header = "routine".ljust(width) + "".join(
        str(u)[:9].rjust(11) for u in units
    )
    lines = [header, "-" * len(header)]
    for routine in all_routines():
        row = routine.name.ljust(width)
        for unit in units:
            row += ("yes" if unit in routine.costs else "-").rjust(11)
        lines.append(row)
    return "\n".join(lines)
