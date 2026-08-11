"""The registry: routines, their implementations, and what each costs.

A *routine* is a task -- "simulate a Hamiltonian", "amplify a success
probability". An *implementation* is a named construction that performs it.
The distinction is not bookkeeping. Hamiltonian simulation appears in this work
twice, as Berry et al.'s reduction to the fractional-query model and as
Low-Chuang qubitization, and they are not interchangeable: the quantum simplex
gate counts are derived from the first, the linear solver query counts from the
second. Collapsing them into one entry would let a composition silently pick up
the wrong construction, which is the class of error this library exists to make
impossible.

So costs hang off implementations, never off routines directly, and a
composition names the implementation it is built on.

Addressing::

    get("HamSim")                 -> the routine
    get("HamSim/qubitization")    -> one implementation of it

A routine with a single implementation can be used directly, so the common case
stays short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple, Union

from .cost import Cost
from .provenance import Unit


@dataclass(frozen=True)
class Implementation:
    """One named construction of a routine, with what it costs."""

    name: str
    summary: str
    costs: Dict[Unit, Cost]
    citation: str = ""
    built_from: Tuple[str, ...] = ()
    routine: str = ""  # filled in by Routine

    @property
    def units(self) -> Tuple[Unit, ...]:
        return tuple(self.costs)

    @property
    def path(self) -> str:
        return "{}/{}".format(self.routine, self.name) if self.routine else self.name

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
                        self.path, ", ".join(str(u) for u in self.units) or "nothing"
                    )
                )
            return next(iter(self.costs.values()))
        if unit not in self.costs:
            raise ValueError(
                "{} has no {} count -- available: {}".format(
                    self.path, unit,
                    ", ".join(str(u) for u in self.units) or "none",
                )
            )
        return self.costs[unit]

    def evaluate(self, unit: Optional[Unit] = None, strict: bool = False,
                 **values: float) -> Cost:
        return self.cost(unit).evaluate(strict=strict, **values)

    def __repr__(self) -> str:
        return "<Implementation {} [{}]>".format(
            self.path, ", ".join(str(u) for u in self.units) or "no cost"
        )


@dataclass(frozen=True)
class Routine:
    """A task, together with every construction of it that is implemented."""

    name: str
    summary: str
    implementations: Tuple[Implementation, ...] = ()

    @property
    def units(self) -> Tuple[Unit, ...]:
        seen = []
        for impl in self.implementations:
            for unit in impl.units:
                if unit not in seen:
                    seen.append(unit)
        return tuple(seen)

    @property
    def parameters(self) -> Tuple[str, ...]:
        seen = []
        for impl in self.implementations:
            for name in impl.parameters:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def implementation(self, name: Optional[str] = None) -> Implementation:
        if name is None:
            if len(self.implementations) != 1:
                raise ValueError(
                    "{} has several constructions ({}); say which".format(
                        self.name,
                        ", ".join(i.name for i in self.implementations),
                    )
                )
            return self.implementations[0]
        for impl in self.implementations:
            if impl.name == name:
                return impl
        raise ValueError(
            "{} has no construction {!r} -- known: {}".format(
                self.name, name, ", ".join(i.name for i in self.implementations)
            )
        )

    def cost(self, unit: Optional[Unit] = None,
             implementation: Optional[str] = None) -> Cost:
        if implementation is None and len(self.implementations) > 1 and unit is not None:
            # A unit can pick the construction out on its own when only one
            # construction offers it -- which is the usual case, since the
            # constructions were derived for different cost models.
            offering = [i for i in self.implementations if unit in i.costs]
            if len(offering) == 1:
                return offering[0].cost(unit)
        return self.implementation(implementation).cost(unit)

    def evaluate(self, unit: Optional[Unit] = None, strict: bool = False,
                 implementation: Optional[str] = None, **values: float) -> Cost:
        return self.cost(unit, implementation).evaluate(strict=strict, **values)

    def __repr__(self) -> str:
        return "<Routine {} [{}]>".format(
            self.name, ", ".join(i.name for i in self.implementations) or "empty"
        )


_REGISTRY: Dict[str, Routine] = {}


def register(routine: Routine) -> Routine:
    if routine.name in _REGISTRY:
        raise ValueError("{} is already registered".format(routine.name))
    # Stamp each implementation with its routine so it can report a full path.
    stamped = Routine(
        name=routine.name,
        summary=routine.summary,
        implementations=tuple(
            Implementation(
                name=i.name, summary=i.summary, costs=i.costs,
                citation=i.citation, built_from=i.built_from,
                routine=routine.name,
            )
            for i in routine.implementations
        ),
    )
    _REGISTRY[routine.name] = stamped
    return stamped


def single(name: str, summary: str, costs: Dict[Unit, Cost], citation: str = "",
           built_from: Tuple[str, ...] = ()) -> Routine:
    """Register a routine that has exactly one construction.

    The construction is named ``default`` and need never be mentioned again.
    """
    return register(Routine(
        name=name,
        summary=summary,
        implementations=(Implementation(
            name="default", summary=summary, costs=costs,
            citation=citation, built_from=built_from,
        ),),
    ))


def get(path: str) -> Union[Routine, Implementation]:
    """Look up ``"Routine"`` or ``"Routine/implementation"``."""
    if "/" in path:
        routine_name, impl_name = path.split("/", 1)
        return _lookup(routine_name).implementation(impl_name)
    return _lookup(path)


def _lookup(name: str) -> Routine:
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


def all_implementations() -> Iterator[Implementation]:
    for routine in all_routines():
        for impl in routine.implementations:
            yield impl


def capability_table() -> str:
    """What can be counted, derived from which cost formulas actually exist."""
    units = (Unit.GATES, Unit.QUERIES, Unit.CYCLES, Unit.ITERATIONS,
             Unit.REPETITIONS, Unit.SUBROUTINE_CALLS)
    rows = []
    for impl in all_implementations():
        label = impl.routine if impl.name == "default" else impl.path
        rows.append((label, impl))
    width = max([len(label) for label, _ in rows] + [8])
    header = "routine".ljust(width) + "".join(
        str(u).split()[-1][:9].rjust(11) for u in units
    )
    lines = [header, "-" * len(header)]
    for label, impl in rows:
        line = label.ljust(width)
        for unit in units:
            line += ("yes" if unit in impl.costs else "-").rjust(11)
        lines.append(line)
    return "\n".join(lines)
