"""hybrid-benchmarking.

Resource analysis of fault-tolerant quantum algorithms without a quantum
computer: subroutines with their costs, composable into published analyses, and
honest about where every number came from.

The shortest useful thing you can do needs no data at all::

    >>> import hybrid_benchmarking as hb
    >>> hb.get("QSearch").evaluate(X=1_000_000, t=1).value  # doctest: +SKIP
    1216.6...

which is the expected number of Grover iterations to find one marked element
among a million, when the algorithm is not told how many marked elements there
are.
"""

from __future__ import annotations

from .cost import Cost, UnitMismatch, ValidityWarning, exact, lower_bound
from .provenance import Bound, Derivation, Provenance, Unit
from .registry import (
    Implementation,
    Routine,
    all_implementations,
    all_routines,
    capability_table,
    get,
    names,
    register,
    single,
)
from .compose import build as compose, code as compose_code, describe as compose_describe
from .dataset import Dataset, FormatError, load, render, run, template, write
from .problems import PROBLEMS, Problem, Route, get_problem, get_route
from .validity import Condition, Validity
from . import symbols
from . import routines  # registers everything  # noqa: F401
from .instances import Instance, InstanceError
from .instances import read as read_instance
from .classical import Budget, Generated, GenerationError, generate, generate_from_file

__version__ = "0.1.0.dev0"

__all__ = [
    "Bound",
    "Condition",
    "Cost",
    "Derivation",
    "Implementation",
    "Provenance",
    "Routine",
    "Unit",
    "UnitMismatch",
    "Validity",
    "ValidityWarning",
    "Budget",
    "Dataset",
    "FormatError",
    "Generated",
    "GenerationError",
    "Instance",
    "InstanceError",
    "PROBLEMS",
    "Problem",
    "Route",
    "all_implementations",
    "all_routines",
    "capability_table",
    "exact",
    "generate",
    "generate_from_file",
    "get",
    "lower_bound",
    "names",
    "read_instance",
    "register",
    "render",
    "single",
    "write",
    "compose",
    "compose_code",
    "compose_describe",
    "get_problem",
    "get_route",
    "load",
    "run",
    "symbols",
    "template",
]
