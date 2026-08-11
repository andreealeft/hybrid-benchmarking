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
from .validity import Condition, Validity
from . import symbols
from . import routines  # registers everything  # noqa: F401

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
    "all_implementations",
    "all_routines",
    "capability_table",
    "exact",
    "get",
    "lower_bound",
    "names",
    "register",
    "single",
    "symbols",
]
