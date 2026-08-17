"""Running the classical algorithm ourselves, so there is a log to cost.

Everything else in this library begins with a log: condition numbers per simplex
iteration, layer sizes per breadth-first sweep, the dimension and sparsity of
each Newton system.  Nobody has those.  Producing them means instrumenting a
classical solver, which is a piece of work most people asking the question are
not going to do -- so the question could not be answered by anyone who did not
already have the answer.

This package closes that loop.  It reads an instance file, runs the classical
algorithm on this machine with the callbacks already in place, and writes the
log.  The log is not skipped: it stays the intermediate artefact, it is shown,
and anyone arriving with a log of their own keeps exactly the path they had.

Two things follow from doing it ourselves, and both are recorded on every number
that comes out rather than mentioned in a readme:

**These are not the published runs.**  The thesis logged GLPK; this logs a few
hundred lines of numpy.  Condition numbers and improving-column counts depend on
the implementation -- on its scaling, its factorisation, its tie-breaking -- so
the totals will differ from the published figures.  Every cost derived this way
carries :class:`~..provenance.Derivation.LOGGED` and the name of the
implementation that produced it.

**A run that was cut off is still data.**  See :mod:`.budget`.
"""

from __future__ import annotations

from .budget import DEFAULT_SECONDS, Budget, Run, Status
from .generate import (
    DEFAULT_ROUTE,
    GenerationError,
    Generated,
    cost,
    generate,
    generate_from_file,
    supported,
)

__all__ = [
    "DEFAULT_ROUTE",
    "DEFAULT_SECONDS",
    "Budget",
    "GenerationError",
    "Generated",
    "Run",
    "Status",
    "cost",
    "generate",
    "generate_from_file",
    "supported",
]
