"""Cade-style instrumented counts, kept deliberately apart.

QUBRABENCH and its successor annotate a *classical* algorithm and count how
often it calls a subroutine that a quantum version would accelerate.  That
count is then converted into an expected quantum cost.  It is a genuine and
useful measurement, and it is not the same quantity as anything else in this
library.

The hazard is entirely one of naming.  Both traditions say "query".  Here a
query is a call to a sparse-access oracle, derived in closed form from a
lemma; there it is a call to a classically-defined subroutine, observed by
running the classical algorithm.  The two are not comparable, cannot be added,
and would be indistinguishable in a table if they shared a word -- which is a
worse kind of duplication than repeated code, because nothing would ever
surface it.

So they get their own unit.  :class:`~hybrid_benchmarking.provenance.Unit`
``SUBROUTINE_CALLS`` will not add to ``QUERIES``, and the cost algebra raises
rather than silently producing a number.

These entries carry no cost formulas.  Their counts come from instrumenting a
run, not from evaluating an expression, and the conversion factors belong to
another group's published work -- supplying them here would mean transcribing
results this library has not read.  The entries exist so the unit exists, so
the boundary is visible, and so a probe that instruments a classical run has
somewhere to report into.
"""

from __future__ import annotations

from ..registry import single

_SOURCE = "Cade, Folkertsma, Niesen and Weggemans, quantifying Grover " \
          "speed-ups beyond asymptotic analysis; implemented in QUBRABENCH"

_NOTE = " Counted by instrumenting a classical run, not by evaluating a " \
        "formula, and measured in subroutine calls -- which are not " \
        "sparse-access oracle queries and must never be compared with them."

CadeSearch = single(
    name="Cade-search",
    summary="Search a classically-defined predicate over a collection, with "
            "the classical and would-be-quantum call counts tracked side by "
            "side." + _NOTE,
    citation=_SOURCE,
    costs={},
)

CadeMax = single(
    name="Cade-max",
    summary="Maximum finding over a classically-defined key function." + _NOTE,
    citation=_SOURCE,
    costs={},
)

CadeAmplitude = single(
    name="Cade-amplitude",
    summary="Amplitude estimation of a classically-defined event." + _NOTE,
    citation=_SOURCE,
    costs={},
)

CadeLinalg = single(
    name="Cade-linalg",
    summary="Linear algebra routines instrumented in the same style." + _NOTE,
    citation=_SOURCE,
    costs={},
)
