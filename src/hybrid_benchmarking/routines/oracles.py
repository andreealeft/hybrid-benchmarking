"""Oracles and access models.

These are the unit of account for query counting, not things with a cost of
their own: a query count says how many times an algorithm reaches for the
matrix, and says nothing about what reaching costs.  That is exactly why the
linear solvers have no gate count.  Fix an implementation for these and gate
counts become available; leave them abstract and they do not.

They are registered with no cost formulas, so the capability table shows them
offering nothing.  That is the honest answer rather than a missing entry.
"""

from __future__ import annotations

from ..registry import single

_SPARSE_ACCESS = "sparse access input model, Section 5.1 of the thesis"

O_F = single(
    name="O_F",
    summary="Sparse access: given a row and an index l, return the column index "
            "of that row's l-th non-zero entry.",
    citation=_SPARSE_ACCESS,
    costs={},
)

O_A = single(
    name="O_A",
    summary="Sparse access: given a row and column, return the matrix entry, "
            "XORed into a register.",
    citation=_SPARSE_ACCESS,
    costs={},
)

P_b = single(
    name="P_b",
    summary="State preparation for the right-hand side. Costed as zero in the "
            "four-way solver comparison, on the grounds that it is identical "
            "across all four: defensible for ranking solvers, not for an "
            "absolute number.",
    citation=_SPARSE_ACCESS,
    costs={},
)
