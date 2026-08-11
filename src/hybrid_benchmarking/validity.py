"""Where a cost formula is allowed to be used.

Pulling a subroutine out of its paper strips its assumptions.  The query-count
lemmas for the linear solvers hold only while the precision sits far below the
inverse condition number; evaluate them outside that regime and you get a
number that looks perfectly reasonable and means nothing.  Each routine
therefore declares the region its formula was derived in, and the caller is
told when it steps outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import sympy as sp


@dataclass(frozen=True)
class Condition:
    """A predicate on the parameters, with a human explanation.

    ``hard`` separates two things that both look like "outside the valid
    range".  A definitional condition -- more marked elements than list
    entries, a probability above one -- describes input that has no meaning,
    and evaluating anyway produces a number that is not wrong so much as
    nonsense; those refuse.  A regime condition -- a precision that is meant
    to sit far below the inverse condition number -- describes input the
    formula was not derived for but can still be computed on; those warn, and
    the warning is recorded on the result.
    """

    predicate: sp.Basic
    message: str
    hard: bool = False

    def holds(self, values: Dict[str, object]) -> Optional[bool]:
        """True / False, or None when the parameters do not decide it."""
        subs = {sym: values[sym.name] for sym in self.predicate.free_symbols
                if sym.name in values}
        result = self.predicate.subs(subs)
        if result in (sp.true, True):
            return True
        if result in (sp.false, False):
            return False
        return None

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Validity:
    """The conjunction of conditions under which a cost formula is derived."""

    conditions: Tuple[Condition, ...] = ()

    def violations(self, values: Dict[str, object]) -> Tuple[Condition, ...]:
        """Conditions that the given parameters definitely break."""
        return tuple(c for c in self.conditions if c.holds(values) is False)

    def hard_violations(self, values: Dict[str, object]) -> Tuple[Condition, ...]:
        """Broken conditions that make the input meaningless."""
        return tuple(c for c in self.violations(values) if c.hard)

    def soft_violations(self, values: Dict[str, object]) -> Tuple[Condition, ...]:
        """Broken conditions that only take the formula out of its regime."""
        return tuple(c for c in self.violations(values) if not c.hard)

    def undecided(self, values: Dict[str, object]) -> Tuple[Condition, ...]:
        """Conditions the given parameters are not enough to check."""
        return tuple(c for c in self.conditions if c.holds(values) is None)

    def __add__(self, other: "Validity") -> "Validity":
        seen = list(self.conditions)
        for c in other.conditions:
            if c not in seen:
                seen.append(c)
        return Validity(tuple(seen))

    def __bool__(self) -> bool:
        return bool(self.conditions)


def much_less_than(a: sp.Basic, b: sp.Basic, factor: int = 10) -> Condition:
    """Encode a paper's ``<<`` with an explicit slack factor.

    Papers write ``eps << 1/kappa`` and leave the margin to the reader.  A
    library cannot, so the factor is named here rather than buried: the
    condition checked is ``a < b / factor``.
    """
    return Condition(
        sp.Lt(a, b / factor),
        "{} << {} (read as a factor of {})".format(a, b, factor),
        hard=False,
    )


def definition(predicate: sp.Basic, message: str) -> Condition:
    """A condition without which the parameters have no meaning."""
    return Condition(predicate, message, hard=True)


def at_least(a: sp.Basic, b: sp.Basic, hard: bool = False) -> Condition:
    return Condition(sp.Ge(a, b), "{} >= {}".format(a, b), hard=hard)


def at_most(a: sp.Basic, b: sp.Basic, hard: bool = False) -> Condition:
    return Condition(sp.Le(a, b), "{} <= {}".format(a, b), hard=hard)
