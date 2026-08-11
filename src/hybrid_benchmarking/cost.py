"""The cost algebra.

A cost is a symbolic expression plus everything needed to know what it means:
its unit, how tight it is, how it was derived, and the regime its formula was
derived in.  Costs compose -- adding or multiplying them carries all of that
along, so a composite count is exactly as hedged as its weakest ingredient and
says so.

Some costs are not closed forms.  The amplification overhead is a convergent
sum whose terms involve ``floor`` and ``min``; there is a formula to display
but no expression a computer algebra system will usefully reduce.  Such costs
carry a numeric *kernel* alongside the expression: the expression is what the
reader sees, the kernel is what gets evaluated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import sympy as sp

from .provenance import Bound, Derivation, Provenance, Unit
from .validity import Validity

Number = Union[int, float, sp.Expr]
Kernel = Callable[[Dict[str, float]], float]


class ValidityWarning(UserWarning):
    """Raised as a warning when a cost is evaluated outside its derived regime."""


class UnitMismatch(TypeError):
    """Attempt to add counts of different things."""


@dataclass(frozen=True)
class Cost:
    """A resource count that knows what it is."""

    expr: sp.Expr
    unit: Unit
    provenance: Provenance = field(default_factory=Provenance)
    validity: Validity = field(default_factory=Validity)
    kernel: Optional[Kernel] = None
    #: Inputs the kernel needs that are not scalars and so cannot appear as
    #: symbols -- the profit and weight vectors of a knapsack instance, say,
    #: whose individual bit patterns enter the circuit cost.  They are required
    #: by :meth:`evaluate` exactly like the symbolic parameters.
    extra_parameters: Tuple[str, ...] = ()

    # -- introspection -------------------------------------------------------

    @property
    def free_symbols(self) -> Tuple[sp.Symbol, ...]:
        return tuple(sorted(self.expr.free_symbols, key=lambda s: s.name))

    @property
    def parameters(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.free_symbols) + self.extra_parameters

    @property
    def is_numeric(self) -> bool:
        return not self.expr.free_symbols

    @property
    def value(self) -> float:
        if not self.is_numeric:
            raise ValueError(
                "still symbolic in {}; call evaluate() first".format(
                    ", ".join(self.parameters)
                )
            )
        return float(self.expr)

    def formula(self) -> str:
        return sp.pretty(self.expr, use_unicode=True)

    def latex(self) -> str:
        return sp.latex(self.expr)

    # -- algebra -------------------------------------------------------------

    def _with(self, **changes) -> "Cost":
        return Cost(
            expr=changes.get("expr", self.expr),
            unit=changes.get("unit", self.unit),
            provenance=changes.get("provenance", self.provenance),
            validity=changes.get("validity", self.validity),
            kernel=changes.get("kernel", self.kernel),
            extra_parameters=changes.get("extra_parameters",
                                         self.extra_parameters),
        )

    def __add__(self, other: Union["Cost", Number]) -> "Cost":
        if not isinstance(other, Cost):
            return self._with(expr=self.expr + sp.sympify(other))
        if self.unit is not other.unit:
            raise UnitMismatch(
                "cannot add {} to {}".format(other.unit, self.unit)
            )
        left, right = self, other
        kernel = None
        if left.kernel or right.kernel:
            def kernel(values):
                return left._numeric(values) + right._numeric(values)
        return Cost(
            expr=left.expr + right.expr,
            unit=left.unit,
            provenance=left.provenance.combine(right.provenance),
            validity=left.validity + right.validity,
            kernel=kernel,
            extra_parameters=tuple(dict.fromkeys(
                left.extra_parameters + right.extra_parameters)),
        )

    __radd__ = __add__

    def __mul__(self, other: Union["Cost", Number]) -> "Cost":
        if not isinstance(other, Cost):
            return self._with(expr=self.expr * sp.sympify(other))

        # Multiplying two counts only makes sense when one of them is a
        # multiplier: iterations x gates-per-iteration comes out in gates.
        if self.unit.is_multiplier:
            unit = other.unit
        elif other.unit.is_multiplier:
            unit = self.unit
        else:
            raise UnitMismatch(
                "cannot multiply {} by {}; one factor must be a multiplier "
                "(iterations or repetitions)".format(self.unit, other.unit)
            )

        left, right = self, other
        kernel = None
        if left.kernel or right.kernel:
            def kernel(values):
                return left._numeric(values) * right._numeric(values)
        return Cost(
            expr=left.expr * right.expr,
            unit=unit,
            provenance=left.provenance.combine(right.provenance),
            validity=left.validity + right.validity,
            kernel=kernel,
            extra_parameters=tuple(dict.fromkeys(
                left.extra_parameters + right.extra_parameters)),
        )

    __rmul__ = __mul__

    # -- evaluation ----------------------------------------------------------

    def _numeric(self, values: Dict[str, float]) -> float:
        if self.kernel is not None:
            return float(self.kernel(values))
        subs = {s: values[s.name] for s in self.free_symbols if s.name in values}
        result = self.expr.subs(subs)
        if hasattr(result, "doit"):
            result = result.doit()
        return float(result.evalf())

    def evaluate(self, strict: bool = False, **values: float) -> "Cost":
        """Substitute parameters and return a numeric cost.

        Checks the validity domain first.  Stepping outside it is a warning,
        not an error, but it is recorded on the result's provenance so the
        number cannot later be mistaken for one that was in-regime.
        """
        missing = [name for name in self.parameters if name not in values]
        if missing:
            raise ValueError("missing parameters: " + ", ".join(missing))

        impossible = self.validity.hard_violations(values)
        if impossible:
            raise ValueError(
                "these parameters have no meaning: "
                + "; ".join(c.message for c in impossible)
            )

        provenance = self.provenance
        broken = self.validity.soft_violations(values)
        if broken:
            note = "evaluated outside its derived regime: " + "; ".join(
                c.message for c in broken
            )
            provenance = provenance.assuming(note)
            message = "{} -- the returned number may not mean anything".format(note)
            if strict:
                raise ValueError(message)
            warnings.warn(message, ValidityWarning, stacklevel=2)

        return Cost(
            expr=sp.Float(self._numeric(values)),
            unit=self.unit,
            provenance=provenance,
            validity=self.validity,
        )

    def subs(self, **values: Number) -> "Cost":
        """Partially substitute, symbolically.  Drops any numeric kernel."""
        subs = {s: sp.sympify(values[s.name]) for s in self.free_symbols
                if s.name in values}
        return Cost(
            expr=self.expr.subs(subs),
            unit=self.unit,
            provenance=self.provenance,
            validity=self.validity,
            kernel=None,
        )

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_numeric:
            head = "{:.6g} {}".format(float(self.expr), self.unit)
        else:
            head = "{} [{}]".format(self.unit, ", ".join(self.parameters))
        return "<Cost {} -- {}>".format(head, self.provenance.describe())


def exact(expr: sp.Expr, unit: Unit, source: str = "", **kwargs) -> Cost:
    """An exact analytic count."""
    return Cost(
        expr=sp.sympify(expr),
        unit=unit,
        provenance=Provenance.of(Bound.EXACT, Derivation.ANALYTIC, source,
                                 kwargs.pop("assumptions", ())),
        **kwargs,
    )


def lower_bound(expr: sp.Expr, unit: Unit, source: str = "", **kwargs) -> Cost:
    """A minimum: the algorithm cannot cost less than this."""
    return Cost(
        expr=sp.sympify(expr),
        unit=unit,
        provenance=Provenance.of(Bound.LOWER, Derivation.ANALYTIC, source,
                                 kwargs.pop("assumptions", ())),
        **kwargs,
    )
