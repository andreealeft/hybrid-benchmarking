"""The quantum simplex: Nannicini's subroutines, costed in gates.

Every one of these bottoms out in the same place -- a linear solve -- and the
solver is always the Fourier one evolving the matrix by Berry's reduction, never
by qubitization.  So each entry here is a prefactor determined by precision and
problem shape, multiplying ``QLS-Fourier/via-berry`` at a precision the caller's
tolerance is divided down into.

That division is the precision propagation the library exists to keep in one
place: the caller chooses one epsilon, and CanEnter hands the solver
``0.1 eps / sqrt(2)``, the steepest-edge comparison hands it
``eps / (10 c_max sqrt(2))``, and FindRow hands it ``delta / 2``.  Each of those
constants is a decision from the source, not a rounding.

All counts here are **lower bounds**: non-leading terms are dropped and every
assumption is tilted toward the quantum side, so the real cost of running these
is higher than what comes out.
"""

from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import Implementation, Routine, register, single
from ..validity import Validity, definition, much_less_than
from .amplification import qsearch_iterations
from .linsolve import fourier_gates

_NANNICINI = "Nannicini's quantum simplex; Chapter 4 and Appendix A of the thesis"



def _solver(epsilon: float, v: Dict[str, float]) -> float:
    """One linear solve at the given precision, in gates."""
    return fourier_gates(
        d=v["d"], kappa=v["kappa"], epsilon=epsilon,
        norm_1=v["A_1"], norm_max=v["A_max"],
    )


# ---------------------------------------------------------------------------
# core subroutines
# ---------------------------------------------------------------------------

def red_cost(v: Dict[str, float]) -> float:
    """Lemma 18: the reduced cost of a column costs at least one solve."""
    return _solver(v["epsilon"], v)


def sign_est_nfn(v: Dict[str, float], inner: float, epsilon: float) -> float:
    """Lemma 20: sign of an amplitude, with no false negatives."""
    return (5.0 * math.sqrt(3.0) * math.pi / epsilon - 1.0) * inner


def sign_est_nfp(v: Dict[str, float], inner: float, epsilon: float) -> float:
    """Lemma 21: no false positives.  Same shape, a factor of nine tighter
    precision inside the amplitude estimation, hence the larger prefactor."""
    return (45.0 * math.sqrt(3.0) * math.pi / epsilon - 1.0) * inner


def can_enter_nfn(v: Dict[str, float]) -> float:
    """Lemma 22: does this column have negative reduced cost, no false negatives."""
    epsilon = v["epsilon"]
    return (50.0 * math.sqrt(6.0) * math.pi / (11.0 * epsilon) - 1.0) * _solver(
        0.1 * epsilon / math.sqrt(2.0), v
    )


def can_enter_nfp(v: Dict[str, float]) -> float:
    """Lemma 23: the same, with no false positives."""
    epsilon = v["epsilon"]
    return (450.0 * math.sqrt(6.0) * math.pi / (11.0 * epsilon) - 1.0) * _solver(
        0.1 * epsilon / math.sqrt(2.0), v
    )


# ---------------------------------------------------------------------------
# main subroutines
# ---------------------------------------------------------------------------

def is_optimal(v: Dict[str, float]) -> float:
    """Lemma 8: is the current basis optimal.

    Amplitude estimation over a superposition of all non-basic columns, asking
    whether any of them could enter.
    """
    epsilon = v["epsilon"]
    non_basic = v["n"] - v["m"]
    return (
        (24.0 * math.sqrt(non_basic) - 1.0)
        * (450.0 * math.sqrt(6.0) * math.pi / (11.0 * epsilon) - 1.0)
        * _solver(0.1 * epsilon / math.sqrt(2.0), v)
    )


def find_column_random(v: Dict[str, float]) -> float:
    """Lemma 24: entering column under a random pivoting rule.

    A single quantum search over the non-basic columns, with CanEnter as the
    marker -- so the marked count is the number of columns that *could* enter,
    those with negative reduced cost, and not the ``t`` of Lemma 10.  That one
    marks a search over the ``m`` components of the pivot direction and would
    be the wrong quantity here in both meaning and range.

    The two were the same parameter until the instrumented GLPK settled it: it
    logs ``candidate_columns`` and ``u_positive_elements`` separately and feeds
    each to its own lemma.
    """
    epsilon = v["epsilon"]
    non_basic = v["n"] - v["m"]
    return (
        qsearch_iterations(non_basic, v["t_improving"])
        * (50.0 * math.sqrt(6.0) * math.pi / (11.0 * epsilon) - 1.0)
        * _solver(0.1 * epsilon / math.sqrt(2.0), v)
    )


def _pivot_sum(non_basic: int) -> float:
    """``sum_t n_Q(n - m, t) / (t + 1)`` -- the expected minimum-finding cost.

    Summed from ``t = 0``, following (A.25) and Lemma 25.  Lemma 9 as stated in
    chapter 4 starts the sum at ``t = 1``; the ``t = 0`` term is the largest,
    being the cost of a search that exhausts its schedule without finding
    anything, so the choice is not cosmetic.
    """
    count = int(non_basic)
    return sum(
        qsearch_iterations(count, t) / (t + 1.0) for t in range(0, count)
    )


def _minimum_finding_rounds(epsilon: float, non_basic: int) -> float:
    """Lemma 28: expected calls to quantum minimum finding with a cutoff."""
    return 3.0 * math.ceil(math.log(1.0 / epsilon, 3.0)) * _pivot_sum(non_basic)


def find_column_steepest_edge(v: Dict[str, float]) -> float:
    """Lemma 9: entering column under the quantum steepest-edge rule."""
    epsilon, c_max = v["epsilon"], v["c_max"]
    non_basic = int(v["n"] - v["m"])
    return (
        (40.0 * math.sqrt(3.0) * math.pi * c_max / epsilon - 1.0)
        * _minimum_finding_rounds(epsilon, non_basic)
        * _solver(epsilon / (10.0 * c_max * math.sqrt(2.0)), v)
    )


def find_column_dantzig(v: Dict[str, float]) -> float:
    """Lemma 25: entering column under the quantum Dantzig rule.

    Identical in shape to steepest edge; the pivot direction's norm enters the
    precision handed to the solver, which is the whole difference.
    """
    epsilon, c_max = v["epsilon"], v["c_max"]
    non_basic = int(v["n"] - v["m"])
    return (
        (40.0 * math.sqrt(3.0) * math.pi * c_max / epsilon - 1.0)
        * _minimum_finding_rounds(epsilon, non_basic)
        * _solver(epsilon / (v["u_norm"] * c_max * 10.0 * math.sqrt(2.0)), v)
    )


def is_unbounded(v: Dict[str, float]) -> float:
    """Lemma 10: is the problem unbounded, by searching for a positive component."""
    delta = v["delta"]
    return (
        qsearch_iterations(v["m"], v["t"])
        * (50.0 * math.sqrt(3.0) * math.pi / (18.0 * delta) - 1.0)
        * _solver(delta / 10.0, v)
    )


def find_row(v: Dict[str, float]) -> float:
    """Lemma 11: the leaving row, by binary search on the ratio test.

    Costed at ``r = 0``, where the search finds nothing and so runs its schedule
    to exhaustion -- the step that dominates, and the reason the marked count of
    zero has to be a meaningful input.
    """
    delta = v["delta"]
    return (
        qsearch_iterations(v["m"], 0)
        * (math.sqrt(3.0) * math.pi * v["u_norm"] / (2.0 * delta) - 1.0)
        * _solver(delta / 2.0, v)
    )


def simplex_iteration(v: Dict[str, float], rule: str) -> float:
    """One full iteration: optimality, entering column, unboundedness, leaving row."""
    find_column = {
        "steepest-edge": find_column_steepest_edge,
        "dantzig": find_column_dantzig,
        "random": find_column_random,
    }[rule]
    return (
        is_optimal(v) + find_column(v) + is_unbounded(v) + find_row(v)
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _lower_bound(expr: sp.Expr, kernel, *extra: str,
                 validity: Validity = None, slots=None) -> Cost:
    return Cost(
        slots=slots or {},
        expr=expr,
        unit=Unit.GATES,
        provenance=Provenance.of(
            Bound.LOWER, Derivation.ANALYTIC, _NANNICINI,
            assumptions=(
                "one Toffoli costs one gate",
                "state preparation is not counted",
                "non-leading terms dropped, in favour of the quantum side",
            ) + extra,
        ),
        validity=validity if validity is not None else _basic(),
        kernel=kernel,
    )


def _basic(tolerance: sp.Symbol = S.epsilon) -> Validity:
    return Validity((
        definition(sp.Lt(tolerance, 1), "precision must be below 1"),
        definition(sp.Ge(S.d, 1), "sparsity is at least 1"),
        definition(sp.Ge(S.kappa, 1), "condition number is at least 1"),
        much_less_than(tolerance, 1 / S.kappa),
    ))


def _shape(tolerance: sp.Symbol = S.epsilon) -> Validity:
    return _basic(tolerance) + Validity((
        definition(sp.Gt(S.n_vars, S.m_cons),
                   "a basis cannot use every column"),
    ))


def _display(*extra: sp.Symbol) -> sp.Expr:
    """A stand-in expression whose free symbols are exactly what the kernel needs.

    These counts are products of prefactors and a solver cost that is itself a
    search over a segment count; there is no closed form worth printing, so the
    displayed form names the ingredients and the kernel does the arithmetic.
    """
    base = (S.d, S.kappa, S.norm_A_1, S.norm_A_max)
    seen = list(base) + [s for s in extra if s not in base]
    return sp.Function("G")(*seen)


_SOLVE = "solve"

Interfere = single(
    name="Interfere",
    summary="Hadamard-interfere two unitaries into element-wise sums and "
            "differences of their amplitudes.",
    citation=_NANNICINI + " (Lemma 19)",
    costs={Unit.GATES: _lower_bound(
        S.inner_gates + S.inner_gates_2,
        lambda v: v["inner_gates"] + v["inner_gates_2"],
        validity=Validity(),
        slots={"inner_gates": Unit.GATES, "inner_gates_2": Unit.GATES},
    )},
)

SignEstNFN = single(
    name="SignEstNFN",
    summary="Is an amplitude positive, with no false negatives. Amplitude "
            "estimation on the interference of the state with a basis vector.",
    citation=_NANNICINI + " (Lemma 20)",
    built_from=("Interfere", "QAE"),
    costs={Unit.GATES: _lower_bound(
        (5 * sp.sqrt(3) * sp.pi / S.epsilon - 1) * S.inner_gates,
        lambda v: sign_est_nfn(v, v["inner_gates"], v["epsilon"]),
        validity=Validity((
            definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
        )),
        slots={"inner_gates": Unit.GATES},
    )},
)

SignEstNFP = single(
    name="SignEstNFP",
    summary="The same, with no false positives. Differs only in the precision "
            "handed to the amplitude estimation: a factor of nine, and so a "
            "factor of nine in cost.",
    citation=_NANNICINI + " (Lemma 21)",
    built_from=("Interfere", "QAE"),
    costs={Unit.GATES: _lower_bound(
        (45 * sp.sqrt(3) * sp.pi / S.epsilon - 1) * S.inner_gates,
        lambda v: sign_est_nfp(v, v["inner_gates"], v["epsilon"]),
        validity=Validity((
            definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
        )),
        slots={"inner_gates": Unit.GATES},
    )},
)

RedCost = single(
    name="RedCost",
    summary="Encode a column's reduced cost in the zeroth amplitude of a state.",
    citation=_NANNICINI + " (Lemma 18)",
    built_from=("QLS-Fourier/via-berry",),
    costs={Unit.GATES: _lower_bound(_display(S.epsilon), red_cost)},
)

CanEnterNFN = single(
    name="CanEnterNFN",
    summary="Does this column have negative reduced cost, with no false negatives.",
    citation=_NANNICINI + " (Lemma 22)",
    built_from=("RedCost", "SignEstNFN"),
    costs={Unit.GATES: _lower_bound(_display(S.epsilon), can_enter_nfn)},
)

CanEnterNFP = single(
    name="CanEnterNFP",
    summary="The same, with no false positives; a tighter internal precision.",
    citation=_NANNICINI + " (Lemma 23)",
    built_from=("RedCost", "SignEstNFP"),
    costs={Unit.GATES: _lower_bound(_display(S.epsilon), can_enter_nfp)},
)

IsOptimal = single(
    name="IsOptimal",
    summary="Is the current basis optimal, by amplitude estimation over all "
            "non-basic columns.",
    citation=_NANNICINI + " (Lemma 8)",
    built_from=("CanEnterNFP", "QAE"),
    costs={Unit.GATES: _lower_bound(
        _display(S.epsilon, S.n_vars, S.m_cons), is_optimal, validity=_shape(),
    )},
)

FindColumn = register(Routine(
    name="FindColumn",
    summary="Choose the entering column. The pivoting rule is the construction.",
    implementations=(
        Implementation(
            name="steepest-edge",
            summary="Quantum steepest edge: minimum finding over a comparison "
                    "of scaled reduced costs.",
            citation=_NANNICINI + " (Lemma 9)",
            built_from=("QMF", "RedCost", "Interfere", "SignEstNFN"),
            costs={Unit.GATES: _lower_bound(
                _display(S.epsilon, S.n_vars, S.m_cons, S.c_max),
                find_column_steepest_edge,
                "the pivoting sum runs from t = 0, per (A.25) and Lemma 25",
                validity=_shape(),
            )},
        ),
        Implementation(
            name="dantzig",
            summary="Quantum Dantzig: the same, comparing raw reduced costs, "
                    "with the pivot direction's norm in the solver precision.",
            citation=_NANNICINI + " (Lemma 25)",
            built_from=("QMF", "RedCost", "Interfere", "SignEstNFN"),
            costs={Unit.GATES: _lower_bound(
                _display(S.epsilon, S.n_vars, S.m_cons, S.c_max, S.u_norm),
                find_column_dantzig,
                validity=_shape(),
            )},
        ),
        Implementation(
            name="random",
            summary="A single quantum search for any improving column.",
            citation=_NANNICINI + " (Lemma 24)",
            built_from=("QSearch", "CanEnterNFN"),
            costs={Unit.GATES: _lower_bound(
                _display(S.epsilon, S.n_vars, S.m_cons, S.t_improving),
                find_column_random,
                "the marked count is the improving-column count of Lemma 24, "
                "not the positive-component count of Lemma 10",
                validity=_shape(),
            )},
        ),
    ),
))

IsUnbounded = single(
    name="IsUnbounded",
    summary="Is the problem unbounded, by searching for a positive component "
            "of the pivot direction.",
    citation=_NANNICINI + " (Lemma 10)",
    built_from=("QSearch", "SignEstNFN", "QLS-Fourier/via-berry"),
    costs={Unit.GATES: _lower_bound(
        _display(S.delta, S.m_cons, S.t), is_unbounded,
        validity=_basic(S.delta),
    )},
)

FindRow = single(
    name="FindRow",
    summary="The leaving row, by binary search on the ratio test.",
    citation=_NANNICINI + " (Lemma 11)",
    built_from=("QSearch", "SignEstNFN", "QLS-Fourier/via-berry"),
    costs={Unit.GATES: _lower_bound(
        _display(S.delta, S.m_cons, S.u_norm), find_row,
        "costed at r = 0, where the search finds nothing",
        validity=_basic(S.delta),
    )},
)

_ITER_SYMBOLS = (S.epsilon, S.delta, S.n_vars, S.m_cons, S.t, S.c_max,
                 S.u_norm)
#: The random rule searches the columns as well as the basis, so it needs both
#: marked counts where the other two rules need only Lemma 10's.
_RANDOM_SYMBOLS = _ITER_SYMBOLS + (S.t_improving,)

SimplexIter = register(Routine(
    name="SimplexIter",
    summary="One iteration of the quantum simplex: optimality test, entering "
            "column, unboundedness test, leaving row.",
    implementations=tuple(
        Implementation(
            name=rule,
            summary="Pivoting by {}.".format(rule.replace("-", " ")),
            citation=_NANNICINI + " (Algorithm 9)",
            built_from=("IsOptimal", "FindColumn/" + rule, "IsUnbounded",
                        "FindRow"),
            costs={Unit.GATES: _lower_bound(
                _display(*(_RANDOM_SYMBOLS if rule == "random"
                           else _ITER_SYMBOLS)),
                (lambda r: (lambda v: simplex_iteration(v, r)))(rule),
                validity=_shape(),
            )},
        )
        for rule in ("steepest-edge", "dantzig", "random")
    ),
))
