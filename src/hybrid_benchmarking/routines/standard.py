"""Standard routines, costed in gates.

The generic wrappers of Appendix A.3: phase estimation, amplitude estimation,
minimum finding, the two amplifications, linear combination of unitaries, and
the controlled-unitary and state-preparation conventions everything else rests
on.

These are wrappers, so their cost is a prefactor times the cost of whatever
they wrap.  That inner cost arrives as an ordinary parameter -- ``inner_gates``
for a unitary, ``oracle_gates`` for a marker -- which keeps them usable on
their own: hand the wrapper a cost of one and it reports how many times it
calls its argument.

The gate model throughout is the thesis's: single-qubit gates, two-qubit
controlled rotations and Toffolis each count as one.  That is an assumption
recorded on every entry, not a fact.
"""

from __future__ import annotations

import math
from typing import Dict

import sympy as sp

from .. import symbols as S
from ..cost import Cost
from ..provenance import Bound, Derivation, Provenance, Unit
from ..registry import single
from ..validity import Validity, definition
from .amplification import qsearch_iterations

_SOURCE = "Appendix A.3 of the thesis"

#: Clock width, written as a function of what determines it so that a caller
#: supplies epsilon and delta rather than the derived register size.
_n_c = sp.Function("n_c")(S.epsilon, S.delta)

_GATE_MODEL = (
    "single-qubit gates, two-qubit rotations and Toffolis each cost one gate",
    "state preparation costs one gate",
    "non-leading terms dropped, in favour of the quantum side",
)


def clock_qubits(epsilon: float, delta: float) -> int:
    """``n_c = ceil(log2(1/eps) + log2(1 + 1/(2 delta)))``.

    The clock register has to resolve the phase to ``eps`` and survive a
    failure probability ``delta``; both enter logarithmically, and the register
    then enters the cost exponentially.
    """
    return int(math.ceil(
        math.log2(1.0 / epsilon) + math.log2(1.0 + 1.0 / (2.0 * delta))
    ))


def qpe_gates(epsilon: float, delta: float, inner_gates: float) -> float:
    """Lemma 30.  The controlled powers dominate: ``2^{n_c} - 1`` of them."""
    n_c = clock_qubits(epsilon, delta)
    return n_c + (2 ** n_c - 1) * inner_gates


def grover_operator_gates(qubits: int, prepare_gates: float,
                          oracle_gates: float) -> float:
    """``G[Q] = Q[xi] + 2 G[A] + 2 G_1 + G_2 + 2 (n - 1) G_T``.

    Two reflections: one about the marked subspace, one about the prepared
    state, which costs the preparation twice.
    """
    return oracle_gates + 2.0 * prepare_gates + 2.0 + 1.0 + 2.0 * (qubits - 1)


def qae_gates(epsilon: float, delta: float, qubits: int, prepare_gates: float,
              oracle_gates: float) -> float:
    """Lemma 29: prepare once, then phase-estimate the Grover operator."""
    grover = grover_operator_gates(qubits, prepare_gates, oracle_gates)
    return prepare_gates + qpe_gates(epsilon, delta, grover)


def minimum_finding_queries(list_length: int, first_rank: int = 0) -> float:
    """Expected oracle calls to find a minimum.

    ``sum_s <n_Q>(s) / (s + 1)`` over the ranks a candidate can hold.

    The sources disagree on where the sum starts: Lemma 27 and Lemma 9 write
    ``s = 1``, while (A.23) and Lemma 25 write ``s = 0``.  It is not cosmetic --
    the ``s = 0`` term is the largest, being the cost of the final search that
    finds nothing and exhausts its schedule, which is what tells the algorithm
    it is done.  That search does happen, so ``first_rank`` defaults to zero;
    pass one to reproduce the other reading.
    """
    count = int(list_length)
    return sum(
        qsearch_iterations(count, s) / (s + 1.0)
        for s in range(first_rank, count)
    )


def minimum_finding_with_cutoff(list_length: int, epsilon: float,
                                first_rank: int = 0) -> float:
    """Lemma 28: repeat until the failure probability is below ``epsilon``."""
    return (3.0 * math.ceil(math.log(1.0 / epsilon, 3.0))
            * minimum_finding_queries(list_length, first_rank))


def qmf_gates(list_length: int, qubits: int, prepare_gates: float,
              oracle_gates: float, epsilon: float = None) -> float:
    """Lemma 26.  Each round costs a Grover operator, a fresh uniform state
    over ``qubits`` Hadamards, and one measurement."""
    rounds = minimum_finding_queries(list_length) if epsilon is None \
        else minimum_finding_with_cutoff(list_length, epsilon)
    per_round = grover_operator_gates(qubits, prepare_gates, oracle_gates) \
        + qubits + 1.0
    return rounds * per_round


def _fixed_schedule(success_probability: float) -> int:
    """``m = floor(pi / 4 theta)`` with ``sin^2 theta = p``.

    The schedule of Lemmas 32 and 33, used where the success probability is
    known in advance -- unlike the geometric schedule in
    :mod:`.amplification`, which is what you need when it is not.
    """
    if not 0 < success_probability <= 1:
        raise ValueError("the success probability must lie in (0, 1]")
    theta = math.asin(math.sqrt(success_probability))
    return int(math.floor(math.pi / (4.0 * theta)))


def qaa_gates(success_probability: float, qubits: int, prepare_gates: float,
              oracle_gates: float) -> float:
    """Lemma 32, the fixed-schedule amplitude amplification gate count."""
    m = _fixed_schedule(success_probability)
    return (m * oracle_gates + (2 * m + 1) * prepare_gates
            + 2 * m + m + 2 * m * (qubits - 1))


def oaa_gates(success_probability: float, ancillas: int,
              prepare_gates: float) -> float:
    """Lemma 33.  No marker: the reflection is about the ancilla register, so
    the cost is the preparation and the reflections alone."""
    m = _fixed_schedule(success_probability)
    return 4 * m + 2 * m + (2 * ancillas - 1) + (2 * m + 1) * prepare_gates


def lcu_gates(terms: int, inner_gates: float, prepare_gates: float = 1.0) -> float:
    """Lemma 31.  Dominated by the controlled unitary over all the terms."""
    return 2.0 * (terms - 1) + inner_gates + prepare_gates


def controlled_gates(controls: int, inner_gates: float) -> float:
    """``G[ctrl_n - U] >= 2 (n - 1) G_T + G[U]`` -- the AND of the controls."""
    return 2.0 * (controls - 1) + inner_gates


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _bound(expr, kernel, *notes, validity=None, slots=None) -> Cost:
    return Cost(
        slots=slots or {},
        expr=expr,
        unit=Unit.GATES,
        provenance=Provenance.of(
            Bound.LOWER, Derivation.ANALYTIC, _SOURCE,
            assumptions=_GATE_MODEL + notes,
        ),
        validity=validity if validity is not None else Validity(),
        kernel=kernel,
    )


QPE = single(
    name="QPE",
    summary="Phase estimation of a unitary to precision epsilon with failure "
            "probability delta.",
    citation=_SOURCE + " (Lemma 30)",
    built_from=("QFT",),
    costs={Unit.GATES: _bound(
        _n_c + (2 ** _n_c - 1) * S.inner_gates,
        lambda v: qpe_gates(v["epsilon"], v["delta"], v["inner_gates"]),
        "the inverse transform is replaced by Hadamards and not counted",
        slots={"inner_gates": Unit.GATES},
        validity=Validity((
            definition(sp.Lt(S.epsilon, 1), "precision must be below 1"),
            definition(sp.Lt(S.delta, 1), "failure probability must be below 1"),
        )),
    )},
)

QAE = single(
    name="QAE",
    summary="Amplitude estimation: phase estimation applied to the Grover "
            "operator of a prepared state.",
    citation=_SOURCE + " (Lemma 29)",
    built_from=("QPE",),
    costs={Unit.GATES: _bound(
        sp.Function("G_QAE")(S.epsilon, S.delta, S.n_qubits, S.prepare_gates,
                             S.oracle_gates),
        lambda v: qae_gates(v["epsilon"], v["delta"], int(v["n_qubits"]),
                            v["prepare_gates"], v["oracle_gates"]),
        slots={"oracle_gates": Unit.GATES, "prepare_gates": Unit.GATES},
    )},
)

QMF = single(
    name="QMF",
    summary="Quantum minimum finding: repeated search against a falling "
            "threshold, until a search finds nothing.",
    citation=_SOURCE + " (Lemmas 26 to 28)",
    built_from=("QSearch",),
    costs={Unit.GATES: _bound(
        sp.Function("G_QMF")(S.X, S.n_qubits, S.prepare_gates, S.oracle_gates),
        lambda v: qmf_gates(int(v["X"]), int(v["n_qubits"]),
                            v["prepare_gates"], v["oracle_gates"]),
        "the rank sum starts at zero; Lemma 27 and Lemma 9 start it at one",
        slots={"oracle_gates": Unit.GATES, "prepare_gates": Unit.GATES},
        validity=Validity((
            definition(sp.Ge(S.X, 2), "a list of one has nothing to minimise"),
        )),
    )},
)

QAA_gates = single(
    name="QAA-fixed",
    summary="Amplitude amplification on a known success probability, with the "
            "fixed schedule m = floor(pi / 4 theta). Distinct from QAA, which "
            "grows its schedule geometrically because it is not told p.",
    citation=_SOURCE + " (Lemma 32)",
    costs={Unit.GATES: _bound(
        sp.Function("G_QAA")(S.p, S.n_qubits, S.prepare_gates, S.oracle_gates),
        lambda v: qaa_gates(v["p"], int(v["n_qubits"]), v["prepare_gates"],
                            v["oracle_gates"]),
        slots={"oracle_gates": Unit.GATES, "prepare_gates": Unit.GATES},
        validity=Validity((
            definition(sp.Le(S.p, 1), "p is a probability"),
        )),
    )},
)

OAA = single(
    name="OAA",
    summary="Oblivious amplitude amplification: the reflection is about the "
            "ancilla register, so no marker is needed.",
    citation=_SOURCE + " (Lemma 33)",
    costs={Unit.GATES: _bound(
        sp.Function("G_OAA")(S.p, S.ancillas, S.prepare_gates),
        lambda v: oaa_gates(v["p"], int(v["ancillas"]), v["prepare_gates"]),
        slots={"prepare_gates": Unit.GATES},
        validity=Validity((
            definition(sp.Le(S.p, 1), "p is a probability"),
        )),
    )},
)

LCU = single(
    name="LCU",
    summary="Linear combination of unitaries: apply a weighted sum of "
            "unitaries probabilistically, flagging success.",
    citation=_SOURCE + " (Lemma 31)",
    built_from=("P_b",),
    costs={Unit.GATES: _bound(
        2 * (S.terms - 1) + S.inner_gates + 1,
        lambda v: lcu_gates(int(v["terms"]), v["inner_gates"]),
        "the state preparation over the coefficients is heuristic and not "
        "expected to dominate",
        slots={"inner_gates": Unit.GATES},
    )},
)

ControlledUnitary = single(
    name="CtrlU",
    summary="A unitary controlled on several qubits, by computing their AND "
            "into an ancilla first.",
    citation=_SOURCE + " (A.34)",
    costs={Unit.GATES: _bound(
        2 * (S.n_qubits - 1) + S.inner_gates,
        lambda v: controlled_gates(int(v["n_qubits"]), v["inner_gates"]),
        slots={"inner_gates": Unit.GATES},
    )},
)
