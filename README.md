# hybrid-benchmarking

Resource analysis of fault-tolerant quantum algorithms without a quantum
computer. Quantum subroutines with their costs, composable into published
analyses, and honest about where every number came from.

The shortest useful thing you can do needs no data at all:

```python
import hybrid_benchmarking as hb

hb.get("QSearch").evaluate(X=1_000_000, t=1)
# <Cost 1411.13 iterations -- exact, analytic -- after Boyer-Brassard-Hoyer-Tapp
#  schedule; Lemma 6 of the thesis -- assuming the number of marked elements t
#  is known>
```

That is the expected number of Grover iterations to find one marked element
among a million, when the algorithm is not told how many marked elements there
are — about 1.4 times the square root of the list length.

## What is here so far

The cost algebra, the amplification generic, both constructions of Hamiltonian
simulation, the four functional quantum linear solvers, the quantum simplex subroutines
with all three pivoting rules, and the quantum tree generator with its Fourier
arithmetic. Quantum breadth-first search, the interior point pipeline and the
interface are not yet written.

```python
hb.capability_table()
```

```
routine                           gates    queries     cycles  iteration  repetitio      calls
----------------------------------------------------------------------------------------------
CanEnterNFN                         yes          -          -          -          -          -
CanEnterNFP                         yes          -          -          -          -          -
FindColumn/steepest-edge            yes          -          -          -          -          -
FindColumn/dantzig                  yes          -          -          -          -          -
FindColumn/random                   yes          -          -          -          -          -
FindRow                             yes          -          -          -          -          -
HHL                                   -        yes          -          -          -          -
HamSim/qubitization                   -        yes          -          -          -          -
HamSim/berry                        yes          -          -          -          -          -
IsOptimal                           yes          -          -          -          -          -
IsUnbounded                         yes          -          -          -          -          -
O_A                                   -          -          -          -          -          -
O_F                                   -          -          -          -          -          -
P_b                                   -          -          -          -          -          -
QAA                                   -          -          -        yes          -          -
QFT                                 yes          -        yes          -          -          -
QFTAdd                              yes          -        yes          -          -          -
QFTSub                              yes          -        yes          -          -          -
QLS-Chebyshev                         -        yes          -          -          -          -
QLS-Fourier/via-qubitization          -        yes          -          -          -          -
QLS-Fourier/via-berry               yes          -          -          -          -          -
QLS-QSVT                              -        yes          -          -          -          -
QSearch                               -          -          -        yes          -          -
QSearchAll                            -          -          -        yes          -          -
QTG                                 yes          -        yes          -          -          -
QTGSearch                           yes          -        yes          -          -          -
RedCost                             yes          -          -          -          -          -
SimplexIter/steepest-edge           yes          -          -          -          -          -
SimplexIter/dantzig                 yes          -          -          -          -          -
SimplexIter/random                  yes          -          -          -          -          -
```

Which units a routine offers is not a maintained table — it is whichever cost
formulas exist. The linear solvers show no gate count because no gate formula
exists for one until an oracle implementation is fixed; the oracles themselves
show nothing at all, which is the honest answer rather than a missing entry.

### Reproducing the published comparison

```python
for name in ("HHL", "QLS-Fourier", "QLS-Chebyshev", "QLS-QSVT"):
    print(name, hb.get(name).evaluate(hb.Unit.QUERIES,
                                      d=4, kappa=50.0, epsilon=1e-8,
                                      x_norm=1.0, A_max=1.0).value)
```

```
HHL             9.26e+16
QLS-Fourier     1.39e+09
QLS-Chebyshev   6.03e+07
QLS-QSVT        2.28e+07
```

HHL orders of magnitude above the rest and excluded on that basis; Chebyshev and
QSVT below 10^8; Fourier an order above them; QSVT lowest — the qualitative
result of chapter 5, from an independent reimplementation.

## Three ideas the code is built around

**Every count knows what it is.** A cost carries its unit, how tight it is
(exact, lower bound, estimate), how it was derived, and under what assumptions.
Composition propagates all of it, so a total is exactly as hedged as its
weakest ingredient and says so. Adding gates to oracle queries raises an error;
multiplying iterations by gates-per-iteration gives gates.

**Amplification is one thing.** Quantum search, amplitude amplification and
find-all-marked are the same expression with different parameters. They are
derived from a single kernel in `routines/amplification.py`, and the test suite
asserts the relationships between them rather than trusting that three
independent implementations agree.

**Formulas say where they are valid.** Each routine declares the regime it was
derived in. Parameters that are merely outside that regime produce a warning
recorded on the result's provenance; parameters that are meaningless — more
marked elements than list entries — are refused outright.

## Install

```sh
pip install -e ".[dev]"
python -m pytest
```

Python 3.9 or newer. The only runtime dependency is sympy.
