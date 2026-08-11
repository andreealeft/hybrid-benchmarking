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

Every routine from the thesis, the maximum-flow study, the quantum interior
point paper, and the boundary with Cade-style instrumented counts — 44 routines
across 51 implementations, in gates, oracle queries, cycles, iterations and
repetitions — plus a local interface and a command line over the same API.

```python
hb.capability_table()
```

```
routine                           gates    queries     cycles  iteration  repetitio      calls
----------------------------------------------------------------------------------------------
Cade-amplitude                        -          -          -          -          -          -
Cade-linalg                           -          -          -          -          -          -
Cade-max                              -          -          -          -          -          -
Cade-search                           -          -          -          -          -          -
CanEnterNFN                         yes          -          -          -          -          -
CanEnterNFP                         yes          -          -          -          -          -
CtrlU                               yes          -          -          -          -          -
Dinic                               yes          -        yes          -          -          -
FindColumn/steepest-edge            yes          -          -          -          -          -
FindColumn/dantzig                  yes          -          -          -          -          -
FindColumn/random                   yes          -          -          -          -          -
FindRow                             yes          -          -          -          -          -
HHL                                   -        yes          -          -          -          -
HamSim/qubitization                   -        yes          -          -          -          -
HamSim/berry                        yes          -          -          -          -          -
IPM/mnes                              -          -        yes          -          -          -
IPM/oss                               -          -        yes          -          -          -
Interfere                           yes          -          -          -          -          -
IsOptimal                           yes          -          -          -          -          -
IsUnbounded                         yes          -          -          -          -          -
IterRefine                            -          -          -          -          -          -
LCU                                 yes          -          -          -          -          -
OAA                                 yes          -          -          -          -          -
O_A                                   -          -          -          -          -          -
O_F                                   -          -          -          -          -          -
P_b                                   -          -          -          -          -          -
QAA                                   -          -          -        yes          -          -
QAA-fixed                           yes          -          -          -          -          -
QAE                                 yes          -          -          -          -          -
QFT                                 yes          -        yes          -          -          -
QFTAdd                              yes          -        yes          -          -          -
QFTSub                              yes          -        yes          -          -          -
QLS-Chebyshev                         -        yes          -          -          -          -
QLS-Fourier/via-qubitization          -        yes          -          -          -          -
QLS-Fourier/via-berry               yes          -          -          -          -          -
QLS-QSVT                              -        yes          -          -          -          -
QMF                                 yes          -          -          -          -          -
QPE                                 yes          -          -          -          -          -
QSearch                               -          -          -        yes          -          -
QSearchAll                            -          -          -        yes          -          -
QTG                                 yes          -        yes          -          -          -
QTGSearch                           yes          -        yes          -          -          -
RedCost                             yes          -          -          -          -          -
SignEstNFN                          yes          -          -          -          -          -
SignEstNFP                          yes          -          -          -          -          -
SimplexIter/steepest-edge           yes          -          -          -          -          -
SimplexIter/dantzig                 yes          -          -          -          -          -
SimplexIter/random                  yes          -          -          -          -          -
Tomography                            -          -          -          -        yes          -
VTAA                                  -          -          -          -          -          -
qBFS                                yes          -        yes          -          -          -
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

## The interface

```sh
hybrid-benchmarking
```

Starts a local server and opens a browser. Nothing is hosted and nothing is
uploaded: it binds the loopback interface and serves one browser on the same
machine. The page has no external resources at all, so it works offline.

Pick a routine, read its formula, see the regime it was derived in, type
parameters, get a number with its provenance attached — and a snippet that
reproduces the same call in a script:

```python
import hybrid_benchmarking as hb

hb.get('HamSim/berry').evaluate(hb.Unit.GATES, A_1=3, A_max=1, d=4, epsilon=0.001, t_sim=10)
```

Units a routine does not offer are shown with the reason rather than greyed
out — *no gate count: the oracle implementation is not fixed, so there is
nothing to count gates for* — because the reason is the interesting part.

Everything the interface does is also available without it:

```sh
hybrid-benchmarking list                     # what can be counted, and in what
hybrid-benchmarking show HamSim              # both constructions, with assumptions
hybrid-benchmarking formula QFT -u GATES     # the expression itself
hybrid-benchmarking cost QLS-Chebyshev -u QUERIES \
    -p d=4 -p kappa=10 -p epsilon=1e-8 -p x_norm=1
```

## Install

```sh
pip install -e ".[dev]"
python -m pytest
```

Python 3.9 or newer. The only runtime dependency is sympy; the interface is
served by the standard library, so a Python installation really is the only
prerequisite.
