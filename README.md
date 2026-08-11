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

This is early. The cost algebra and the amplification generic are implemented
and tested; the linear solvers, the simplex subroutines and the interface are
not yet.

```python
hb.capability_table()
```

```
routine         gates  oracle qu     cycles  iteration  repetitio
-----------------------------------------------------------------
HamSim              -        yes          -          -          -
QAA                 -          -          -        yes          -
QSearch             -          -          -        yes          -
QSearchAll          -          -          -        yes          -
```

Which units a routine offers is not a maintained table — it is whichever cost
formulas the routine actually has. A linear solver will show no gate count
because no gate formula exists for one until an oracle implementation is fixed.

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
