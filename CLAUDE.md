# Working on hybrid-benchmarking

Resource analysis of fault-tolerant quantum algorithms without quantum hardware.
Reimplements, from their published lemmas, the analyses in Andreea-Iulia
Lefterovici's PhD thesis *Hybrid benchmarking of quantum algorithms* (Hannover
2026), the quantum-BFS maximum-flow study, and Binkowski's quantum interior
point paper — as one library where each result exists once.

**Read this before writing code.** What follows is judgement that is expensive
to reconstruct and invisible in the source.

## The one thing that matters

Every number carries what it is worth. A cost knows its unit, how tight it is,
how it was derived, what it assumes, and the regime its formula holds in — and
all of that survives composition, so an assembled cost is exactly as hedged as
its weakest ingredient. **A plausible number with lost provenance is the failure
mode this library exists to prevent.** When in doubt, make the caveat explicit
in the code rather than in a comment or a commit message.

Concretely, four rules that are already enforced and must stay enforced:

1. **Units refuse to mix.** Gates cannot be added to oracle queries. Cade-style
   instrumented `SUBROUTINE_CALLS` cannot be added to sparse-access `QUERIES` —
   both are called "queries" in the literature, they are not the same quantity,
   and that collision is the single most dangerous thing in the merge.
2. **Routines carry named implementations.** Hamiltonian simulation appears
   twice: Berry et al.'s fractional-query reduction (gates, used by the quantum
   simplex) and Low–Chuang qubitization (queries, used by the linear solvers).
   They are different algorithms. A composition names which one it uses; that
   is what `HamSim/berry` versus `HamSim/qubitization` is for.
3. **Validity is two-tiered.** Parameters outside the regime a formula was
   derived in produce a warning recorded on the result. Parameters with no
   meaning — more marked elements than list entries — are refused. Do not
   collapse the two.
4. **Reimplement from lemmas; never vendor.** The original repositories are
   validation fixtures, not inputs. This is also what keeps GLPK's GPL out of
   the distribution.

## When a source is ambiguous

Record the ambiguity; do not pick quietly. Five have come up, each ruled by
Andreea and each noted on the entries that depend on it:

| Where | The problem | Ruling |
|---|---|---|
| (B.74) vs Lemma 16 | j₀'s inner factor is linear in one slot, quadratic in the other — B.74 contradicts itself | **Quadratic.** Agrees with Childs–Kothari–Somma and with Binkowski's Lemma 1 |
| Lemma 17 | `n_1/x` defined with three arguments, called with two | **Keep the `d/2κ` rescaling** — it encodes the block-encoding sub-normalisation |
| Lemma 6 vs Lemma 13 | Same product term; one carries `m_k/2` and truncates, the other does not | **An error, not a unit difference — direction still unresolved.** Each keeps its own paper's convention behind `HALF_IN_QSEARCH`, so no published number moves |
| Lemma 9/27 vs (A.23)/Lemma 25 | Minimum-finding rank sum starts at 1 in two places, 0 in two others | **From 0** — the t=0 term is the terminating search that finds nothing, and it is the largest |
| (C.18) vs (C.25), (C.24) vs (C.28) | Comparison strategy reads bits of `w` in one and `w−1` in the other; cycle weights grouped differently | Transcribed as written, asymmetry flagged in the docstrings |

Two further corrections already made, both worth not re-breaking:

- **qBFS and Dinic report gates only.** Lemma 1 counts cycles, but the paper
  converts at one cycle per gate and reports gates. Offering both units implied
  two independent results where there is one number wearing two labels. QTG
  *does* keep both, because Appendix C derives its gate and cycle counts
  separately.
- **LP-formulated problems use the standard form.** Vertex cover written
  naturally has one variable per vertex and one constraint per edge, so any
  graph with more edges than vertices asks the simplex for a basis wider than
  the matrix. After a surplus per inequality and a slack per bound:
  `n = 2V + E`, `m = V + E`. The validity check caught this; keep it able to.

## Layout

```
src/hybrid_benchmarking/
  provenance.py   Unit, Bound, Derivation, Provenance — and how they combine
  cost.py         Cost: symbolic expression + unit + provenance + validity;
                  add, multiply, bind a slot, evaluate
  validity.py     Conditions, hard (refuse) versus regime (warn)
  registry.py     Routine -> Implementation -> costs; "Name/implementation"
  symbols.py      One canonical sympy symbol per quantity, shared everywhere
  compose.py      Assemble routines: fill a slot, add, multiply
  problems.py     Seven problems under friendly names, and routes through them
  dataset.py      The log format: records from an instrumented classical run
  web.py, cli.py  Interface and command line — clients, holding no logic
  routines/       The registry itself, one module per family
```

Costs that are closed forms are pure sympy. Costs that are not — the
amplification schedule, anything depending on an instance's bit patterns —
carry a numeric `kernel` alongside a display expression: the expression is what
the reader sees, the kernel is what evaluates.

## Conventions

- Python ≥ 3.9 (stock macOS `python3`), `sympy` and `numpy` only, `pytest`.
  The interface is standard-library `http.server` with a page that fetches
  nothing — someone offline must get the same tool. There is a test for that.
- Tests state what the mathematics claims, not what the code happens to do.
  The strongest ones assert relationships between results: that quantum search
  is truncated amplification times one half, that a hand-assembled
  `SimplexIter` equals the registered one, that Lemma 16 and Binkowski's
  Lemma 1 agree and stop agreeing under the rejected reading.
- Commit messages explain why, in prose. No bullet lists of files changed.
- Andreea prefers open discussion to multiple-choice questions, and asked for
  coding decisions to be made without checking in. Surface anything that
  changes the *science*; decide the rest.

## Where it stands

Complete: 44 routines / 51 implementations, all of Appendices A, B and C, the
maximum-flow study, the interior point pipeline, the Cade boundary, the
composition layer, the problem-first entry point, the log format, a local web
interface and a CLI. 313 tests.

Not done, and the next piece of work: **the tool cannot yet produce the logs it
asks for.** Users have a graph or a knapsack instance, not a condition number.
Closing that loop means reading instance files and running the classical
algorithm ourselves, instrumented, on the user's machine — generating the log
rather than replacing it, and showing it to them.

Two things to hold onto when that lands:

- Our Python simplex is not GLPK. Logged κ and improving-column counts depend
  on the implementation, so results will differ from the published figures. Say
  so: `Derivation.LOGGED` exists for this and nothing uses it yet.
- A truncated run is still data. Cutting a solve off after N iterations gives a
  cost that is a lower bound for a *second*, unrelated reason. Record it.
