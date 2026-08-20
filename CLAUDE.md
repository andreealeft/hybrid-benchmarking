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

Record the ambiguity; do not pick quietly. Where two readings are each
defensible, **take the one giving the lower count** — every number here is meant
to be a bound favourable to the quantum side, and a bound built on the larger of
two candidates is not the bound it claims to be. That rule is a tie-breaker
between defensible readings, not a licence to adopt a statement that
contradicts itself. Seven have come up, each ruled by
Andreea and each noted on the entries that depend on it:

| Where | The problem | Ruling |
|---|---|---|
| (B.74) vs Lemma 16 | j₀'s inner factor is linear in one slot, quadratic in the other — B.74 contradicts itself | **Quadratic.** Agrees with Childs–Kothari–Somma and with Binkowski's Lemma 1 |
| Lemma 17 | `n_1/x` defined with three arguments, called with two | **Keep the `d/2κ` rescaling** — it encodes the block-encoding sub-normalisation |
| Lemma 6 vs Lemma 13 | Same product term; one carries `m_k/2` and truncates, the other does not | **Both carry the half** (`HALF_ROUND_COUNT`). Neither statement contradicts itself, so the standing rule decides: where two defensible readings disagree, take the lower count. Halves every solver's query count; changes no ordering, since all four carry the same overhead |
| Lemma 9/27 vs (A.23)/Lemma 25 | Minimum-finding rank sum starts at 1 in two places, 0 in two others | **From 0** — the t=0 term is the terminating search that finds nothing, and it is the largest |
| Lemma 15 vs Lemma 16 | Chapter 5 writes a bare `log` in the Fourier weight and simulation time, and an explicit `log2` in the Chebyshev truncation degree. The two differ by `ln 2`, which survives a square root into every query count | **Base two where the lemma leaves it open; the lemma wins where it does not.** So base two for Lemma 15's weight and time and Lemma 17's `n_exp`, and `log2` for Lemma 16's `j0` as printed — but *natural* logs in Lemma 12, which writes `ln(1/δ)/e` and divides by `log(e + …)`. Fourier's count moves by ×1.74; ordering unchanged, and the Fourier weight now agrees exactly with the qls-comparison repository |
| §4.6.2's `d` | The list of iteration-related measures names the maximum non-zeros in the *columns* of `A_B`, but the normalisation four paragraphs later needs `‖A_B‖₂ ≤ d‖A_B‖max`, which the column maximum does not give when a row is denser | **`d` is `max(row, column)`.** Ruled by Andreea. Taking the column maximum makes `A_1` and `A_max` over-estimates rather than the lower bounds they are declared to be — on a thirty-relay max-flow network, 56 of 121 bases violate it and the total lands 16 % high. It is also the sparsity of the Hermitian dilation a quantum solver acts on, which is what `linsystem.py` had concluded independently |
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
  instances/      Readers for the files people have — DIMACS, Pisinger,
                  Matrix Market, MPS. Standard library only, and they import
                  nothing from the rest of the library
  classical/      What runs to produce a log: Dinic, a revised simplex, an
                  interior point method, a linear solve, and knapsack's
                  read-only route; plus budget.py (how a run ends) and
                  generate.py (file to log to cost). numpy lives here only
  static/         The one page the interface serves, with no external resources
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
  *is* truncated amplification, now that the disputed half applies to both, that
  a hand-assembled `SimplexIter` equals the registered one, and that Lemma 16
  and Binkowski's Lemma 1 agree and stop agreeing under the rejected reading.
- Commit messages explain why, in prose. No bullet lists of files changed.
- Andreea prefers open discussion to multiple-choice questions, and asked for
  coding decisions to be made without checking in. Surface anything that
  changes the *science*; decide the rest.

## Producing the logs, not just asking for them

The loop is closed: `hybrid-benchmarking run <instance-file>` reads the file, runs the classical
algorithm here under a budget, writes the log, and costs it. A log arriving from
somewhere else takes the path it always took — this generates one, it does not
replace the format.

The instrumented quantities are the expensive part, and each has a reading that
is wrong and still produces a gate count that sums and plots like a right one.
They are set out at length in `classical/simplex.py`; in short:

- **κ is GLPK's**, `max(1, κ₁/m)` with `κ₁ = ‖A_B‖₁‖A_B⁻¹‖₁` from (4.33), not
  the exact condition number. An exact κ₂ would be larger *and* no longer a
  bound.
- **`d` is `max(row, column)`**, though §4.6.2's list names the column maximum.
  Ruled by Andreea; see the ambiguity table above for why the column maximum
  breaks the normalisation stated in the same section.
- **A_1 and A_max are the normalised lower bounds** `‖A_B‖₁/(d‖A_B‖max)` and
  `1/d`, because SimplexIter runs on a normalised matrix. The raw norms are
  logged beside them so the convention is reversible.
- **t is the positive components of `u = A_B⁻¹A_k`** — Lemma 10 says so — not
  the improving-column count, which marks Lemma 24's search over the `n−m`
  columns and routinely exceeds `m`.

Everything a run decides that a reader could not reconstruct lands on the cost's
provenance rather than in a docstring, via `Run.assumptions`. Two of these are
structural and should stay:

- **Our solvers are not the published ones.** Costs from a generated log carry
  `Derivation.LOGGED` and the implementation's name, and the totals will not
  match the published figures. A route whose inputs are the instance rather than
  a measurement — the knapsack circuits — declares itself `ANALYTIC` instead,
  because a caveat that is not true costs as much as a missing one.
- **A truncated run is still data.** The budget is checked between iterations so
  a cut-off solve keeps its partial log; the total then understates the solve,
  which is a lower bound for a reason unrelated to the lemmas'. It is a status
  carried in the log file, not a warning in a terminal.

One thing left open, surfaced rather than decided:

- **The interior point route's numbers depend on which basis is chosen.** Both
  of Binkowski's systems are built on a set of `m` independent columns of `A`;
  the paper finds them by sparse QR, this finds them by orthogonalisation, and a
  different independent set gives a different condition number. Recorded on
  every cost rather than assumed away.
One earlier entry here is now closed, and the resolution is worth keeping.
**`flow_shape` predicted `n = 2E` and it is `2E + 1`.** Conservation at every
vertex, over the arc variables alone, forces the net flow out of the source to
zero — those rows sum to zero, because every arc leaves one vertex and enters
another — so such a program can express circulations and nothing else, and its
maximum flow is zero on every network. The missing column is the flow value.
A flow variable and an uncapacitated return arc both give `2E + 1`, so the two
defensible readings agree and this was a correction rather than a choice. It
moves the `n − m` non-basic count that the optimality test and the pivoting
rule search over, so anyone holding an old max-flow simplex log should recost.

Related, and it is why the interior point route works on networks at all: the
conservation rows are redundant by construction, so `classical/ipm.py` now
presolves dependent rows away rather than refusing, as any interior point code
does, and logs the reduced system dimension.

## The interior point route follows the paper, now that we have it

`arXiv:2604.24362` was read against the code, and three things were wrong.

- **The system was the wrong system.** `classical/ipm.py` built the plain normal
  equations `A D² Aᵀ` and walked a Mehrotra path. The paper's MNES is equation
  (6), which at its canonical iterate `(x, y, s) = (1, 0, 1)` reduces to
  `M̂ = I + F̄F̄ᵀ` with `F̄ = A_B⁻¹A_N`, and its condition number comes from
  `λ_i(M̂) = 1 + σ_i(F̄)²` — not from anything resembling `A D² Aᵀ`.
- **One system is costed, not a solve.** Section IV-B assumes convergence in a
  single iteration and benchmarks only the first Newton system. Summing a path
  gave several times the paper's number and was no longer a lower bound.
- **`newton_system_cycles` restated Lemma 1 and got it wrong**, with
  `log2(2/ε)` where equation (10) has `log2(γ/ε)`. It now calls
  `binkowski_chebyshev_queries`, so the two cannot drift; and the factor of
  eight moved from `Tomography` to the solver, where the paper puts it.

`IPM/oss` is now reachable too, as `quantum-interior-point-oss`: equation (8),
`O = [−XAᵀ , SV]`, n-dimensional. On the instances here it is the *larger*
system and the *easier* one — lower sparsity and condition number, so lower
difficulty `γ = sκ` — which is why the paper reports both rather than picking.

The interior point routes default to **ε = 1e-1**, not 1e-3: Section IV-A puts
`10⁻¹` into (10) and assumes one step of iterative refinement does the rest.

## Checked against the source repositories

They are fixtures, not inputs (rule 4). Where we differ, the difference is now
attributed rather than assumed.

**QUBRABENCH** — all four `Cade-*` entries reproduce its functions to twelve
digits, on seventeen reference points frozen in `tests/test_cade.py`.

**The quantum breadth-first search paper** (`QBFS___MaxFlow-2.pdf`; note that
`-4` contains an error) — clean, and it closes three questions an audit had left
open. The search space is the graph's whole vertex count, not `2^V`: Lemma 1
counts `2|L|` cycles per Grover iteration over a register of one qubit per
vertex, while the Methods section records "the total number of vertices in the
original graph, which corresponds to `|L|`" and Lemma 2 caps the schedule at
`√|L|`. The source's own layer *is* charged — QSearch runs "for each layer `l`
containing `t` vertices out of `|L|` total" — and an empty layer is not, not
being one of those. Lemma 2 also carries the disputed factor of one half
explicitly, agreeing with the ruling above.

**qls-comparison** (`evaluation/query_costs.py`, Method 2) — after the base-two
ruling the Fourier weight and simulation time agree exactly. Three differences
remain, and in each the repository parts from the thesis's own printed lemma
while this library follows it:

- Its `amplitude_amplification` omits the floor on `m_k` and divides by `4 m_l`
  where (5.40) divides by `4 (m_l + 1)`. Our `rounds(..., half=False)`
  reproduces (5.40) to twelve digits; the repository's value is up to **6×**
  larger at small success probabilities and smaller at large ones.
- Its `j0` uses natural logs where Lemma 16 writes `log2` explicitly — the one
  place it does not use base two. Ours is ×1.45 there, and stays there.
- Its `hamiltonian_simulation` uses base two where Lemma 12 writes `ln`, and
  tests the *unscaled* time against the crossover, where the lemma's condition
  is on `t' = d ‖A‖max t`. We follow the lemma on both counts.

This means the published Method 2 figures came from code that differs from the
lemmas as printed. Worth knowing before comparing any number here with one from
that paper.

**simplex-benchmarks** (Method 1, plus the instrumented GLPK fork it points at)
— the logged quantities agree almost everywhere. `κ = max(1, κ₁/m)` from
`bfd_condest` is byte-for-byte our definition, floor and division included;
`A_1` and `A_max` are the same two normalised expressions applied at the same
two sites of the Berry bound (in their cost code rather than their logger);
`u_norm` is the same Euclidean norm of the same column; `c_max` is the original
objective and not the phase-1 penalty; both log phase 1 alongside phase 2;
Lemmas 8 and 11 are identical constant for constant. It logs the *column*
maximum for `d`, which is the divergence the ruling above already covers.

Three things it settles, and three worth knowing about it:

- **`t` for Lemma 24.** It logs `candidate_columns` and `u_positive_elements`
  separately and feeds each to its own lemma. That closes the open question:
  `FindColumn/random` now takes `t_improving`, and `SimplexIter/random` needs
  both counts. The other two rules are unaffected.
- **Lemma 10's `− 1`.** Ours has it, theirs does not; the printed lemma does.
- **The pivot sum** starts at `t = 1` there and `t = 0` here, which is the
  standing ruling; worth about 6.9× on their own fixture.
- Its `qsearch.cpp` computes `1/(1<<(k-1))` in **integer** division, so
  `nQ(t, L)` returns exactly 0.5 for every `t > 0` regardless of `t` or `L`
  (intended: 23.75 at `t=1`, 2.01 at `t=42`, for `L=378`). `nQ(0, ·)` is
  unaffected, which is why FindRow and IsOptimal escape. It suppresses their
  steepest-edge total by about 12.9×.
- Its `qlsa.py` opens `compute_lower_bound` with `kappa = 1`, overwriting the
  logged condition number before any caller can use it. **The published Method 1
  gate counts therefore do not depend on the condition number the study went to
  the trouble of instrumenting.** On their own `dano3mip` fixture this is the
  whole of the 10⁵–10⁷ gap between the two pipelines: forcing `κ = 1` on our
  side too brings every subroutine to within a factor of 3 to 90.
- Its final "logged because of optimality" record is emitted on a path where
  `u` was never written, so that record's `u_norm` and `t` are recycled heap.

**qipm** (Binkowski's own code) — the MNES construction, the canonical iterate,
`κ` from `σ(F̄)`, the `n − m < m ⇒ λ_min = 1` shortcut, the cost formula's
structure, the placement of the factor eight, and `ε = 1e-1` all agree exactly.
The OSS matrices agree up to a sign on the null-space columns, which changes no
singular value. Three differences, two now fixed here:

- **The OSS readout dimension is `2n`, not `n`** — footnote 2's dilation applies
  to the dimension as well as to `s` and `κ`, which his commit `33bbc70` says in
  as many words. Fixed.
- **The basis is chosen by column-pivoted QR**, as his sparse QR is a
  sparsity-aware version of. Taking the columns in index order, which is what
  our Gram-Schmidt amounted to, lands at the ill-conditioned end of a range
  spanning four orders of magnitude in `κ(M̂)` — the wrong end, since the cost
  is quadratic in it. Fixed; on random programs it lowers `κ(M̂)` by 30–350×.
- His code uses natural logs where Lemma 1 and equation (10) print `log2`, as
  qls-comparison does in its `j0`. Ours follows the printed lemma, so ours is
  ×1.4427 per solve there. **Two independent codebases make the same
  substitution**, which is worth weighing before assuming it is a slip.

Still open on the qipm side: his pipeline runs HiGHS presolve before the
standard form and ours does not, so our logged dimension is systematically
larger on instances with removable structure; and his `σ_min` falls back to
random probing when ARPACK does not converge, which underestimated `κ` by 8× and
11.5× on two instances of ours — his `κ` is a lower bound by construction, ours
is exact.

## The knapsack variants

`arXiv:2503.22325` (Wilkening, Lefterovici, Binkowski, Funck, Perk, Karimov,
Fekete, Osborne) extends the tree generator to the quadratic and
multidimensional knapsack problems. They are **separate entries with separate
counts** — `QTG-quadratic` and `QTG-multidimensional` — not a parameterisation
of `QTG`. Different circuits, different instances; only the primitives are
shared, because those really are the same gates.

The paper prints one closed form, the multidimensional qubit count
`n + Σ|cᵢ| + |P| + max(n, Σ|cᵢ| + 1, |P|)`, which is reproduced as written. It
gives the gate and cycle counts structurally only — "rather straight-forward as
the QTG for the QKP effectively arises from the QTG for the KP plus additional
doubly-controlled profit additions" — and **its simulator is not published**:
Zenodo 16895828 and `SoerenWilkening/QTG_0-1Knapsack` are the *0-1* code, and
the `CBQS-*` repositories belong to a different paper. So three constants are
derived from Appendix C's own rules and named on every cost:

- a **doubly-controlled addition** is a singly-controlled one plus two Toffolis,
  computing and uncomputing the conjunction of its controls in the shared
  ancilla register — the cheapest reading of "multi-controlled gates share one
  ancilla register";
- the **dimensions of a multidimensional instance run in parallel**, their
  registers being disjoint, so the cycle count takes the deepest dimension where
  the gate count takes the sum;
- **feasibility across dimensions** is a balanced Toffoli tree over the `d`
  flags: `d − 1` gates, `2⌈log₂ d⌉` cycles.

One convention that is not pedantry: a pair's profit is what it earns **in
total**. The paper's matrix is symmetric and its objective sums ordered pairs,
so a pair earns `p_{mm'} + p_{m'm}` while the circuit adds once per unordered
pair. The count depends on the position of the lowest set bit, so a factor of
two moves it. A full symmetric matrix and its upper triangle are both accepted
and agree.

Not implemented: the further parallelisation the paper describes for the
quadratic profits, `O(log₂(n²))` layers of pairwise additions giving `O(log₂ n)`
depth. It is stated asymptotically, with no constants to transcribe, and it is
the cheaper of the two — so these entries are the dearer form.

## Where it stands

Complete: 46 routines / 53 implementations, all of Appendices A, B and C, the
maximum-flow study, the interior point pipeline, the Cade family, the
composition layer, the problem-first entry point, the log format, instance
readers and instrumented classical solvers for every problem, a local web
interface and a CLI. 866 tests.
