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
  problems.py     Nine families and the seventy-one names people use for them
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

**The quantum breadth-first search paper** -- published since as *Use case
study: benchmarking quantum breadth-first search for maximum flow problems*,
arXiv:2604.24962; the drafts read were `QBFS___MaxFlow-2.pdf`, and note that
`-4` contains an error — clean, and it closes three questions an audit had left
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

## What the studies found, on the front page

`figures.py` and `static/figures.json` hold one chart per publication, redrawn
in the page's own colour variables and served at `/api/figures`. Each declares
`kind`: **redrawn** means the values are as published or computed from real
data, with `method` saying which and how; **schematic** means the shape only,
and its caption must begin with the word. There is no third kind. A chart is a
number wearing a picture, so it falls under rule one.

All six came back `redrawn`, and two of them earned it the hard way: the
knapsack figure was digitised from the published figure's own source PDF in
`~/Documents/papers-quantum-tree-generator`, and the linear-solver figure was
computed from `qls-comparison`'s `queries.csv` rather than from the paper's
plots.

**The knapsack variants figure is digitised from the paper's own figure 2**,
which plots quantum tree generator runtime against Gurobi runtime for both
variants: 4416 vector markers read out of the PDF, mapped through the gridlines,
with marker area giving the instance size and marker colour the gap to the best
known bound. Two things came out of that reading and both are the author's to
confirm. **The colourbar is inverted** relative to the natural reading, yellow
being a zero gap; the corrected decoding is what reproduces the paper's own
Discussion. And **the Results paragraph and the Discussion disagree** on where
the quadratic advantage is largest, Results saying larger optimality gaps and
Discussion saying smaller; the digitised data agrees with the Discussion, median
ratio 129 near a zero gap against 53 above a hundred percent.

**A second computation turned up a discrepancy worth resolving.** On the 10,233
simplex-derived systems, Chebyshev is the cheapest solver in **100 %** of rows
and QSVT is third, where the paper's abstract reports the QSVT method as the
best performer. Independently re-checked here on the raw file, with the
authors' own columns and with the plain ones: same answer either way. QSVT does
win the random set, and the simplex systems have median condition number 1.0
and median sparsity 1, which is the regime the Chebyshev bound is cheapest in,
so the likely reading is that the abstract's claim is about the instances where
the functional assumptions hold exactly. Andreea's to rule on; the caption
currently states it per data set.

**Determinism was broken and is now fixed.** `synthesise._seed` built its seed
from `hash(key)` over strings, and Python salts string hashing per process: the
same answers repeated within one run of the server and produced a *different*
instance in every new process, so the command line and the server disagreed.
Found by an agent while sweeping network sizes, verified here, and fixed by
seeding from `zlib.crc32` of a canonical form of the answers. The promise is
that a number does not wobble between two askings, which has to hold across
processes or it is not a promise; `tests/test_beginner.py` now checks it by
running two interpreters.

## Names, and the problems under them

Each of the seventy-one now opens with **a paragraph and a drawing**, held in
`static/illustrations.json` and served through `problem_detail`. The paragraph
is the situation in the reader's own nouns with real quantities in it, since a
number is what makes somebody recognise their own case; the drawing is inline
SVG in the page's colour variables, so it follows the theme and fetches nothing.
Seventy-one separate drawings, not nine: a satellite over a horizon, a van and
its stops, a spacecraft bay with mass and power gauges, a rota grid.

Two rules are tested rather than trusted, in `tests/test_illustrations.py`. **No
mathematics reaches the reader**: no jargon in a story, and no story telling
somebody their problem is another one in disguise, which would undo the masking
the catalogue exists for. What a story may do is offer a second set of nouns for
the same picture, where a neighbouring field would recognise itself. **And the
drawings fetch nothing**: no scripts, addresses, external images or fonts, no
literal colours, no dashed strokes.

Writing them turned up a leak in the catalogue itself: `circuit-analysis` asked
"how many nodes in the circuit", and *node* is on the forbidden list. Its nouns
are junctions now, so the paragraph and the questions agree.



A problem in the interface is a **name and a story**; the *family* underneath
owns the routes, the fields and the shapes. Siting satellites and siting
charging points are one family with two names, and someone arriving with either
should not have to learn that to get a number. So there are nine families and
seventy-one names, and adding a seventy-second costs a line of prose rather than
a copy of a route.

The menu shows the names. It does not show which problem each one is underneath
— that classification is what the catalogue exists to spare people. It is still
in the data, and it is still on every cost's provenance, which is the one place
it must never be dropped.

Two families are new: `quadratic-knapsack` and `multidimensional-knapsack`.
Both read files now — Billionnet–Soutif for the first, OR-Library for the second
— and both sets ship as `.txt`, so `detect` tells the three all-but-numeric
knapsack layouts apart by the shape of their opening rather than by extension;
`--layout` names one outright where that is not enough.

Their pair bonuses are **bonuses only**. The circuit has a gate where a pair
earns something together and none where it costs something, so a negative one is
refused rather than counted — and that refusal matters more than it looks, since
the lowest set bit of a negative number is a perfectly good integer and the count
would otherwise come back.

**The positive-integer rule blocks most of both published sets**, and this is
worth a decision rather than a shrug. A zero profit or weight has no lowest set
bit, so it is refused at the file rather than surprising someone in a gate
count. But the quadratic set's density thins its *linear* coefficients too, so
only the full-density files parse; and of OR-Library's multidimensional sets,
`mknap1` and 44 of `mknap2`'s 48 problems carry zero costs, leaving only the 270
Chu–Beasley instances — none of which states an optimum. What a zero-valued item
should cost is the open question: nothing, since its layer has nothing to add,
or a full-width addition, which is what Sören's 0-1 code counts for a zero
summand. It changes counts either way.

## Installed by double-clicking, and kept current by itself

`install/macOS-install.command` and `install/Windows-install.bat` install the
package from a zip of `main` and write a desktop icon. The icon calls
`hybrid-benchmarking open`, which is where the logic lives, because it has to be
right on Windows too and a batch file cannot be tested here: already running,
show it; not running, start it **detached** and wait. Detached is not a detail.
An icon that stays running is one the Finder will not launch a second time, and
an app with no windows has no front to bring forward, so the second
double-click failed with `-600` until the icon became a doorbell rather than the
house.

`update.py` runs at every launch and installs a newer version if there is one.
Three rules hold it: **a checkout is never touched**, so somebody's source tree
is safe; **failure is silent**, since being out of date beats not opening; and
it is **bounded** to a few seconds. It is also the only thing this tool sends
anywhere, so the front page says so rather than claiming that nothing leaves at
all, which would now be false.

**The data-file route is hidden from the interface**, at Andreea's request, and
only from the interface: the readers, the route and the tests are untouched and
the command line still reads files. The page now says plainly that it makes up
an instance of the size you describe, that the charts come from far harder real
benchmark instances, and where to write for real work.

## Three levels, and what the first one costs honestly

The interface now meets people at three depths. **Describe it** asks for the
problem in its own nouns -- how many people, how many shifts -- builds an
instance of that shape, runs the real classical solvers on it, and costs every
route side by side. **I have a data file** is the path that was there before.
The **Browse** and **Build** tabs are still served and still work, but their
buttons are hidden and the tab bar with them: the page opens on the problems and
shows nothing else. Unhide the two buttons in `static/index.html` to restore.

The page opens on an **introduction** rather than on a form: what is counted
(gates, cycles, queries), that these are logical operations in the fault-tolerant
regime and never added to one another, and the two ways in -- describe your
problem in ordinary numbers, or bring your own instance, which is the one that
is about you rather than about your size. The header title returns to it. This
also fixed a real bug: the subroutine list and the problem menu are the same
`<ul>`, both were fetched at startup, and whichever finished last owned the left
menu -- so a menu of problems sometimes came up holding subroutines. Browse now
loads when Browse is entered, which cannot happen while its button is hidden.

The introduction also states the **gate model** where a reader will ask it --
a single-qubit gate, a two-qubit controlled rotation and a Toffoli each count as
one, state preparation counts as one, disjoint gates share a cycle, and this is
*not* a Clifford+T count: a fault-tolerant Toffoli costs several T gates and the
magic states behind them, so a physical estimate needs a code, a distance and a
T-count on top. It is the thesis's model, carried on `_GATE_MODEL` in
`routines/standard.py` and on every entry's assumptions.

It also **lists every work the library reimplements**, without author names --
Andreea asked for the papers alone. The credit did not move off the numbers:
`cost.provenance.sources` still names the authors, which is the copy that
travels, and `tests/test_credits.py` asserts both halves. Algorithm and format
names that happen to be surnames stay, since Dinic's algorithm and the Pisinger
layout are what those things are called. The works listed are: the seven
studies -- the thesis, its three case-study papers (the simplex runtime analysis
arXiv:2311.09995, the linear-solver comparison arXiv:2503.21420, and the 0-1
knapsack paper in npj Quantum Information 11, 146, which is where the quantum
tree generator itself comes from), the knapsack variants at QCE 2025, the
max-flow study and the interior point paper -- the quantum results they are
built on, the classical algorithms and
instance formats that actually run here, and the five repositories checked
against but never copied from. `tests/test_credits.py` holds it to the registry
-- every provenance source must name something the page names, so a routine
added with a new citation fails until the credit goes up. Every work with a
published address is **linked**, which refined the offline rule rather than
breaking it: the invariant is that nothing is *fetched*, so `<a href>` is
allowed and `src`, `@import` and `url(` are not, and the test now checks that
every external address in the file is inside an anchor. It folds diacritics,
because the registry writes Hoyer and Gilyen where the page writes Høyer and
Gilyén; the page is right and neither is a missing credit.

The seventy-one names are grouped under nine headings, and the headings
deliberately **cut across the families**. Grouping by family would put siting a
satellite next to siting a charging point and announce that they are one
problem, which is the classification the catalogue exists to spare people; so
the headings are kinds of task -- choosing what to fund, deciding where to put
things, avoiding clashes -- and four of the nine mix families while
`quadratic-knapsack` is spread over three. `web.problems()` sorts by heading so
the page groups runs of equal `category` without knowing what the headings are;
`tests/test_categories.py` holds the grouping to that, and to headings free of
the vocabulary the names are there to avoid.

The first level is the one with a hazard, and it is handled in two places rather
than one. `classical/synthesise.py` generates the instance, and every run built
that way carries `CAVEAT` in its assumptions: *the instance was generated to the
size given, not read from your data*. It travels on `Run.assumptions`, so it
reaches the cost's provenance by the ordinary route and cannot be dropped by a
change to the page; the interface then leads with it, above the numbers. There
is a test that a run from a real file does *not* carry it.

Generation is deterministic -- the seed comes from the answers -- because a
number that wobbled when you asked twice would be worse than useless. Somebody
would average them.

**The columns are not a race, and the data still says so — the page no longer
does.** `compare()` puts every route on *one* generated instance, because two
routes costed on two instances would differ for a reason nobody asked about; it
returns `comparable: False` and the units present. The paragraph that spelled
this out beneath the columns was removed at Andreea's request: the algorithms
are self-explanatory, and each column already carries its own unit, bound and
assumptions.

The tension `problems.py` has always named -- three numbers in a row imply a
comparison none of the sources supports -- is therefore resolved by the columns
themselves rather than by a caption: each carries its own unit beneath its
total, its own bound, derivation and assumptions, `compare()` refuses to
reconcile them, and the cost algebra still raises if anything tries to add gates
to cycles.

## Counts on a clock, and the number that does not age

`runtime.py` turns a count into seconds and, more usefully, seconds back into a
required rate. The construction is Chapter 4's: the simplex study does not
assume a gate time and report a duration, it **inverts** -- for each instance it
computes the time per gate the quantum side would need to match the classical
solver, and compares that with `6.5e-9 s`, the record for an *isolated* gate
operation (the ultrafast Rydberg experiment, reference [95] in the thesis). A
required rate is falsifiable and does not go stale; a projected duration ages
the moment hardware moves.

Both directions are offered and they are not equally safe. **The required rate
is a measured classical wall clock over a lower-bounded count**, so it is an
upper bound on the requirement -- generous to quantum, like everything else
here. **The projection is illustrative only**, and every reason it flatters the
quantum side is carried on it in `PROJECTION_ASSUMPTIONS`: no error correction
(a logical operation is `d` rounds of syndrome extraction, measurement-limited,
so orders of magnitude slower), a record for an isolated operation rather than a
sustained rate, and a gate model that charges a synthesised rotation the same as
a Toffoli. All three point the same way, which is what makes a negative result
drawn from it robust and a positive one worthless.

**Only gates and cycles can be timed.** Queries and subroutine calls raise
`NoClock`: nothing here fixes what answering a query costs, so a duration would
be invented rather than derived. The interface shows the reason in the same
table rather than leaving a gap.

The denominator comes from the loggers: all four instrumented solvers now stamp
every record with `at_seconds`, the elapsed classical time at which that
iteration was reached, so consecutive stamps differ by one iteration's cost and
`per_iteration()` recovers the per-iteration series Figure 4.5 is built from. A
log written by somebody else has no stamps and gets `()` rather than a guess.

**A routine that replaces part of a solve is compared as part of a solve.**
`Run.replaced_seconds` says how much of the classical run the quantum side
actually takes over; Dinic logs the layering sweeps separately from the
blocking-flow phases (`sweep_seconds` per record), because quantum BFS replaces
the sweeps and nothing else. So the quantum column is *the replaced part costed
plus the retained part measured*, and the required rate divides by the replaced
part alone. This matters more than it sounds: the sweeps are around a tenth of
Dinic's time, so charging the count against the whole solve made the route look
like it cleared the speed record when it does not.

The comparison view ends with a **log-scale chart** carrying the numbers on the
bars: the hybrid quantum total against the classical time measured on this
machine, one pair per route, each route timed against the classical algorithm
that produced its own log. On a sixty-node network the thesis's own shape
reappears -- the breadth-first route needs a few nanoseconds per gate, within a
factor of two of the record and on the wrong side of it, while the simplex route
needs about `2.7e-18 s`, some 2.4 billion times faster than anything measured.

## The QTG search, and where the square root lives

Checked against the original's driver, `_qtg_bindings.cpp::execute_q_max_search`.
It accumulates, once per threshold segment,

    (rounds + 2·iter)·C_QTG + iter·(C_mc(n−1) + C_comp(profit_qubits, P))

where `rounds` counts amplification attempts and `iter` sums the Grover powers
drawn. `QTGSearch` writes the same thing per drawn power — `(2j+1)·C_QTG +
j·(zero + marker)` — and the two are the same sum, since `Σⱼ(2j+1) = 2·iter +
rounds`. **The formulas agree exactly**; there is a test, because the two
groupings look nothing alike.

So the handoff is narrower than it looked: what is missing is not a formula but
the *schedule*, and the schedule is genuinely random. `q_search` runs BBHT with
`c = 6/5`, `m_l = ⌈(6/5)^l⌉`, and draws `j` uniformly from `{1..m_l}`; whether a
round succeeds is decided by a simulated measurement against the amplified
distribution. There is no closed form for `rounds` or `iter`.

**The square root is in the classical tree generator**, `execute_ctg`, and it is
exact: for the same drawn power `j`, the quantum applies the generator `2j+1`
times while the classical emulation draws `4j²` samples. So the count of QTG
applications is the square root of the count of classical samples, round for
round, on the same schedule and the same random stream. That is what closes the
handoff — instrument a classical tree-generator run, log per segment the
attempt count, the sum of drawn powers, and the incumbent profit, and the cycle
count above follows with no quantum simulation at all.

Two cautions if that gets built. The original's `qtg_estimate_cycles` divides by
an unexplained 10 and accumulates a prefix sum of prefix sums, so it is inflated
relative to the driver's own expression — the analysis notebooks quietly drop
the 10 and recompute. And its comparator is passed `P + 1` for cycles and `P`
for gates, which ours does not distinguish.

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

## Open decisions

Everything here is a question about the mathematics, not the code. Each is
recorded where it arises as well; this is the list, so that a new session picks
them all up.

**1. The break item, and roughly 1.6× on every knapsack gate count.** Sören's
`gate_count_qtg` skips the capacity comparison *and* the cost-register QFT pair
for every item before `b = min{h : Σ_{m≤h} w_m > c}`, because up to there every
partial assignment is feasible. Appendix C as transcribed here does neither. On
the Pisinger fixture: 295 gates theirs, 462 ours, and 144 of the 167-gate gap is
exactly that saving. The QKP paper *describes* the optimisation, which suggests
it is an implementation refinement rather than part of the printed bound — but
if (6.6) assumes it, every knapsack count here is high by about 1.6×.

**2. What a zero-valued knapsack item costs.** Three readings, and the readers
currently take the first: **refuse** (a zero has no lowest set bit, so it is a
file-format complaint with a line number); **charge nothing** (the layer has no
addition to perform); or **charge a full-width addition** — which is what the
reference implementation does, since its `lso(0)` returns 0 and
`gate_count_add(reg, 0) = 3·reg − 2`, nineteen gates on a seven-bit register for
adding nothing. Almost certainly unintended there, but it is what the published
figures were computed with. Refusing blocks most of both benchmark sets: only
full-density quadratic files parse, and of OR-Library's multidimensional sets
only the 270 Chu–Beasley instances, none of which states an optimum.

**3. The interior point condition number.** The route reads the *unmodified*
normal-equation system's κ; a diagonal equilibration is logged beside it as
`kappa_equilibrated`. Equilibration is not reliably smaller — it lowers κ by
about a third on a vertex cover and raises it on an independent set — so there
is no safe default, only a stated one. Binkowski's own code was checked and does
not settle it.

**4. Which basis the interior point route picks.** Both his systems are built on
`m` independent columns of `A`. His comes from sparse QR, ours from
column-pivoted QR, and across defensible selections κ(M̂) spans four orders of
magnitude. Ours is no longer at the bad end, but the choice is still a choice
and it is recorded on every cost.

**5. Two independent codebases use natural logs where the papers print `log2`** —
qls-comparison's `j0` and qipm's Chebyshev count. We follow the printed lemmas
per the ruling above, so we sit ×1.4427 from each. Two codes agreeing against
two papers is worth weighing before calling it a slip.

**6. Method 1's published gate counts do not use the condition number.**
`simplex-benchmarks`'s `qls/qlsa.py` opens `compute_lower_bound` with
`kappa = 1`, discarding the logged value before any caller sees it. On its own
`dano3mip` fixture that single line is the whole 10⁵–10⁷ gap between the two
pipelines. Not ours to fix; worth knowing before any number here is compared
with a published one. Its `qsearch.cpp` also computes `1/(1<<(k-1))` in integer
division, so `nQ(t, L)` returns 0.5 for every `t > 0`.

**7. `QTGSearch` is one probe away from closing.** Its formula already agrees
with the original's driver exactly. What is missing is the schedule, and the
schedule is what an instrumented *classical tree generator* produces: for a
drawn Grover power `j` the quantum applies the generator `2j+1` times while the
emulation draws `4j²` samples, so generator applications are the square root of
classical samples, round for round. Log per segment the attempt count, the sum
of drawn powers, and the incumbent profit, and the cycle count follows with
nothing quantum simulated.

**8. No HiGHS presolve.** Binkowski's pipeline presolves before the standard
form and ours does not, so our logged Newton-system dimension runs larger on any
instance with removable structure. Unquantified — `highspy` is not installed.

## The look of it, and the two branches

The palette is **deep navy against orange**, from a Dopely card Andreea chose,
over cool neutrals: navy `#032c42` carries the identity and the controls both,
orange `#c26608` is what was measured, gold `#a6720a` is the cautions, and the
dark theme lightens all three. It arrived after several rejected schemes, and
the rejections are the useful part: a Classiq pink and olive read as two
opinions, a warm Ember scheme went flat because cream paper and brown ink pull
everything into one tonal band, and a mid blue was too light to hold a page
down. **Every drawing and figure is painted in the theme variables**, never in
literal colours, which is what made each of those swaps a five-line edit rather
than seventy-one.

`docs/` and the `pages` branch hold a **browser build**: the same page with
Pyodide starting Python in the tab and answering `/api` from a dispatcher
instead of a server. It works and it is published, and Andreea did not want it,
so it is unlinked from everywhere. Delete the branch and turn off Pages if it
is ever in the way.

**Figures are rendered and looked at, not merely parsed.** Substitute the
palette into the SVG, `qlmanage -t` it, and read the PNG. Three faults surfaced
that way and would not have otherwise: swatches struck through centred labels,
a legend landed on top of a label it had replaced, and the classical curve of
the variants figure ran through the tail of the name of the other one. The
scratchpad script that does it is worth rebuilding rather than working blind,
and two things about it are worth getting right, because both cost a wrong
conclusion: give the standalone file an `xmlns`, or Quick Look renders the
markup as text rather than as a drawing, and paint the background `--surface`
rather than `--ground`, because the figures sit on a card — paint the page
ground and a perfectly correct masking rect reads as a white scar across the
chart.

**Every key is now named in its own colour, and that is a test rather than a
habit.** `test_a_key_is_named_in_its_own_colour` walks each figure and holds
every legend swatch and reference rule to the rule the last three commits kept
re-establishing by hand. It excludes the data marks themselves, since the bar
chart's group headings follow its last bar in document order and are headings
rather than series names. The two the rule had never reached were the two that
are not line charts: the linear-solver bars, where the four names were in plain
ink, and the simplex bands, where the record line was introduced in grey. It
also caught a fourth nobody had looked at, the variants figure's crossover
annotation.

The variants legend moved out of the left panel and under both of them, which
is where a legend describing two panels belongs: stacked in the corner it had
nowhere to go, the band between the two curves being eleven points tall.

## Installed by a pasted line, and kept current by itself

`install/install.sh` (macOS and Linux) and `install/install.ps1` (Windows,
untested) are what the README tells people to paste. They build a venv of their
own under Application Support, `.local/share` or AppData, install from a zip of
`main`, and **write the desktop icon locally**, which is the whole trick: an
icon made on the machine carries no quarantine flag, where a downloaded app is
refused by macOS 26 with a message about malware and a highlighted *Move to
Bin*. Downloadable installers were built, signed nothing, and were removed for
exactly that reason.

The icon calls `hybrid-benchmarking open`: check, start **detached**, show. It
must not stay running, or the Finder refuses the second double-click with
`-600`. `update.py` runs first and installs a newer version if there is one,
never over a checkout, silently on failure, and it stops a running server
through `POST /api/quit` so the new code takes over rather than waiting for a
reboot.

**It updates on the commit, not on the version number**, and that was a bug
that looked exactly like working code. It compared `version` in
`pyproject.toml` and installed only when that number rose, so six consecutive
pushes reached nobody: five of them changed what a reader sees, none bumped a
version, and the updater reported success by saying nothing at all. Remembering
to bump a number by hand is the step that was missed six times running, so the
question it asks is now which commit `main` is on, and pushing is shipping.
Nothing else was needed on the delivery side: pip reinstalls a package from a
URL even when the version is unchanged, which was checked rather than assumed.

The source moved for a second reason, independent of the first.
`raw.githubusercontent.com` serves `cache-control: max-age=300`, so the file it
hands back can be five minutes behind the push — which is how an app installed
minutes ago could still miss what was already on `main`. The commit feed at
`commits/main.atom` is served `max-age=0, must-revalidate` and is current the
moment a push lands. The version is still read, but only for the sentence a
person sees and as the fallback when the feed cannot be reached, so an outage
degrades to the old behaviour rather than to none.

What the running copy was built from is written to `installed-commit` in the
same data directory the installers put the venv in — **outside** the venv,
since installing over it is precisely what would erase it. A copy with no stamp
counts as out of date, which is what makes this self-healing: every install
made before the stamp existed updates once, writes one, and settles.

**And the page itself is served `Cache-Control: no-store`.** The package is
replaced underneath a server at an address that never changes, so a browser
holding the old page would show the old interface over the new code, with
nothing on screen to say so and no reason for somebody who opened a desktop
icon to think of reloading. Local traffic costs nothing, so there was never
anything to weigh against saying it plainly.

`tests/test_update.py` holds all of it, and it is new: this was the only
machinery in the library whose job is to run on a stranger's machine, and the
only machinery with no test at all.

## Where it stands

Complete: 46 routines / 53 implementations, all of Appendices A, B and C, the
maximum-flow study, the interior point pipeline, the Cade family, the
composition layer, the problem-first entry point with 71 names over 9 families, the log format, instance
readers and instrumented classical solvers for every problem, a local web
interface and a CLI. 2554 tests.
