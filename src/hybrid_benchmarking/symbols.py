"""Canonical symbols.

Every routine draws its parameters from here, so that a condition number in a
linear solver is literally the same object as the condition number in the
simplex subroutine that calls it.  Substituting ``kappa=12`` once reaches all
of them.
"""

from __future__ import annotations

import sympy as sp

# --- problem geometry -------------------------------------------------------

N = sp.Symbol("N", positive=True, integer=True)
"""Dimension of the linear system / size of the state space."""

d = sp.Symbol("d", positive=True, integer=True)
"""Sparsity: most non-zero entries in any row or column, whichever is larger.

Both, not either: it is the sparsity of the Hermitian matrix a quantum linear
solver acts on, and for a matrix that is not symmetric that is the dilation,
whose rows are the original's rows and its columns.
"""

kappa = sp.Symbol("kappa", positive=True)
"""Condition number of the matrix."""

norm_A_max = sp.Symbol("A_max", positive=True)
"""Largest absolute entry of the matrix."""

norm_A_1 = sp.Symbol("A_1", positive=True)
"""Induced 1-norm: the largest absolute column sum.

Of the *normalised* matrix wherever the quantum simplex is concerned; the
instrumented run divides by ``d ||A||_max`` before logging it.
"""

norm_x = sp.Symbol("x_norm", positive=True)
"""Norm of the solution vector, which sets the true success probability."""

# --- search -----------------------------------------------------------------

X = sp.Symbol("X", positive=True, integer=True)
"""Length of the list being searched."""

t = sp.Symbol("t", positive=True, integer=True)
"""Number of marked elements in the list."""

# --- amplification ----------------------------------------------------------

p = sp.Symbol("p", positive=True)
"""True success probability of the algorithm being amplified."""

p0 = sp.Symbol("p_0", positive=True)
"""Known lower bound on the success probability -- all the algorithm gets."""

# --- precision --------------------------------------------------------------

epsilon = sp.Symbol("epsilon", positive=True)
"""Precision of the returned state or value."""

delta = sp.Symbol("delta", positive=True)
"""Tolerance of the surrounding classical algorithm."""

# --- simulation -------------------------------------------------------------

t_sim = sp.Symbol("t_sim", positive=True)
"""Evolution time for Hamiltonian simulation."""

# --- linear programs --------------------------------------------------------

n_vars = sp.Symbol("n", positive=True, integer=True)
"""Number of variables in the linear program."""

m_cons = sp.Symbol("m", positive=True, integer=True)
"""Number of constraints, and so the size of the basis."""

c_max = sp.Symbol("c_max", positive=True)
"""Largest absolute entry of the cost vector."""

u_norm = sp.Symbol("u_norm", positive=True)
"""Norm of the pivot direction, ||A_B^{-1} A_k||."""


ALL = {
    "N": N,
    "d": d,
    "kappa": kappa,
    "A_max": norm_A_max,
    "x_norm": norm_x,
    "X": X,
    "t": t,
    "p": p,
    "p_0": p0,
    "epsilon": epsilon,
    "delta": delta,
    "t_sim": t_sim,
}

# --- knapsack circuits ------------------------------------------------------

register_bits = sp.Symbol("bits", positive=True, integer=True)
"""Size of a register, in qubits."""

capacity = sp.Symbol("capacity", positive=True, integer=True)
"""Knapsack capacity.  Fixes the capacity register size at floor(log2 c) + 1."""

profit_bound = sp.Symbol("profit_bound", positive=True, integer=True)
"""Upper bound on the optimal profit.  Fixes the profit register size."""

# --- generic wrappers -------------------------------------------------------

inner_gates = sp.Symbol("inner_gates", positive=True)
"""Gate cost of the unitary a wrapper is applied to."""

oracle_gates = sp.Symbol("oracle_gates", positive=True)
"""Gate cost of one call to the marker or comparison oracle."""

prepare_gates = sp.Symbol("prepare_gates", positive=True)
"""Gate cost of preparing the state a routine amplifies or estimates."""

n_qubits = sp.Symbol("n_qubits", positive=True, integer=True)
"""Width of the register a routine acts on."""

ancillas = sp.Symbol("ancillas", positive=True, integer=True)
"""Ancilla qubits an oblivious amplification reflects about."""

terms = sp.Symbol("terms", positive=True, integer=True)
"""Number of unitaries in a linear combination."""

clock_qubits_sym = sp.Symbol("n_c", positive=True, integer=True)
"""Clock register width in phase estimation, fixed by epsilon and delta."""

inner_gates_2 = sp.Symbol("inner_gates_2", positive=True)
"""Gate cost of a second unitary, where a routine interferes two of them."""


# --- instrumented counts in the manner of Cade et al. -----------------------
# These reuse X and t deliberately: the search space and its marked elements
# are the same quantities there as here, whatever the counting model.

classical_budget = sp.Symbol("K", positive=True, integer=True)
"""Classical queries allowed before the quantum part begins, in Cade's search."""

failure = sp.Symbol("eta", positive=True)
"""Upper bound on the probability that the quantum routine fails.

Not a precision.  Cade's search and maximum finding are exact once they
succeed, and this bounds how often they do not -- which is why it enters
through a logarithm rather than through the amplitude.
"""

amplitude = sp.Symbol("a", positive=True)
"""The squared amplitude an estimation routine is asked to resolve."""

subnormalisation = sp.Symbol("alpha", positive=True)
"""Block-encoding subnormalisation: the factor A is divided by to be encoded."""


t_improving = sp.Symbol("t_improving", positive=True, integer=True)
"""Columns with negative reduced cost: the marked count of Lemma 24's search.

Not :data:`t`.  That one is the number of positive components of the pivot
direction, which Lemma 10 searches the *basis* for; this one marks a search over
the ``n - m`` non-basic columns and routinely exceeds ``m``.  The instrumented
GLPK the thesis ran logs both, under ``candidate_columns`` and
``u_positive_elements``, and feeds each to its own lemma.
"""
