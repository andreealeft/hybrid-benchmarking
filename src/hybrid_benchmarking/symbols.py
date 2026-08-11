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
"""Sparsity: maximum number of non-zero entries in any row or column."""

kappa = sp.Symbol("kappa", positive=True)
"""Condition number of the matrix."""

norm_A_max = sp.Symbol("A_max", positive=True)
"""Largest absolute entry of the matrix."""

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
