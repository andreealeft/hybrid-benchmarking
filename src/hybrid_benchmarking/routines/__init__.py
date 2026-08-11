"""Registry entries.

Importing this package registers every routine it defines.  Order matters only
in that a composition must be imported after what it is built from.
"""

from . import oracles  # noqa: F401
from . import amplification  # noqa: F401
from . import hamsim  # noqa: F401
from . import linsolve  # noqa: F401

__all__ = ["oracles", "amplification", "hamsim", "linsolve"]
