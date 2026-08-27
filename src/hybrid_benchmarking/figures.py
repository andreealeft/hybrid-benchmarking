"""What the studies found, redrawn for the front page.

The library reimplements six published analyses, and until now the front page
listed them without showing what any of them concluded.  A reader deciding
whether this tool is worth their afternoon deserves the results themselves: the
gate time the quantum simplex would need, the orders of magnitude between the
linear solvers, the readout that dominates the interior point method.

Each figure declares how it was made, in ``kind``:

``redrawn``
    The values are as published, or computed from real data on this machine,
    and ``method`` says which and how.

``schematic``
    The shape of the result, drawn honestly, with nothing implied about
    particular values.  The caption says so in its first word.

There is no third kind, and in particular there is no figure whose numbers were
invented to make a chart look complete.  That is the same rule the costs
themselves follow: a plausible number with lost provenance is the failure mode
this library exists to prevent, and a chart is a number wearing a picture.

Every figure carries a link to the paper and, where the code is public, to the
code, because a redrawing is not a source and should never be mistaken for one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_DATA = Path(__file__).parent / "static" / "figures.json"

#: The order they appear in, which is the order of the thesis: the three
#: methods first, then the work that extends them.
ORDER = ("simplex", "knapsack", "knapsack-variants", "max-flow",
         "interior-point", "linear-solvers")

_LOADED: List[Dict[str, Any]] = []


def all_figures() -> List[Dict[str, Any]]:
    """Every figure, in the order above, with anything unknown left out."""
    global _LOADED
    if not _LOADED and _DATA.exists():
        loaded = json.loads(_DATA.read_text())
        rank = {key: index for index, key in enumerate(ORDER)}
        _LOADED = sorted(loaded.values() if isinstance(loaded, dict) else loaded,
                         key=lambda entry: rank.get(entry.get("key", ""), 99))
    return _LOADED


def figure(key: str) -> Dict[str, Any]:
    for entry in all_figures():
        if entry.get("key") == key:
            return entry
    return {}
