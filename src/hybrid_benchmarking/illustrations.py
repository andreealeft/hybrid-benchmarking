"""What each problem looks like, in words and in a drawing.

A problem in the catalogue is a name and a story, and until now the name did
most of the work: somebody arriving at "Assigning people to shifts" got a
one-line blurb and then a form.  A paragraph in the reader's own nouns, with
numbers in it, and a small picture of the situation, are worth more than either
the label or the blurb, because recognising your own problem is the step people
actually fail at.

Two rules hold here, and both are tested:

**No mathematics reaches the reader.**  The stories say sites, people, budgets
and limits.  They never name the family underneath, and they never tell
somebody that their problem is somebody else's problem in disguise: that is the
classification the catalogue exists to spare them.  What a story may do is
offer a second set of nouns for the same picture, where a reader from a
neighbouring field would recognise themselves in it.

**The drawings fetch nothing.**  They are inline SVG in the page's own colour
variables, so they follow the light and dark themes, and they carry no scripts,
no external images and no fonts of their own.  The page works with the network
unplugged, and a picture that broke that would be worse than no picture.

The data sits in ``static/illustrations.json`` rather than in this module, so
that seventy-one drawings do not have to be read past to reach the code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_DATA = Path(__file__).parent / "static" / "illustrations.json"

_LOADED: Dict[str, Dict[str, str]] = {}


def _all() -> Dict[str, Dict[str, str]]:
    global _LOADED
    if not _LOADED and _DATA.exists():
        _LOADED = json.loads(_DATA.read_text())
    return _LOADED


def story(problem_key: str) -> str:
    """The paragraph shown above the questions, in the problem's own nouns."""
    return _all().get(problem_key, {}).get("story", "")


def picture(problem_key: str) -> str:
    """The inline SVG shown beside it, or nothing where none was drawn."""
    return _all().get(problem_key, {}).get("picture", "")


def described() -> int:
    """How many problems have both, for the tests and for a quick check."""
    return sum(1 for entry in _all().values()
               if entry.get("story") and entry.get("picture"))
