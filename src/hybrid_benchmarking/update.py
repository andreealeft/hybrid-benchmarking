"""Keeping the installed copy current, and being honest that it does.

Somebody who installed this by double-clicking an icon has no git checkout, no
terminal and no way of knowing that the numbers moved.  So the desktop icon
checks for a newer version when it starts, and installs one if there is one.

That check is the **only** thing this tool sends anywhere, and the front page
says so.  It is one request to a public file listing the current version; it
carries no instance, no log, no answer and no identifier beyond the address the
request comes from.  Everything the tool computes stays on the machine, as it
always did.

Three rules hold it in place:

**A checkout is never touched.**  Somebody working on the library has it
installed from a source tree, and overwriting that with a release would destroy
work.  If the package is not living in a site-packages directory, the updater
declines.

**Failure is silent and harmless.**  No network, a slow connection, GitHub
down: the tool starts anyway on what is already installed.  An update check
that can stop the tool from opening would be a worse bug than being out of
date.

**It is bounded.**  A few seconds, once, at launch.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

#: Where the current version is written down.  The raw file rather than the
#: API: no token, no rate limit worth worrying about, and a text file is
#: harder to misread than a JSON schema that may change.
SOURCE = ("https://raw.githubusercontent.com/andreealeft/"
          "hybrid-benchmarking/main/pyproject.toml")

#: What gets installed when there is something newer.
PACKAGE = ("https://github.com/andreealeft/hybrid-benchmarking/"
           "archive/refs/heads/main.zip")

TIMEOUT = 4.0


def installed() -> str:
    """The version running now."""
    try:
        from importlib.metadata import version

        return version("hybrid-benchmarking")
    except Exception:
        return "0"


def _numbers(version: str) -> Tuple[int, ...]:
    """A version as numbers, so 0.10.0 sorts above 0.9.0 rather than below."""
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def newer(there: str, here: str) -> bool:
    return _numbers(there) > _numbers(here)


def from_a_checkout() -> bool:
    """Whether this is somebody's working copy rather than an installed one."""
    here = Path(__file__).resolve()
    return not any(part in ("site-packages", "dist-packages")
                   for part in here.parts)


def latest() -> Optional[str]:
    """The version on offer, or nothing if the question could not be asked."""
    from urllib.request import urlopen

    try:
        with urlopen(SOURCE, timeout=TIMEOUT) as answer:
            text = answer.read(4096).decode("utf-8", "replace")
    except Exception:
        return None
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return found.group(1) if found else None


def check_and_update(announce=None) -> Optional[str]:
    """Install a newer version if there is one, and say which.

    Returns the version installed, or ``None`` when nothing was done, which is
    the ordinary case: already current, working from a checkout, or offline.
    """
    if from_a_checkout():
        return None

    here = installed()
    there = latest()
    if not there or not newer(there, here):
        return None

    if announce:
        announce("Updating to {}.".format(there))
    done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "--upgrade",
         "--quiet", PACKAGE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return there if done.returncode == 0 else None
