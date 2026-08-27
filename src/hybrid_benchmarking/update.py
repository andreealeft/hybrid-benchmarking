"""Keeping the installed copy current, and being honest that it does.

Somebody who installed this by double-clicking an icon has no git checkout, no
terminal and no way of knowing that the numbers moved.  So the desktop icon
checks for a newer version when it starts, and installs one if there is one.

That check is the **only** thing this tool sends anywhere, and the front page
says so.  It is one request to a public file naming the current state of the
repository; it carries no instance, no log, no answer and no identifier beyond
the address the request comes from.  Everything the tool computes stays on the
machine, as it always did.

**What is compared is the commit, not the version number.**  This was the whole
of a bug that made the updater look like it worked and deliver nothing: it
asked for ``version`` in ``pyproject.toml`` and installed only when that number
rose, so six consecutive pushes -- five of which changed what a reader sees --
reached nobody, because none of them bumped a version.  Every push is a change
somebody may need, and remembering to bump a number by hand is exactly the step
that was forgotten six times running.  Asking which commit ``main`` is on makes
pushing and shipping the same event, and pip reinstalls a package from a URL
even when its version is unchanged, so the delivery half needs nothing else.

The source moved for a second reason, and it is a bug of its own.
``raw.githubusercontent.com`` serves ``cache-control: max-age=300``, so the
file it hands back can be five minutes behind the push -- which is why an app
installed minutes ago could still miss what was already on ``main``.  The
commit feed is served ``max-age=0, must-revalidate`` and is current the moment
a push lands.

Four rules hold it in place:

**A checkout is never touched.**  Somebody working on the library has it
installed from a source tree, and overwriting that with a release would destroy
work.  If the package is not living in a site-packages directory, the updater
declines.

**Failure is silent and harmless.**  No network, a slow connection, GitHub
down: the tool starts anyway on what is already installed.  An update check
that can stop the tool from opening would be a worse bug than being out of
date.  If the commit cannot be had, the old version-number comparison is still
there as a fallback, so an outage degrades to the previous behaviour rather
than to none.

**It is bounded.**  One request and a few seconds in the ordinary case, which
is the case where nothing needs doing.  The second request is made only when
something is going to be installed anyway.

**What was installed is written down outside the venv.**  The stamp records the
commit the running copy was built from.  It cannot live inside the package,
because installing over it is precisely what would erase it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

#: Which commit ``main`` is on.  An Atom feed rather than the API: public, no
#: token, no rate limit worth worrying about, and -- unlike the raw file this
#: used to read -- served ``must-revalidate``, so a push is visible at once.
HEAD = ("https://github.com/andreealeft/hybrid-benchmarking/"
        "commits/main.atom")

#: Where the version is written down, still read for the message a person sees
#: and as the fallback when the feed cannot be reached.
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


def in_an_environment() -> bool:
    """Whether this is running inside a virtual environment of its own."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def data_dir() -> Path:
    """Where this keeps what must outlive an install.

    The same three places the installers put the environment itself, so the
    stamp lands beside it rather than in a fourth location nobody expects.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/hybrid-benchmarking"
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData/Roaming")
        return Path(base) / "hybrid-benchmarking"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(base) / "hybrid-benchmarking"


def _stamp_file() -> Path:
    return data_dir() / "installed-commit"


def stamp() -> Optional[str]:
    """The commit the installed copy was built from, if it is known."""
    try:
        written = _stamp_file().read_text().strip()
    except Exception:
        return None
    return written or None


def write_stamp(commit: str) -> None:
    """Record what was just installed.  Failure here is not worth an exception:
    the cost of not writing it is one redundant update next launch."""
    try:
        path = _stamp_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commit + "\n")
    except Exception:
        pass


def _fetch(url: str, limit: int = 4096) -> Optional[str]:
    from urllib.request import urlopen

    try:
        with urlopen(url, timeout=TIMEOUT) as answer:
            return answer.read(limit).decode("utf-8", "replace")
    except Exception:
        return None


def latest_commit() -> Optional[str]:
    """The commit ``main`` is on, or nothing if the question could not be asked.

    The feed's first entry is the head commit, and its identifier carries the
    full hash.  Read from the front of the document, so the first match is the
    newest.
    """
    text = _fetch(HEAD)
    if not text:
        return None
    found = re.search(r"Commit/([0-9a-f]{7,40})", text)
    return found.group(1) if found else None


def latest() -> Optional[str]:
    """The version on offer, or nothing if the question could not be asked."""
    text = _fetch(SOURCE)
    if not text:
        return None
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return found.group(1) if found else None


def what_to_install() -> Optional[str]:
    """The commit to move to, or ``None`` when there is nothing to do.

    A copy with no stamp is treated as out of date, which is what makes this
    self-healing: every install made before the stamp existed updates once,
    writes one, and settles.
    """
    there = latest_commit()
    if there:
        return None if there == stamp() else there

    # The feed could not be reached.  Fall back to the comparison this made
    # before, which still catches a released bump, and say so by returning the
    # version rather than a commit.
    version = latest()
    if version and newer(version, installed()):
        return version
    return None


def check_and_update(announce=None) -> Optional[str]:
    """Install a newer version if there is one, and say which.

    Returns something truthy when an install happened -- the caller uses that
    to stop a server still running the old code -- and ``None`` when nothing
    was done, which is the ordinary case: already current, working from a
    checkout, or offline.
    """
    if from_a_checkout():
        return None

    target = what_to_install()
    if not target:
        return None

    # Only now is it worth a second request: this is the rare launch, and the
    # number is for the person watching rather than for the decision.
    name = latest() or target[:7]
    if announce:
        announce("Updating to {}.".format(name))

    # Inside its own environment, which is where the desktop installer puts it,
    # pip installs there and --user is not merely unnecessary but refused.
    where = [] if in_an_environment() else ["--user"]
    done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"]
        + where + [PACKAGE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if done.returncode != 0:
        return None

    if re.fullmatch(r"[0-9a-f]{7,40}", target):
        write_stamp(target)
    return name
