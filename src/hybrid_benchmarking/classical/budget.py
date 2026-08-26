"""How long a classical run is allowed, and what it means when it is cut off.

The tool asks for a log of a classical solve.  Producing one means actually
solving, and some of the instances people have take longer than anyone wants to
wait in front of a browser.  So every run here carries a budget, checked at
iteration boundaries and nowhere else -- a half-written iteration is not a
record, and a log whose last row is half an iteration is worse than a log that
stops one row earlier.

Three ways a run ends, and they are not the same number:

``COMPLETE``   the classical algorithm finished.  The log is the whole solve.

``TRUNCATED``  the budget ran out first.  The records are real and each one is
               costed exactly as it would have been; what is missing is the rest
               of the solve.  So the total understates the true cost, which
               makes it a lower bound **for a second reason** -- one that has
               nothing to do with the lemmas themselves being lower bounds.
               That distinction is why this is a status and not a warning.

``FAILED``     nothing usable came out: the instance was degenerate, the solver
               broke down, or the budget expired before the first iteration
               finished.  There is no number, and the right output is the reason
               rather than a zero.

Truncation is offered a remedy in that order too: a longer budget is a flag, and
only when the flag is not enough is the answer to get in touch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

#: What a run gets if nobody says otherwise.  Long enough for the instances the
#: underlying studies benchmark, short enough that a browser window is not left
#: hanging on an instance that will never finish.
DEFAULT_SECONDS = 300.0


class Status(Enum):
    """How a classical run ended."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


class Budget:
    """A wall-clock allowance, asked at iteration boundaries.

    Deliberately not a signal or a thread: an interrupted numerical routine
    leaves a basis half-updated, and there is no honest record to write from
    that.  Asking :meth:`spent` between iterations costs one clock read and
    cannot corrupt anything.
    """

    def __init__(self, seconds: float = DEFAULT_SECONDS):
        if seconds <= 0:
            raise ValueError("a budget of {} seconds leaves no time to run "
                             "anything".format(seconds))
        self.seconds = float(seconds)
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def spent(self) -> bool:
        """Whether the allowance is gone.  Ask this, and only this, in a loop."""
        return self.elapsed >= self.seconds

    def restart(self) -> "Budget":
        self.started = time.perf_counter()
        return self


@dataclass
class Run:
    """The outcome of one instrumented classical run, before it is costed.

    Carries the records the run produced and everything the log has to say about
    itself.  :meth:`stated` is what ends up in the file: the reader a week later
    gets the same caveats as the person who pressed the button.
    """

    #: What the instrumented algorithm was.  Named, because our implementation
    #: is not the one the published figures came from and the numbers will not
    #: match.
    implementation: str
    status: Status = Status.COMPLETE
    records: tuple = ()
    instance: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    #: How much of ``elapsed`` the quantum routine would actually take over.
    #: Dinic's quantum version replaces the layering sweeps and leaves the
    #: blocking-flow phase classical, so comparing a count of the sweeps with
    #: the whole solve would compare two different algorithms.  ``None`` means
    #: the routine replaces the run entire, which is the common case.
    replaced_seconds: Optional[float] = None
    budget: float = DEFAULT_SECONDS
    #: What the classical algorithm found, where it found something -- the
    #: maximum flow value, the optimal objective.  Not a cost; a sanity check
    #: anyone can verify against a reference solver.
    result: Optional[Dict[str, Any]] = None
    #: Why a failed run failed, or what a truncated one was in the middle of.
    reason: str = ""
    #: Anything the log has to declare that is not a number -- a convention
    #: taken from the thesis, a quantity we chose to lower-bound rather than
    #: compute.  These land on the cost's provenance.
    assumptions: tuple = ()
    #: How the records were arrived at, named as
    #: :class:`~..provenance.Derivation` names it.  Almost everything here is
    #: ``LOGGED``: the numbers are measurements of a classical run and would
    #: differ under another implementation.  A route whose inputs are the
    #: instance itself -- the knapsack circuits read the binary representations
    #: of the profits, and nothing was run to find them -- says ``ANALYTIC``,
    #: because calling that logged would hedge a number that is not hedged.
    derivation: str = "LOGGED"
    #: What this run deliberately does not produce, and what would be needed to
    #: get it.  Shown to the user rather than discovered later as a missing
    #: parameter.
    handoff: str = ""

    @property
    def truncated(self) -> bool:
        return self.status is Status.TRUNCATED

    @property
    def usable(self) -> bool:
        """Whether there is anything here to cost.

        Records or an instance: a route whose cost depends on the instance
        rather than on how the solve went produces no records at all, and a run
        that produced none of either produced nothing.
        """
        return (self.status is not Status.FAILED
                and bool(self.records or self.instance))

    def stated(self) -> Dict[str, Any]:
        """The ``generated`` block that travels in the log file."""
        block: Dict[str, Any] = {
            "status": str(self.status),
            "implementation": self.implementation,
            "derivation": self.derivation,
            "records": len(self.records),
            "elapsed_seconds": round(self.elapsed, 3),
            "budget_seconds": self.budget,
        }
        if self.replaced_seconds is not None:
            block["replaced_seconds"] = round(self.replaced_seconds, 6)
        if self.handoff:
            block["handoff"] = self.handoff
        if self.result:
            block["result"] = self.result
        if self.reason:
            block["reason"] = self.reason
        if self.assumptions:
            block["assumptions"] = list(self.assumptions)
        return block

    def advice(self) -> str:
        """What to do about a run that did not finish.

        A longer budget first, because most truncations are a solve that wanted
        eleven minutes rather than five and the flag is right there.  Only when
        that is not the answer is the answer a conversation.
        """
        if self.status is Status.COMPLETE:
            return ""
        if self.status is Status.TRUNCATED:
            # Four times what it had, floored at a minute: the suggestion has to
            # be proportionate to the run that was cut off, or it reads as a
            # ritual rather than as advice.
            return (
                "cut off after {:.1f}s and {} records; rerun with "
                "--budget {:.0f} to let it finish, and if it still will not, "
                "the instance is larger than this tool is packaged for -- "
                "that is worth telling us about".format(
                    self.elapsed, len(self.records), max(self.budget * 4, 60)
                )
            )
        return "produced nothing usable: {}".format(self.reason or "no reason given")
