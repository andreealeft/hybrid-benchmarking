"""From counts to seconds, and back to the question that does not age.

A gate count is not a runtime.  Turning one into the other needs a rate, and any
rate anybody quotes today will be wrong in five years -- which is why the
simplex study does not assume one and report a duration.  It **inverts**: for
each instance it computes the time per gate the quantum side would need in order
to match the classical solver that actually ran, and compares that against the
fastest gate operation anyone has performed.  The result is falsifiable and it
does not go stale: hardware either crosses the line later or it does not.

Both directions live here, because both are worth showing and only one of them
is safe to quote on its own.

**The required time is the honest number.**  It is a measured classical wall
clock divided by a lower bound on the quantum count, so it is an *upper* bound
on the speed the quantum side would need -- generous to quantum, in the same
direction as everything else in this library.

**The projection is the illustrative one.**  Multiplying a count by a gate time
answers "how long would this take if a gate took that long", and every reason it
is optimistic is stated on the result rather than left for the reader:

* the counts carry no error correction, and a logical operation is not one
  physical operation.  It is ``d`` rounds of syndrome extraction, each limited
  by measurement rather than by the interaction, so a real logical gate is
  several orders of magnitude slower than the figure used here;
* the reference rate itself is a record for an *isolated* gate operation --
  two atoms exchanging energy, with no initialisation, no readout and no
  neighbours -- not a rate any machine sustains;
* the gate model charges a Toffoli and an arbitrary rotation one gate each,
  and a rotation has no exact fault-tolerant implementation at all.

Every one of those points the same way: the projected duration is a floor, not
a forecast.  A route that loses on this comparison loses by more in reality,
which is what makes a negative result drawn from it robust and a positive one
worthless.

Only counts with a duration can be projected.  Gates and cycles have one;
oracle queries and subroutine calls do not, because nothing here fixes what
answering a query costs -- so those refuse rather than being handed a rate that
would quietly invent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .provenance import Unit

#: The fastest gate operation reported: two single Rydberg atoms exchanging
#: energy on a nanosecond timescale.  The simplex study uses this as its
#: reference line, and it is a record for an isolated operation rather than a
#: rate any device sustains.
RECORD_SECONDS = 6.5e-9

#: Control electronics put a floor under gate speed that physics does not:
#: shaped pulses come from arbitrary waveform generators, and the fastest are
#: in the 1-10 GHz regime, so a gate cannot be driven faster than about a
#: tenth of a nanosecond however quickly the atoms would oblige.
CONTROL_FLOOR_SECONDS = 1e-10

#: Named rates, so a projection says which one it used.
RATES: Dict[str, Tuple[float, str]] = {
    "record": (RECORD_SECONDS,
               "the speed record for an isolated gate operation, 6.5 ns"),
    "control": (CONTROL_FLOOR_SECONDS,
                "the floor set by control electronics at 10 GHz, 0.1 ns, "
                "faster than any gate has run, and not a claim that one will"),
}

#: Units that name something with a duration.
TIMEABLE = (Unit.GATES, Unit.CYCLES)

#: Said on every projection, because the projection is the number somebody
#: would otherwise quote out of context.
PROJECTION_ASSUMPTIONS = (
    "one counted operation takes one gate time, which assumes no error "
    "correction: a logical operation is d rounds of syndrome extraction, each "
    "limited by measurement, so a fault-tolerant machine is orders of "
    "magnitude slower than this",
    "the rate is a record for an isolated gate operation, not a rate any "
    "device sustains",
    "every counted operation costs the same, though a synthesised rotation is "
    "hundreds of gates where a Toffoli is seven",
)


class NoClock(ValueError):
    """Raised when a unit has no duration to give it."""


@dataclass(frozen=True)
class Projection:
    """What a count would take, and what it would have to take to win."""

    unit: Unit
    count: float
    #: Seconds per counted operation, and what that rate is.
    rate: float
    rate_named: str
    #: The count at that rate.
    seconds: float
    #: What the classical solver actually took, where one ran.
    classical_seconds: Optional[float] = None
    #: Classical time divided by the count: the rate the quantum side would
    #: need to break even.  An upper bound on the requirement, since the count
    #: is a lower bound.
    required: Optional[float] = None
    assumptions: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def shortfall(self) -> Optional[float]:
        """How many times faster than the reference rate it would have to be.

        Above one, the quantum side needs gates faster than the record; below
        one, the record already suffices.
        """
        if not self.required:
            return None
        return self.rate / self.required

    def describe(self) -> str:
        if self.required is None:
            return "{} at {}".format(humanise(self.seconds), self.rate_named)
        if self.shortfall and self.shortfall > 1:
            return ("would need {} per operation to match the classical "
                    "solver, which is {:.3g}x faster than {}".format(
                        humanise(self.required), self.shortfall,
                        self.rate_named))
        return ("would need {} per operation to match the classical solver, "
                "which {} already provides".format(
                    humanise(self.required), self.rate_named))


def timeable(unit: Unit) -> bool:
    return unit in TIMEABLE


def project(count: float, unit: Unit, classical_seconds: Optional[float] = None,
            rate: str = "record") -> Projection:
    """Put a count on a clock, and ask what clock it would need.

    Refuses units without a duration rather than inventing one for them.
    """
    if not timeable(unit):
        raise NoClock(
            "{} has no duration: nothing here fixes what answering one costs, "
            "so a time would be invented rather than derived. Gates and cycles "
            "can be timed".format(unit.value)
        )
    if rate not in RATES:
        raise NoClock("unknown rate {!r}; known rates are {}".format(
            rate, ", ".join(sorted(RATES))))

    seconds_each, named = RATES[rate]
    required = None
    if classical_seconds is not None and count > 0:
        required = classical_seconds / count
    return Projection(
        unit=unit, count=float(count), rate=seconds_each, rate_named=named,
        seconds=float(count) * seconds_each,
        classical_seconds=classical_seconds, required=required,
        assumptions=PROJECTION_ASSUMPTIONS,
    )


def per_iteration(records: Tuple[Dict[str, object], ...]) -> Tuple[float, ...]:
    """What each logged iteration cost the classical solver, in seconds.

    The loggers stamp every record with the elapsed time at which it was
    reached, so consecutive stamps differ by one iteration's classical cost.
    The first is measured from the start of the solve, which includes whatever
    setup preceded it.  Records without a stamp -- a log somebody else wrote --
    give nothing rather than a guess.
    """
    stamps = [record.get("at_seconds") for record in records]
    if any(stamp is None for stamp in stamps):
        return ()
    times, previous = [], 0.0
    for stamp in stamps:
        times.append(max(0.0, float(stamp) - previous))
        previous = float(stamp)
    return tuple(times)


def humanise(seconds: float) -> str:
    """A duration somebody can picture, across the thirty orders of magnitude
    these numbers actually span."""
    if seconds <= 0:
        return "no time at all"
    for size, name in ((1e-18, "as"), (1e-15, "fs"), (1e-12, "ps"),
                       (1e-9, "ns"), (1e-6, "us"), (1e-3, "ms")):
        if seconds < size * 1000:
            return "{:.3g} {}".format(seconds / size, name)
    if seconds < 90:
        return "{:.3g} seconds".format(seconds)
    if seconds < 5400:
        return "{:.3g} minutes".format(seconds / 60)
    if seconds < 172800:
        return "{:.3g} hours".format(seconds / 3600)
    if seconds < 3.15e7 * 2:
        return "{:.3g} days".format(seconds / 86400)
    years = seconds / 3.15576e7
    if years < 1e3:
        return "{:.3g} years".format(years)
    if years < 1.38e10:
        return "{:.3g} thousand years".format(years / 1e3)
    return "{:.3g} times the age of the universe".format(years / 1.38e10)
