"""Reading 0-1 knapsack instance files.

Three writings of the same thing are in circulation, and the difference between
them is one number's position:

**Pisinger's generator**, the layout of the ``knapPI_*`` files distributed with
his hard-instance benchmarks.  A stated name, a small header, then one
comma-separated ``index,profit,weight`` row per item, then a row of dashes::

    knapPI_1_6_100_1
    n 6
    c 60
    z 91
    time 0.00
    1,44,23
    ...
    -----

``z`` is the optimal value the generator recorded; it becomes
:attr:`~.Knapsack.optimum`.  One file may hold many instances back to back.

**The plain Martello-Toth layout**, which is the item count, the capacity and a
``profit weight`` line per item -- written with the capacity on the second line,
or with the capacity last, or with the count and the capacity together on the
first line.  All three are the same instance differently punctuated, so all
three report ``martello-toth``; what they are not is interchangeable while being
read.  A file whose capacity could sit at either end is refused rather than
resolved, because a knapsack read one column out still produces plausible
numbers, and a plausible number nobody can trace is the thing this library
exists to prevent.

Profits and weights are positive integers here and nowhere later.  The circuits
that cost this problem read the binary representation of each value -- the
position of its lowest set bit decides a comparison's depth -- so a zero profit
or a fractional weight is not an approximation of anything, it is meaningless.
Refusing it in this module keeps it a file-format complaint, naming the line,
instead of a puzzling exception several layers down inside a gate count.

Four entry points.  :func:`parse` and :func:`read` return one
:class:`~.Knapsack` and **raise** on a file holding several, naming
:func:`parse_all` and :func:`read_all`, which return every instance in file
order.  Returning the first of a hundred would be the quiet discard this
package is written to avoid.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from . import InstanceError, Knapsack

__all__ = ["parse", "parse_all", "read", "read_all"]

#: The header keys Pisinger's generator writes.  Anything else is refused: an
#: unrecognised key means the file is not the format we think it is.
_HEADER_KEYS = ("n", "c", "z", "time")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def parse(text: str, name: str = "", source: str = "") -> Knapsack:
    """The single instance ``text`` holds.

    Raises :class:`InstanceError` if it holds more than one -- Pisinger ships
    files of a hundred -- rather than returning the first and dropping the
    rest.  :func:`parse_all` is the way to read those.
    """
    found = parse_all(text, name=name, source=source)
    if len(found) > 1:
        raise InstanceError(
            "{} holds {} instances back to back, and this returns one.  Use "
            "parse_all() or read_all(), which return them in file order; "
            "returning the first would discard {} instances silently.".format(
                source or "the text", len(found), len(found) - 1
            )
        )
    return found[0]


def parse_all(text: str, name: str = "", source: str = "") -> Tuple[Knapsack, ...]:
    """Every instance ``text`` holds, in file order.

    Only the Pisinger layout can hold more than one; the plain layouts always
    give a tuple of length one, so a caller that does not know which it has can
    use this and be right either way.
    """
    lines = _clean(text)
    if not lines:
        raise InstanceError(
            "{} holds no knapsack instance: the file is empty".format(
                source or "the text"
            )
        )
    if _looks_pisinger(lines):
        return _parse_pisinger(lines, name, source)
    return (_parse_plain(lines, name, source),)


def read(path: Union[str, Path]) -> Knapsack:
    """The single instance in a file.

    ``source`` becomes the path.  ``name`` becomes the name the file states,
    where the layout states one, and the file stem otherwise: within a Pisinger
    file the stated name identifies the instance, where the stem identifies
    only the file it came in.
    """
    location = Path(path).expanduser()
    return parse(_text_of(location), name=location.stem, source=str(location))


def read_all(path: Union[str, Path]) -> Tuple[Knapsack, ...]:
    """Every instance in a file, in file order."""
    location = Path(path).expanduser()
    return parse_all(_text_of(location), name=location.stem, source=str(location))


def _text_of(location: Path) -> str:
    try:
        return location.read_text(errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))


# ---------------------------------------------------------------------------
# numbers, and what they are allowed to be
# ---------------------------------------------------------------------------

def _clean(text: str) -> List[Tuple[int, str]]:
    """Non-blank lines, each carrying the line number it came from.

    Every complaint below names a line, and it has to be the line the user sees
    in their editor, not an index into whatever survived stripping.
    """
    kept = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if stripped:
            kept.append((number, stripped))
    return kept


def _count(token: str, number: int, what: str) -> int:
    """An integer that counts something and must be positive."""
    try:
        value = int(token)
    except ValueError:
        raise InstanceError(
            "line {}: the {} is {!r}, which is not an integer".format(
                number, what, token
            )
        )
    if value <= 0:
        raise InstanceError(
            "line {}: the {} is {}, and must be a positive integer".format(
                number, what, value
            )
        )
    return value


def _integer(token: str, number: int, what: str) -> int:
    """An integer, of any sign."""
    try:
        return int(token)
    except ValueError:
        raise InstanceError(
            "line {}: the {} is {!r}, which is not an integer".format(
                number, what, token
            )
        )


def _value(token: str, number: int, what: str) -> int:
    """A profit or a weight: a positive integer, and nothing else.

    The refusal is deliberate and its reason belongs in the message.  Downstream
    the value is read as a binary representation -- where its ones sit, and in
    particular the position of the lowest set bit, is what the cost of a
    comparison depends on -- so zero, negative and fractional values have no
    meaning there.  Better a format error naming this line than an arithmetic
    surprise inside a gate count.
    """
    try:
        value = int(token)
    except ValueError:
        raise InstanceError(
            "line {}: the {} is {!r}, which is not an integer.  Profits and "
            "weights are read as binary representations: the circuit cost "
            "depends on where the ones sit, so a fractional value has no "
            "meaning downstream".format(number, what, token)
        )
    if value <= 0:
        raise InstanceError(
            "line {}: the {} is {}, and must be a positive integer.  Profits "
            "and weights are read as binary representations: the circuit "
            "cost depends on the position of the lowest set bit, so a zero "
            "or negative value has no meaning downstream".format(
                number, what, value
            )
        )
    return value


def _assemble(
    name: str,
    source: str,
    layout: str,
    profits: Sequence[int],
    weights: Sequence[int],
    capacity: int,
    optimum: Optional[int],
    optimum_line: int,
) -> Knapsack:
    """The last check both layouts share, and then the instance.

    A stated optimum larger than every profit put together is not an optimum of
    anything, and it is the cheap way to catch a file read one column out.
    """
    if optimum is not None and optimum > sum(profits):
        raise InstanceError(
            "line {}: the stated optimum {} exceeds {}, the profit of taking "
            "every item, so no selection achieves it: the columns are "
            "probably not profit and weight in that order".format(
                optimum_line, optimum, sum(profits)
            )
        )
    return Knapsack(
        name=name,
        source=source,
        layout=layout,
        profits=tuple(profits),
        weights=tuple(weights),
        capacity=capacity,
        optimum=optimum,
    )


# ---------------------------------------------------------------------------
# Pisinger's generator
# ---------------------------------------------------------------------------

def _key_line(line: str) -> Optional[Tuple[str, str]]:
    """``('n', '100')`` for a header line, ``None`` for anything else."""
    words = line.split()
    if len(words) == 2 and words[0].isalpha():
        return words[0].lower(), words[1]
    return None


def _is_rule(line: str) -> bool:
    """The row of dashes that ends an instance."""
    return set(line) == {"-"}


def _looks_pisinger(lines: Sequence[Tuple[int, str]]) -> bool:
    """Whether the header says ``n <count>`` in words.

    The plain layouts write the count as a bare number, so a line whose first
    word is the letter ``n`` decides this without ambiguity.  It is either the
    first line or the second, the second when the file states a name.
    """
    for _, line in lines[:2]:
        key = _key_line(line)
        if key is not None and key[0] == "n":
            return True
    return False


def _parse_pisinger(
    lines: Sequence[Tuple[int, str]], fallback: str, source: str
) -> Tuple[Knapsack, ...]:
    found: List[Knapsack] = []
    index = 0
    while index < len(lines):
        instance, moved = _one_pisinger(lines, index, fallback, source)
        if moved <= index:  # cannot happen; a wrong reader must not spin
            raise InstanceError(
                "line {}: cannot make progress reading this instance".format(
                    lines[index][0]
                )
            )
        found.append(instance)
        index = moved
    return tuple(found)


def _one_pisinger(
    lines: Sequence[Tuple[int, str]], start: int, fallback: str, source: str
) -> Tuple[Knapsack, int]:
    """One instance, and the index the next one starts at."""
    index = start
    first_line = lines[start][0]

    stated = ""
    if _key_line(lines[index][1]) is None:
        number, line = lines[index]
        if "," in line:
            raise InstanceError(
                "line {}: expected an instance name or a header line such as "
                "'n 100', got the item row {!r}".format(number, line)
            )
        stated = line
        index += 1

    count = capacity = optimum = None
    count_line = optimum_line = first_line
    while index < len(lines):
        number, line = lines[index]
        key = _key_line(line)
        if key is None:
            break
        word, token = key
        if word not in _HEADER_KEYS:
            raise InstanceError(
                "line {}: unrecognised header key {!r}; this layout states "
                "{}".format(number, word, ", ".join(_HEADER_KEYS))
            )
        if word == "n":
            count, count_line = _count(token, number, "item count"), number
        elif word == "c":
            capacity = _count(token, number, "capacity")
        elif word == "z":
            optimum, optimum_line = _integer(token, number, "optimum"), number
            if optimum < 0:
                raise InstanceError(
                    "line {}: the stated optimum is {}, and no selection of "
                    "positive profits is negative".format(number, optimum)
                )
        index += 1  # 'time' is the generator's own runtime; nothing needs it

    if count is None:
        raise InstanceError(
            "line {}: this instance states no item count ('n <count>')".format(
                first_line
            )
        )
    if capacity is None:
        raise InstanceError(
            "line {}: this instance states no capacity ('c <capacity>')".format(
                first_line
            )
        )

    profits: List[int] = []
    weights: List[int] = []
    while index < len(lines) and len(profits) < count:
        number, line = lines[index]
        if _is_rule(line):
            break
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise InstanceError(
                "line {}: expected an item row 'index,profit,weight', got {} "
                "comma-separated field{} in {!r}".format(
                    number, len(fields), "" if len(fields) == 1 else "s", line
                )
            )
        _integer(fields[0], number, "item index")
        profits.append(_value(fields[1], number, "profit"))
        weights.append(_value(fields[2], number, "weight"))
        index += 1

    if len(profits) != count:
        where = lines[index][0] if index < len(lines) else lines[-1][0]
        raise InstanceError(
            "line {}: line {} states {} items, but the instance ends after "
            "{}".format(where, count_line, count, len(profits))
        )
    if index < len(lines) and "," in lines[index][1]:
        raise InstanceError(
            "line {}: line {} states {} items, and a further item row "
            "follows".format(lines[index][0], count_line, count)
        )
    if index < len(lines) and _is_rule(lines[index][1]):
        index += 1

    return (
        _assemble(
            stated or fallback, source, "pisinger",
            profits, weights, capacity, optimum, optimum_line,
        ),
        index,
    )


# ---------------------------------------------------------------------------
# the plain Martello-Toth layouts
# ---------------------------------------------------------------------------

def _parse_plain(
    lines: Sequence[Tuple[int, str]], name: str, source: str
) -> Knapsack:
    number, line = lines[0]
    words = line.split()
    if len(words) == 2:
        count = _count(words[0], number, "item count")
        capacity = _count(words[1], number, "capacity")
        rows = lines[1:]
    elif len(words) == 1:
        count = _count(words[0], number, "item count")
        capacity, rows = _capacity_of(lines, count)
    else:
        raise InstanceError(
            "line {}: expected the item count, alone or followed by the "
            "capacity, got {} numbers".format(number, len(words))
        )

    if len(rows) != count:
        where = rows[0][0] if rows else number
        raise InstanceError(
            "line {}: line {} states {} items, but the file gives {} item "
            "line{}".format(
                where, number, count, len(rows), "" if len(rows) == 1 else "s"
            )
        )

    profits: List[int] = []
    weights: List[int] = []
    for row_number, row in rows:
        pair = row.split()
        if len(pair) != 2:
            raise InstanceError(
                "line {}: expected an item line 'profit weight', got {} "
                "number{} in {!r}".format(
                    row_number, len(pair), "" if len(pair) == 1 else "s", row
                )
            )
        profits.append(_value(pair[0], row_number, "profit"))
        weights.append(_value(pair[1], row_number, "weight"))

    return _assemble(
        name, source, "martello-toth",
        profits, weights, capacity, None, number,
    )


def _capacity_of(
    lines: Sequence[Tuple[int, str]], count: int
) -> Tuple[int, List[Tuple[int, str]]]:
    """The capacity and the item lines, when the count stood alone.

    Two orderings are in circulation and they are not distinguished by their
    contents: ``n, c, then the pairs`` and ``n, the pairs, then c`` hold the
    same ``2n + 1`` numbers in the same reading order.  What distinguishes them
    is the shape of the lines -- the lone number sits either before the pairs or
    after them -- and where the file has no such shape, because every number is
    on a line of its own or all of them share one, nothing distinguishes them at
    all.  The two readings then disagree about every item, so this refuses.
    """
    rest = list(lines[1:])
    numbers = sum(len(line.split()) for _, line in rest)
    if numbers != 2 * count + 1:
        raise InstanceError(
            "line {}: line {} states {} items, so {} numbers should follow "
            "it: a capacity and a profit and a weight each, but {} "
            "do".format(
                rest[0][0] if rest else lines[0][0],
                lines[0][0], count, 2 * count + 1, numbers,
            )
        )

    alone = [position for position, (_, line) in enumerate(rest)
             if len(line.split()) == 1]
    paired = len(rest) - len(alone)
    if len(alone) == 1 and paired == count and alone[0] in (0, len(rest) - 1):
        if alone[0] == 0:
            return _count(rest[0][1], rest[0][0], "capacity"), rest[1:]
        return _count(rest[-1][1], rest[-1][0], "capacity"), rest[:-1]

    number, line = rest[0]
    raise InstanceError(
        "line {}: cannot tell whether {} is the capacity or the first profit.  "
        "This file has {} numbers on {} lines, so both 'count, capacity, then "
        "a profit and weight per line' and 'count, then a profit and weight "
        "per line, then capacity' fit it, and they disagree about every item.  "
        "Write the items two numbers to a line, which tells the two "
        "apart".format(number, line.split()[0], numbers + 1, len(rest) + 1)
    )
