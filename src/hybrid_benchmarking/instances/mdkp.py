"""Reading multidimensional knapsack instance files.

One value per item, one cost per item per dimension, and one budget per
dimension.  OR-Library holds the two collections everyone benchmarks against,
and they are **not one format**.  What follows was read off the files
themselves -- ``mknap1.txt``, ``mknap2.txt`` and ``mknapcb1.txt`` to
``mknapcb9.txt``, fetched from
``https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/`` -- rather than
remembered, because the two differ in the order of the very first two numbers,
and a file read with the wrong one of them still produces a knapsack.

**The counted layout**, which is what ``mknap1.txt`` and every ``mknapcb`` file
hold, contrary to the folklore that ``mknap1`` is the odd one out.  A count of
problems, then each problem as its item count, its dimension count and its
optimal value, then the values, then the costs a dimension at a time, then the
budgets::

    7                                   <- 7 problems in this file
    6 10 3800                           <- 6 items, 10 dimensions, optimum 3800
    100 600 1200 2400 500 2000          <- the 6 profits
    8 12 13 64 22 41                    <- what each item costs in dimension 1
    ...                                    (10 such rows)
    80 96 20 36 44 48 10 18 22 24       <- the 10 capacities

**The mknap2 layout**, which is what ``mknap2.txt`` holds, and which the file
states in its own preamble.  No leading problem count; the two header numbers
are the other way round -- *dimensions first, then items*; the budgets come
before the cost matrix; and the optimum sits at the **end** of the instance,
not in its header::

    problem PB7.DAT                     <- the instance's name
    +++++++++++++++++++++++++++++
    30 37                               <- 30 dimensions, 37 items
    47 77 110 67 65 3 6 39 33 63        <- the 37 profits
    6 21 56 29 69 61                       (wrapped over five lines, 10, 6,
    ...                                     10, 10 and 1 numbers wide)
    5875 4351 5221 7099 ...             <- the 30 capacities
    785 774 818 56 699 22 42 465 21     <- dimension 1's costs, 37 of them
    ...                                    (30 such rows)
    1035                                <- the optimum

Both layouts write the cost matrix **one row per dimension**, which is already
:attr:`~.MultidimensionalKnapsack.weights`'s orientation: ``weights[i][m]`` is
what item ``m`` costs in dimension ``i``.  Neither needs transposing, and a
reader that transposed one would produce an instance of the right size on the
square problems and nonsense on the rest.

Two things about the real files, observed and worth knowing before trusting a
count built on one:

* **The numbers wrap wherever the writer ran out of room.**  ``PB7``'s 37
  profits arrive ten, six, ten, ten and one to a line; ``mknapcb``'s hundred
  arrive seven to a line, so each of its cost rows ends on a line of two.  Only
  the opening line of numbers is read as a line, to tell the two layouts apart
  by its width; everything after it is a stream of numbers.
* **Zero costs are common, and this reader refuses them.**  Every one of
  ``mknap1``'s seven problems has zero entries in its cost matrix (99 of them
  across the file), as do 44 of ``mknap2``'s 48; ``mknap1``'s second problem
  has fractional profits besides.  The nine ``mknapcb`` files are clean:
  positive integers throughout, 270 problems.  Profits and weights are read
  downstream as binary representations -- the cost of a comparison depends on
  where the ones sit, and in particular on the position of the lowest set bit
  -- so a zero or a fraction is not an approximation of anything.  Refusing it
  here keeps it a complaint about a file, naming the item and the dimension, in
  place of an arithmetic surprise inside a gate count.

Three rulings, none of them forced by the files:

* **Which layout a file is in is decided by the width of its first line of
  numbers**, and by nothing else: one number means the counted layout's problem
  count, two mean the mknap2 layout's dimension and item counts.  Every one of
  the eleven files follows this, and every one of ``mknap2``'s 48 instances
  does.  A file that opens with some other width is refused rather than tried
  both ways -- the two readings disagree about every value, so a wrong guess
  yields a plausible instance and no error at all.
* **An optimum of 0 means "not stated"**, in both layouts, and becomes ``None``.
  That is what the ``mknapcb`` files mean by it -- all 270 of their problems
  carry it, none of ``mknap1``'s or ``mknap2``'s do.  Since profits are
  positive, a true optimum of zero would say that no single item fits any
  budget, which no benchmark set is written to express.
* **Everything before the first ``problem <name>`` marker is documentation.**
  ``mknap2.txt`` opens with eighty lines of bibliography and a worked example
  before its first instance.  Where such markers are present each instance
  belongs to one, takes its name from it, and must be consumed exactly; where
  they are absent every word in the file has to be a number.

Four entry points, and :func:`parse` and :func:`read` make the same choice the
0-1 reader makes: they return the one instance a file holds and **raise** when
it holds more, naming :func:`parse_all` and :func:`read_all`.  These files hold
seven, thirty and forty-eight, so returning the first quietly is how the other
forty-seven go missing.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from . import InstanceError, MultidimensionalKnapsack

__all__ = ["LAYOUT", "parse", "parse_all", "read", "read_all"]

#: What :func:`..detect` returns for these files, and what every instance read
#: here reports.  Both layouts report it: they are two writings of one problem,
#: as the 0-1 reader's three Martello-Toth punctuations are.
LAYOUT = "multidimensional-knapsack"

#: ``problem WEING1.DAT`` -- the name mknap2 gives an instance.
_MARKER = re.compile(r"(?i)^problem\s+(\S+)$")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def parse(text: str, name: str = "", source: str = "") -> MultidimensionalKnapsack:
    """The single instance ``text`` holds.

    Raises :class:`InstanceError` if it holds more than one, rather than
    returning the first and dropping the rest.  :func:`parse_all` is the way to
    read the OR-Library files, every one of which holds several.
    """
    found = parse_all(text, name=name, source=source)
    if len(found) > 1:
        raise InstanceError(
            "{} holds {} problems back to back, and this returns one.  Use "
            "parse_all() or read_all(), which return them in file order; "
            "returning the first would discard {} problems silently.".format(
                source or "the text", len(found), len(found) - 1
            )
        )
    return found[0]


def parse_all(
    text: str, name: str = "", source: str = ""
) -> Tuple[MultidimensionalKnapsack, ...]:
    """Every instance ``text`` holds, in file order.

    A file of one gives a tuple of one, so a caller who does not know which
    they hold can use this and be right either way.
    """
    lines = _clean(text)
    if not lines:
        raise InstanceError(
            "{} holds no multidimensional knapsack instance: the file is "
            "empty".format(source or "the text")
        )

    found: List[MultidimensionalKnapsack] = []
    for block_name, block in _blocks(lines, name):
        found.extend(_block(block, block_name, source))
    if not found:
        raise InstanceError(
            "{} holds no multidimensional knapsack instance".format(
                source or "the text"
            )
        )
    return tuple(found)


def read(path: Union[str, Path]) -> MultidimensionalKnapsack:
    """The single instance in a file, raising when the file holds several."""
    location = Path(path).expanduser()
    return parse(_text_of(location), name=location.stem, source=str(location))


def read_all(path: Union[str, Path]) -> Tuple[MultidimensionalKnapsack, ...]:
    """Every instance in a file, in file order."""
    location = Path(path).expanduser()
    return parse_all(
        _text_of(location), name=location.stem, source=str(location)
    )


def _text_of(location: Path) -> str:
    try:
        return location.read_text(errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))


# ---------------------------------------------------------------------------
# lines, blocks, and the numbers in them
# ---------------------------------------------------------------------------

def _clean(text: str) -> List[Tuple[int, str]]:
    """Non-blank lines, each carrying the line number it came from.

    Blank lines, banner rules of ``+`` and trailing ``//`` comments go: all
    three are punctuation in ``mknap2.txt``, which documents the ``//`` form in
    its own worked example.  The line numbers are the ones the user sees in
    their editor, since every complaint below names one.
    """
    kept: List[Tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.split("//")[0].strip()
        if not stripped:
            continue
        if set(stripped) == {"+"}:
            continue
        kept.append((number, stripped))
    return kept


def _blocks(
    lines: Sequence[Tuple[int, str]], fallback: str
) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """The file's named sections, or the whole file as one unnamed section.

    ``mknap2.txt`` names each of its instances with a ``problem NAME`` line and
    prefaces the lot with eighty lines of bibliography.  Where such markers are
    present, the preamble is the text before the first of them and is dropped;
    where they are absent nothing is dropped, and every word left has to be a
    number.
    """
    marked = [
        (number, line, _MARKER.match(line)) for number, line, in lines
    ]
    if not any(match for _, _, match in marked):
        return [(fallback, list(lines))]

    blocks: List[Tuple[str, List[Tuple[int, str]]]] = []
    current: Optional[List[Tuple[int, str]]] = None
    for number, line, match in marked:
        if match is not None:
            current = []
            blocks.append((match.group(1), current))
        elif current is not None:
            current.append((number, line))
    for name, body in blocks:
        if not body:
            raise InstanceError(
                "the instance named {!r} has no numbers under it".format(name)
            )
    return blocks


def _numbers(lines: Sequence[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Every word in ``lines``, with its line number, refusing non-numbers.

    This is where reading these files line by line would go wrong and where
    this reader stops doing it: the values run across line boundaries wherever
    the writer ran out of room, so from here on a block is a stream of numbers
    and the lines survive only to be named in complaints.
    """
    tokens: List[Tuple[int, str]] = []
    for number, line in lines:
        for word in line.split():
            if not _numeric(word):
                raise InstanceError(
                    "line {}: expected a number, got {!r}.  Outside the "
                    "documentation that precedes a 'problem <name>' marker, "
                    "these files hold nothing but numbers".format(number, word)
                )
            tokens.append((number, word))
    return tokens


def _numeric(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


class _Stream:
    """The numbers of one block, taken in order, each knowing its line."""

    def __init__(self, tokens: Sequence[Tuple[int, str]]) -> None:
        self._tokens = list(tokens)
        self._at = 0

    def __len__(self) -> int:
        return len(self._tokens) - self._at

    @property
    def line(self) -> int:
        """The line the next number sits on, or the last line of the block."""
        if self._at < len(self._tokens):
            return self._tokens[self._at][0]
        return self._tokens[-1][0]

    def take(self, what: str) -> Tuple[int, str]:
        if self._at >= len(self._tokens):
            raise InstanceError(
                "line {}: the file ends before {}".format(self.line, what)
            )
        token = self._tokens[self._at]
        self._at += 1
        return token

    def several(self, count: int, what: str) -> List[Tuple[int, str]]:
        return [
            self.take("{} {} of {}".format(what, index + 1, count))
            for index in range(count)
        ]

    def count(self, what: str) -> int:
        """The next number, as something that counts and so is positive."""
        line, word = self.take(what)
        return _count(line, word, what)


# ---------------------------------------------------------------------------
# which layout, and reading it
# ---------------------------------------------------------------------------

def _block(
    lines: Sequence[Tuple[int, str]], name: str, source: str
) -> List[MultidimensionalKnapsack]:
    """Every instance in one block, in file order."""
    width = len(lines[0][1].split())
    tokens = _numbers(lines)
    if width == 1:
        found = _counted(_Stream(tokens), name, source)
    elif width == 2:
        found = _mknap2(_Stream(tokens), name, source)
    else:
        raise InstanceError(
            "line {}: cannot tell which layout this is.  The counted layout "
            "of mknap1 and mknapcb opens with the number of problems alone on "
            "a line; the mknap2 layout opens with a dimension count and an "
            "item count, two numbers on a line.  This opens with {} numbers, "
            "and the two readings disagree about every value that "
            "follows".format(lines[0][0], width)
        )
    if len(found) > 1:
        return [
            _renamed(instance, "{}-{}".format(name, index + 1))
            for index, instance in enumerate(found)
        ]
    return found


def _renamed(
    instance: MultidimensionalKnapsack, name: str
) -> MultidimensionalKnapsack:
    """The same instance under a numbered name.

    The counted layout names no problem, so within a file of thirty the file's
    own name identifies all thirty equally badly; a suffix says which.
    """
    return replace(instance, name=name)


def _counted(
    stream: _Stream, name: str, source: str
) -> List[MultidimensionalKnapsack]:
    """mknap1 and mknapcb: a problem count, then ``n m optimum`` each time."""
    problems = stream.count("the number of problems")

    found: List[MultidimensionalKnapsack] = []
    for index in range(problems):
        which = "problem {} of {}".format(index + 1, problems)
        items = stream.count("the item count of " + which)
        dimensions = stream.count("the dimension count of " + which)
        optimum = stream.take("the stated optimum of " + which)
        profits = stream.several(items, "the profit of " + which + ", item")
        weights = [
            stream.several(
                items,
                "{}, dimension {} of {}, item".format(which, row + 1, dimensions),
            )
            for row in range(dimensions)
        ]
        capacities = stream.several(
            dimensions, "the capacity of " + which + ", dimension"
        )
        found.append(
            _instance(name, source, profits, weights, capacities, optimum)
        )

    if len(stream):
        raise InstanceError(
            "line {}: the file states {} problem{}, and {} further number{} "
            "follow the last of them".format(
                stream.line, problems, "" if problems == 1 else "s",
                len(stream), "" if len(stream) == 1 else "s",
            )
        )
    return found


def _mknap2(
    stream: _Stream, name: str, source: str
) -> List[MultidimensionalKnapsack]:
    """mknap2: dimensions before items, budgets before costs, optimum last.

    There is no problem count, so instances are read until the numbers run
    out; a block that ends partway through one is refused, naming what it ended
    before.
    """
    found: List[MultidimensionalKnapsack] = []
    while len(stream):
        which = "problem {}".format(len(found) + 1)
        dimensions = stream.count("the dimension count of " + which)
        items = stream.count("the item count of " + which)
        profits = stream.several(items, "the profit of " + which + ", item")
        capacities = stream.several(
            dimensions, "the capacity of " + which + ", dimension"
        )
        weights = [
            stream.several(
                items,
                "{}, dimension {} of {}, item".format(which, row + 1, dimensions),
            )
            for row in range(dimensions)
        ]
        optimum = stream.take("the stated optimum of " + which)
        found.append(
            _instance(name, source, profits, weights, capacities, optimum)
        )
    return found


# ---------------------------------------------------------------------------
# what the numbers are allowed to be
# ---------------------------------------------------------------------------

def _count(line: int, word: str, what: str) -> int:
    """An integer that counts something, and so must be positive."""
    try:
        value = int(word)
    except ValueError:
        raise InstanceError(
            "line {}: {} is {!r}, which is not an integer".format(
                line, what, word
            )
        )
    if value <= 0:
        raise InstanceError(
            "line {}: {} is {}, and must be a positive integer".format(
                line, what, value
            )
        )
    return value


def _value(line: int, word: str, what: str) -> int:
    """A profit or a weight: a positive integer, and nothing else.

    The refusal is deliberate and its reason belongs in the message.  These
    values are read downstream as binary representations -- the cost of a
    comparison depends on where the ones sit, and in particular on the position
    of the lowest set bit -- so zero, negative and fractional values have no
    meaning there.  Real OR-Library files do contain zeros: every problem in
    ``mknap1.txt`` and 44 of ``mknap2.txt``'s 48 have them in the cost matrix,
    which is why the message names the item and the dimension, so that whoever
    hit it can see what their file actually holds.
    """
    try:
        value = int(word)
    except ValueError:
        raise InstanceError(
            "line {}: {} is {!r}, which is not an integer.  Profits and "
            "weights are read as binary representations -- the circuit cost "
            "depends on where the ones sit -- so a fractional value has no "
            "meaning downstream".format(line, what, word)
        )
    if value <= 0:
        raise InstanceError(
            "line {}: {} is {}, and must be a positive integer.  Profits and "
            "weights are read as binary representations -- the circuit cost "
            "depends on the position of the lowest set bit -- so a zero or "
            "negative value has no meaning downstream".format(line, what, value)
        )
    return value


def _instance(
    name: str,
    source: str,
    profits: Sequence[Tuple[int, str]],
    weights: Sequence[Sequence[Tuple[int, str]]],
    capacities: Sequence[Tuple[int, str]],
    optimum: Tuple[int, str],
) -> MultidimensionalKnapsack:
    """One instance, once every number in it has been checked."""
    checked_profits = tuple(
        _value(line, word, "the profit of item {}".format(index + 1))
        for index, (line, word) in enumerate(profits)
    )
    checked_weights = tuple(
        tuple(
            _value(
                line, word,
                "the weight of item {} in dimension {}".format(
                    item + 1, dimension + 1
                ),
            )
            for item, (line, word) in enumerate(row)
        )
        for dimension, row in enumerate(weights)
    )
    checked_capacities = tuple(
        _count(line, word, "the capacity of dimension {}".format(index + 1))
        for index, (line, word) in enumerate(capacities)
    )

    line, word = optimum
    try:
        stated = int(word)
    except ValueError:
        raise InstanceError(
            "line {}: the stated optimum is {!r}, which is not an integer.  "
            "Every profit is a positive integer, so every achievable value "
            "is one too".format(line, word)
        )
    if stated < 0:
        raise InstanceError(
            "line {}: the stated optimum is {}, and no selection of positive "
            "profits is negative".format(line, stated)
        )
    if stated > sum(checked_profits):
        raise InstanceError(
            "line {}: the stated optimum {} exceeds {}, the profit of taking "
            "every item, so no selection achieves it -- the profits and the "
            "weights are probably not being read in the order the file writes "
            "them".format(line, stated, sum(checked_profits))
        )

    return MultidimensionalKnapsack(
        name=name,
        source=source,
        layout=LAYOUT,
        profits=checked_profits,
        weights=checked_weights,
        capacities=checked_capacities,
        optimum=stated or None,
    )
