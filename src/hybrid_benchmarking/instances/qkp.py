"""Reading quadratic knapsack instance files.

The benchmark set everyone means by "the (QKP) instances" is Billionnet and
Soutif's, distributed at ``https://cedric.cnam.fr/~soutif/QKP/`` as files named
``jeu_<n>_<density>_<index>.txt``.  What follows was read off ``jeu_100_100_1``,
``jeu_100_25_1`` and ``jeu_200_25_1`` together with the site's own ``format.html``
and the typeset objective it links (``instance2.gif``), not off a remembered
description::

    r_100_100_1                     the instance reference
    100                             n, the number of variables
    91 78 22 4 ...                  n linear coefficients c_i
    55 23 35 44 ...                 quadratic row 1: c_12 c_13 ... c_1n
    92 11 20 43 ...                 quadratic row 2: c_23 c_24 ... c_2n
    ...
    17                              quadratic row n-1: c_{n-1,n}
                                    a blank line
    0                               0 for '<= capacity', 1 for '= capacity'
    145                             the capacity
    34 33 12 3 ...                  n weights a_i
                                    a blank line, then free-form comments

Four parts of that are worth stating because a plausible instance comes out of
getting any of them wrong.

**The quadratic block is the strict upper triangle**, ``n - 1`` rows, row ``i``
holding ``n - i - 1`` entries -- 4950 numbers for ``n = 100``, which is
``100 * 99 / 2`` and not ``100 * 101 / 2``.  The diagonal is absent because the
linear coefficients *are* the diagonal: at 25 % density 72 of the 100 linear
coefficients of ``jeu_100_25_1`` are zero, in the same proportion as its
quadratic entries, so the density the file quotes counts ``c_ii`` among the
coefficients it thins out.

**The capacity comes before the weights**, which is the reverse of how the
layout is often described.  The file settles it twice over: ``jeu_100_25_1``
writes ``669`` on one line and a hundred numbers between 1 and 50 on the next,
and the site's typeset instance shows ``34x_1 + ... + 39x_10 <= 145`` against a
file reading ``145`` then ``34 33 12 ...``.  Both orders are nevertheless read
here, because the two lines are told apart by how many numbers they hold rather
than by their position -- one against ``n`` -- and a count is not a guess.  The
one case where counting cannot decide, ``n = 1``, is refused rather than picked.

**A pair's entry is the whole pair's profit.**  The objective is
``max sum_i c_i x_i + sum_{i<j} c_ij x_i x_j``: the typeset instance for
``r_10_100_13`` prints ``+ 55 x_1x_2 + 23 x_1x_3 + ...`` once per unordered
pair, and those coefficients are the first triangle row of that file, read left
to right.  There is no ``c_ji`` anywhere to add to them.  So each entry goes
into :attr:`~.QuadraticKnapsack.pairs` as it stands, keyed ``(j, i)`` with
``j > i``, with nothing doubled and nothing halved.  This is the one reading
here whose error would be invisible: the library's circuits add once per
unordered pair and their cost turns on the position of the lowest set bit of
what they add, so a pair carried at twice its value shifts that bit by one and
returns a gate count that sums and plots like a right one.  A symmetric file --
one storing both ``c_ij`` and ``c_ji``, each half the pair -- would have to be
summed instead; this format is not one, having ``n(n-1)/2`` entries rather than
``n^2``.

**The indicator line says which constraint the file means.**  ``0`` is
``<= capacity``, which is the quadratic knapsack and the only value any
distributed instance carries.  ``1`` is an equality constraint, which is a
different problem -- :class:`~.QuadraticKnapsack` holds a budget, not a target
-- and is refused rather than quietly relaxed.  Anything else is refused as
unrecognised.

Two consequences of the contract, both deliberate:

Profits and weights are **positive integers**, as in :mod:`.knapsack` and for
the same reason -- the circuits read their binary representations, so a zero or
a fraction is meaningless rather than imprecise.  This refuses every instance
below 100 % density, whose linear coefficients are largely zero.  That is a
refusal at the point where it is still a file-format problem, with a line
number, rather than a surprise inside a gate count; the alternative, dropping
zero-profit items, would renumber every pair in the file.

A **negative pair profit is refused**; a zero one is simply absent from
:attr:`~.QuadraticKnapsack.pairs`, which is what "absent pairs earn nothing"
means.  Bonuses only: the circuit has a gate where a pair earns something
together and none where it costs something.

The distributed files state no optimum, so :attr:`~.QuadraticKnapsack.optimum`
is ``None``.  The optima exist -- in the result tables ``N<n>D<density>.txt``
beside the instances -- but those are separate files keyed by the generator's
seed, and reading one instance's claim out of another file is not something
this reader can do without being handed it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from . import InstanceError, QuadraticKnapsack

__all__ = ["LAYOUT", "parse", "read"]

#: What :func:`..detect` calls this, and what the instances report.
LAYOUT = "quadratic-knapsack"


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def parse(text: str, name: str = "", source: str = "") -> QuadraticKnapsack:
    """The instance ``text`` holds.

    ``name`` is used only when the text states no reference of its own, which
    the layout always does.  Every complaint names the line it is about, and
    every disagreement between what a line declares and what follows it is a
    complaint rather than a repair.
    """
    lines = _lines(text)
    index = _skip_blank(lines, 0)
    if index == len(lines):
        raise InstanceError(
            "{} holds no quadratic knapsack instance: the file is "
            "empty".format(source or "the text")
        )

    stated, index = _reference(lines, index)
    count, count_line, index = _item_count(lines, index)
    profits, index = _linear(lines, index, count, count_line)
    pairs, index = _quadratic(lines, index, count, count_line)
    index = _indicator(lines, index)
    capacity, weights = _constraint(lines, index, count, count_line)

    return QuadraticKnapsack(
        name=stated or name,
        source=source,
        layout=LAYOUT,
        profits=tuple(profits),
        weights=tuple(weights),
        capacity=capacity,
        pairs=pairs,
        optimum=None,
    )


def read(path: Union[str, Path]) -> QuadraticKnapsack:
    """The instance in a file.

    ``source`` becomes the path and ``name`` the reference the file states,
    falling back to the file stem for a file that states none.
    """
    location = Path(path).expanduser()
    try:
        text = location.read_text(errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))
    return parse(text, name=location.stem, source=str(location))


# ---------------------------------------------------------------------------
# lines, and the numbers on them
# ---------------------------------------------------------------------------

def _lines(text: str) -> List[Tuple[int, str]]:
    """Every line, stripped, carrying the number the user's editor shows.

    Blank lines are kept rather than dropped, because one of them is load
    bearing: the blank line after the quadratic block is what says the block
    has ended, and without it a block one row short would swallow the
    constraint type and read on quite happily.
    """
    return [(number, raw.strip())
            for number, raw in enumerate(text.splitlines(), 1)]


def _skip_blank(lines: Sequence[Tuple[int, str]], index: int) -> int:
    while index < len(lines) and not lines[index][1]:
        index += 1
    return index


def _require(
    lines: Sequence[Tuple[int, str]], index: int, what: str
) -> Tuple[int, str]:
    """The next line, or a complaint that the file ended before it."""
    if index >= len(lines):
        where = lines[-1][0] if lines else 1
        raise InstanceError(
            "line {}: the file ends before {}".format(where, what)
        )
    return lines[index]


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


def _value(token: str, number: int, what: str) -> int:
    """A profit or a weight: a positive integer, and nothing else.

    Word for word the rule :mod:`.knapsack` states, and for the same reason.
    The value is read downstream as a binary representation -- the cost of a
    comparison depends on where its ones sit, and in particular on the position
    of its lowest set bit -- so zero, negative and fractional values have no
    meaning there.  A format error naming this line is worth more than an
    arithmetic surprise inside a gate count.
    """
    try:
        value = int(token)
    except ValueError:
        raise InstanceError(
            "line {}: the {} is {!r}, which is not an integer.  Profits and "
            "weights are read as binary representations -- the circuit cost "
            "depends on where the ones sit -- so a fractional value has no "
            "meaning downstream".format(number, what, token)
        )
    if value <= 0:
        raise InstanceError(
            "line {}: the {} is {}, and must be a positive integer.  Profits "
            "and weights are read as binary representations -- the circuit "
            "cost depends on the position of the lowest set bit -- so a zero "
            "or negative value has no meaning downstream".format(
                number, what, value
            )
        )
    return value


def _bonus(token: str, number: int, first: int, second: int) -> int:
    """What a pair earns together: a non-negative integer.

    Zero is allowed and carries no entry, which is how the sparse instances
    say a pair earns nothing.  Negative is refused, and the message says why:
    the circuit that costs this has a gate where a pair earns something when
    both items are chosen and none where a pair costs something, so there is
    nothing for a negative bonus to be counted as.  It would not fail on its
    own, either -- the lowest set bit of a negative number is a perfectly good
    integer, and a gate count would come back.
    """
    try:
        value = int(token)
    except ValueError:
        raise InstanceError(
            "line {}: the profit of the pair (item {}, item {}) is {!r}, "
            "which is not an integer.  Pair profits are read as binary "
            "representations, like the linear ones, so a fractional value has "
            "no meaning downstream".format(number, first, second, token)
        )
    if value < 0:
        raise InstanceError(
            "line {}: the profit of the pair (item {}, item {}) is {}, and a "
            "pair profit must not be negative.  These are bonuses only: the "
            "circuit has a gate for a pair that earns something when both "
            "items are chosen and no gate for a pair that costs something, so "
            "a negative entry has nothing to count -- and it would not fail "
            "loudly either, the lowest set bit of a negative number being a "
            "perfectly good integer".format(number, first, second, value)
        )
    return value


# ---------------------------------------------------------------------------
# the parts of the layout, in the order the file writes them
# ---------------------------------------------------------------------------

def _reference(
    lines: Sequence[Tuple[int, str]], index: int
) -> Tuple[str, int]:
    """The instance reference, which every distributed file states first."""
    number, line = lines[index]
    words = line.split()
    if len(words) != 1:
        raise InstanceError(
            "line {}: expected the instance reference alone on the first "
            "line, such as 'r_100_25_1', got {} words in {!r}".format(
                number, len(words), line
            )
        )
    return words[0], _skip_blank(lines, index + 1)


def _item_count(
    lines: Sequence[Tuple[int, str]], index: int
) -> Tuple[int, int, int]:
    """``n``, the line it was stated on, and where to read on.

    A file written without the reference line reads its count as the reference
    and its linear coefficients as the count, which lands here: hence the
    hint, rather than an attempt to work out which of the two the first line
    was.  Both readings fit such a file and they disagree about every number
    in it.
    """
    number, line = _require(lines, index, "the number of items")
    words = line.split()
    if len(words) != 1:
        raise InstanceError(
            "line {}: expected the number of items alone on a line, got {} "
            "numbers in {!r}.  The layout states the instance reference "
            "first and the count second, so a file written without a "
            "reference line arrives here with its linear profits".format(
                number, len(words), line
            )
        )
    return _count(words[0], number, "number of items"), number, index + 1


def _linear(
    lines: Sequence[Tuple[int, str]], index: int, count: int, count_line: int
) -> Tuple[List[int], int]:
    """The ``n`` linear coefficients, all on one line as the layout writes."""
    index = _skip_blank(lines, index)
    number, line = _require(lines, index, "the linear profits")
    tokens = line.split()
    if len(tokens) != count:
        raise InstanceError(
            "line {}: line {} states {} items, so this line should hold {} "
            "linear profits, and it holds {}".format(
                number, count_line, count, count, len(tokens)
            )
        )
    profits = [
        _value(token, number, "linear profit of item {}".format(position + 1))
        for position, token in enumerate(tokens)
    ]
    return profits, index + 1


def _quadratic(
    lines: Sequence[Tuple[int, str]], index: int, count: int, count_line: int
) -> Tuple[Dict[Tuple[int, int], int], int]:
    """The strict upper triangle, row by row, as pairs keyed ``(j, i)``.

    The block is the run of non-blank lines after the linear profits, and it
    must hold ``n - 1`` rows of ``n - 1, n - 2, ..., 1`` entries.  Both the row
    count and each row's width are checked against ``n`` rather than trusted,
    because a triangle read one row out is still a triangle: it pairs item 3
    with item 5 where the file paired item 3 with item 4, and every number
    downstream stays plausible.
    """
    if count == 1:
        return {}, index  # one item pairs with nothing: the block is empty

    start = index = _skip_blank(lines, index)
    block: List[Tuple[int, str]] = []
    while index < len(lines) and lines[index][1]:
        block.append(lines[index])
        index += 1

    if len(block) != count - 1:
        where = block[-1][0] if block else lines[start - 1][0]
        raise InstanceError(
            "line {}: line {} states {} items, so the quadratic block holds "
            "{} rows -- one per item but the last -- and the block starting "
            "at line {} holds {}.  The layout ends that block with a blank "
            "line before the constraint; a missing blank line reads the "
            "constraint as further rows".format(
                where, count_line, count, count - 1,
                block[0][0] if block else lines[start - 1][0], len(block),
            )
        )

    pairs: Dict[Tuple[int, int], int] = {}
    for row, (number, line) in enumerate(block):
        tokens = line.split()
        expected = count - row - 1
        if len(tokens) != expected:
            raise InstanceError(
                "line {}: row {} of the quadratic block pairs item {} with "
                "the {} item{} after it, so it should hold {} entr{}, and it "
                "holds {}".format(
                    number, row + 1, row + 1, expected,
                    "" if expected == 1 else "s", expected,
                    "y" if expected == 1 else "ies", len(tokens),
                )
            )
        for offset, token in enumerate(tokens):
            column = row + 1 + offset
            bonus = _bonus(token, number, row + 1, column + 1)
            if bonus:
                pairs[(column, row)] = bonus

    return pairs, index


def _indicator(lines: Sequence[Tuple[int, str]], index: int) -> int:
    """The ``0``/``1`` line, and where the constraint itself begins."""
    index = _skip_blank(lines, index)
    number, line = _require(lines, index, "the constraint type, '0' or '1'")
    words = line.split()
    if len(words) == 1 and words[0] == "0":
        return index + 1
    if len(words) == 1 and words[0] == "1":
        raise InstanceError(
            "line {}: the constraint type is 1, an equality constraint -- the "
            "file asks for a selection weighing exactly the capacity, not at "
            "most it.  That is a different problem from the quadratic "
            "knapsack this reads, so it is refused rather than relaxed to "
            "'at most'".format(number)
        )
    raise InstanceError(
        "line {}: expected the constraint type, '0' for 'at most the "
        "capacity' or '1' for 'exactly the capacity', got {!r}".format(
            number, line
        )
    )


def _constraint(
    lines: Sequence[Tuple[int, str]], index: int, count: int, count_line: int
) -> Tuple[int, List[int]]:
    """The capacity and the ``n`` weights, in whichever order they are written.

    The distributed files write the capacity first.  A file writing the
    weights first holds the same two lines the other way round, and the two
    are told apart by their widths -- one number against ``n`` -- so reading
    both is a matter of counting rather than of guessing.  When ``n`` is 1 the
    widths coincide, the two readings disagree about both numbers, and this
    refuses.
    """
    first_index = _skip_blank(lines, index)
    first = _require(lines, first_index, "the capacity and the weights")
    second_index = _skip_blank(lines, first_index + 1)
    second = _require(lines, second_index, "the weights")

    widths = (len(first[1].split()), len(second[1].split()))
    if count == 1 and widths == (1, 1):
        raise InstanceError(
            "line {}: cannot tell whether {} is the capacity or the weight of "
            "the single item.  This instance has one item, so the capacity "
            "line and the weight line hold one number each and nothing "
            "distinguishes them; the distributed layout writes the capacity "
            "first, but the two readings disagree about both numbers".format(
                first[0], first[1]
            )
        )
    if widths == (1, count):
        capacity_line, weight_line = first, second
    elif widths == (count, 1):
        capacity_line, weight_line = second, first
    else:
        raise InstanceError(
            "line {}: line {} states {} items, so the constraint is a "
            "capacity on a line of its own and {} weights on another, in "
            "either order; lines {} and {} hold {} and {} numbers".format(
                first[0], count_line, count, count,
                first[0], second[0], widths[0], widths[1],
            )
        )

    capacity = _count(capacity_line[1], capacity_line[0], "capacity")
    weights = [
        _value(token, weight_line[0], "weight of item {}".format(position + 1))
        for position, token in enumerate(weight_line[1].split())
    ]
    return capacity, weights
