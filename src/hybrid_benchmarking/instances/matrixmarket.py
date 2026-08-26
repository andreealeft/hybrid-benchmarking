"""Matrix Market, as NIST wrote the specification and as files in the wild
write it.

The format is a banner naming four things -- object, format, field, symmetry --
then a size line, then the entries.  Almost all of the work here is in the
symmetry field: a symmetric file states one triangle and expects the reader to
know the other one is there, and a reader that forgets loses half the non-zeros
of every matrix a linear solver is ever handed.  :attr:`Matrix.entries` holds
both triangles, so nothing downstream has to remember what the header said.

Three readings the specification leaves open, decided here and worth knowing:

* **A complex field is refused.**  :class:`Matrix` holds real values, and a
  complex matrix silently reduced to its real part is a different matrix with a
  plausible-looking condition number.  Refusing is the only honest option.
* **An explicit zero in a coordinate file is kept; a zero in an array file is
  not.**  In coordinate format the presence of a line is itself information --
  the file is stating the sparsity pattern -- so a stated zero stays.  In array
  format every position is written whether or not it holds anything, so a zero
  there says nothing, and keeping them would turn a dense file into a list of
  ``rows * columns`` "non-zeros".
* **Comments are tolerated below the size line.**  The specification allows them
  only above it.  Files do not obey, and a comment cannot be misread as data.

Everything else raises :class:`InstanceError` naming the line: a declared entry
count that disagrees with the number of data lines, an index outside the
declared shape, a diagonal entry in a skew-symmetric file, a duplicate.
Duplicates matter more than they look -- a symmetric file that states both
triangles is the common way for a matrix to arrive with twice the non-zeros it
should have, and it is indistinguishable from a legitimate file unless you check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

from . import InstanceError, Matrix

__all__ = ["parse", "parse_vector", "read", "read_vector"]

#: What :class:`Instance.layout` says for anything this module produces.
LAYOUT = "matrix-market"

_FORMATS = ("coordinate", "array")
_FIELDS = ("real", "integer", "pattern", "complex")
_SYMMETRIES = ("general", "symmetric", "skew-symmetric", "hermitian")


# ---------------------------------------------------------------------------
# the header
# ---------------------------------------------------------------------------

class _Header:
    """The four words of the banner, lower-cased, with the line they were on."""

    def __init__(self, line: int, layout: str, field: str, symmetry: str):
        self.line = line
        self.layout = layout
        self.field = field
        self.symmetry = symmetry

    @property
    def mirrored(self) -> bool:
        """True where the file states one triangle and means two."""
        return self.symmetry != "general"

    @property
    def symmetric(self) -> bool:
        """True where the mirrored entry equals the stated one.

        ``hermitian`` over a real field is symmetric; the specification only
        defines it over a complex one, which we refuse, so a real hermitian file
        is a slight abuse of the format that costs nothing to accept.
        """
        return self.symmetry in ("symmetric", "hermitian")


def _banner(number: int, line: str) -> _Header:
    words = line.split()
    if not words or words[0].lower() != "%%matrixmarket":
        raise InstanceError(
            "line {}: expected a %%MatrixMarket banner, found {!r}".format(
                number, line.strip()[:60]
            )
        )
    if len(words) != 5:
        raise InstanceError(
            "line {}: the banner names object, format, field and symmetry, "
            "four words after %%MatrixMarket, not {}".format(number, len(words) - 1)
        )
    obj, layout, field, symmetry = (word.lower() for word in words[1:])
    if obj != "matrix":
        raise InstanceError(
            "line {}: only the 'matrix' object is read, not {!r}".format(number, obj)
        )
    if layout not in _FORMATS:
        raise InstanceError(
            "line {}: unknown format {!r}; expected one of {}".format(
                number, layout, ", ".join(_FORMATS)
            )
        )
    if field not in _FIELDS:
        raise InstanceError(
            "line {}: unknown field {!r}; expected one of {}".format(
                number, field, ", ".join(_FIELDS)
            )
        )
    if symmetry not in _SYMMETRIES:
        raise InstanceError(
            "line {}: unknown symmetry {!r}; expected one of {}".format(
                number, symmetry, ", ".join(_SYMMETRIES)
            )
        )
    if field == "complex":
        raise InstanceError(
            "line {}: complex matrices are not supported, a Matrix holds real "
            "values, and taking the real part would be a different matrix".format(
                number
            )
        )
    if layout == "array" and field == "pattern":
        raise InstanceError(
            "line {}: a pattern field states which entries exist, which an array "
            "file cannot do; pattern goes with coordinate".format(number)
        )
    return _Header(number, layout, field, symmetry)


def _scan(text: str) -> Tuple[_Header, List[Tuple[int, List[str]]]]:
    """Banner, then every line below it that is neither blank nor a comment."""
    lines = text.splitlines()
    banner_at = 0
    for number, line in enumerate(lines, 1):
        if line.strip():
            banner_at = number
            break
    if not banner_at:
        raise InstanceError("empty file: no %%MatrixMarket banner")
    header = _banner(banner_at, lines[banner_at - 1])

    body = []
    for number, line in enumerate(lines[banner_at:], banner_at + 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        body.append((number, stripped.split()))
    if not body:
        raise InstanceError(
            "line {}: the banner is followed by no size line".format(banner_at)
        )
    return header, body


def _shape(number: int, words: List[str], wanted: int) -> List[int]:
    if len(words) != wanted:
        raise InstanceError(
            "line {}: the size line needs {} numbers, found {}: {!r}".format(
                number, wanted, len(words), " ".join(words)[:60]
            )
        )
    sizes = []
    for word in words:
        try:
            size = int(word)
        except ValueError:
            raise InstanceError(
                "line {}: {!r} in the size line is not a whole number".format(
                    number, word
                )
            )
        if size < 0:
            raise InstanceError(
                "line {}: the size line cannot be negative: {}".format(number, size)
            )
        sizes.append(size)
    return sizes


# ---------------------------------------------------------------------------
# the entries
# ---------------------------------------------------------------------------

def _value(number: int, word: str, field: str) -> float:
    if field == "integer":
        try:
            return float(int(word))
        except ValueError:
            raise InstanceError(
                "line {}: the field is integer, but {!r} is not a whole "
                "number".format(number, word)
            )
    try:
        return float(word)
    except ValueError:
        raise InstanceError("line {}: {!r} is not a number".format(number, word))


def _index(number: int, word: str, limit: int, what: str) -> int:
    """One file index, checked against the declared shape and made zero-based."""
    try:
        value = int(word)
    except ValueError:
        raise InstanceError(
            "line {}: the {} index {!r} is not a whole number".format(
                number, what, word
            )
        )
    if not 1 <= value <= limit:
        raise InstanceError(
            "line {}: the {} index {} is outside 1..{}, which the size line "
            "declared".format(number, what, value, limit)
        )
    return value - 1


class _Entries:
    """Somewhere to put entries that refuses to hold the same one twice.

    The duplicate check is what catches a symmetric file that states both
    triangles: the second triangle collides with the mirror of the first, and the
    message names both lines rather than leaving a matrix with doubled
    off-diagonals to be discovered by whatever it makes wrong.
    """

    def __init__(self, header: _Header):
        self._header = header
        self._at: Dict[Tuple[int, int], int] = {}
        self.rows: List[Tuple[int, int, float]] = []

    def add(self, number: int, row: int, column: int, value: float) -> None:
        header = self._header
        if header.symmetry == "skew-symmetric" and row == column:
            raise InstanceError(
                "line {}: entry ({}, {}) is on the diagonal, which a "
                "skew-symmetric matrix does not have".format(
                    number, row + 1, column + 1
                )
            )
        self._place(number, row, column, value, mirror=False)
        if header.mirrored and row != column:
            mirrored = value if header.symmetric else -value
            self._place(number, column, row, mirrored, mirror=True)

    def _place(
        self, number: int, row: int, column: int, value: float, mirror: bool
    ) -> None:
        seen = self._at.get((row, column))
        if seen is not None:
            if mirror:
                raise InstanceError(
                    "line {}: a {} file states one triangle, but the mirror of "
                    "this entry lands on ({}, {}), already given on line "
                    "{}".format(
                        number, self._header.symmetry, row + 1, column + 1, seen
                    )
                )
            raise InstanceError(
                "line {}: entry ({}, {}) was already given on line {}; Matrix "
                "Market allows each position once".format(
                    number, row + 1, column + 1, seen
                )
            )
        self._at[(row, column)] = number
        self.rows.append((row, column, value))


def _coordinate(
    header: _Header, body: List[Tuple[int, List[str]]]
) -> Tuple[int, int, List[Tuple[int, int, float]]]:
    number, words = body[0]
    rows, columns, declared = _shape(number, words, 3)
    _square(header, number, rows, columns)

    data = body[1:]
    if len(data) != declared:
        raise InstanceError(
            "line {}: the size line declares {} entr{}, but {} data line{} "
            "follow{}".format(
                number, declared, "y" if declared == 1 else "ies",
                len(data), "" if len(data) == 1 else "s",
                "s" if len(data) == 1 else "",
            )
        )

    wanted = 2 if header.field == "pattern" else 3
    entries = _Entries(header)
    for line_number, tokens in data:
        if len(tokens) != wanted:
            raise InstanceError(
                "line {}: a {} coordinate entry is {} number{}, found {}: "
                "{!r}".format(
                    line_number, header.field, wanted, "" if wanted == 1 else "s",
                    len(tokens), " ".join(tokens)[:60],
                )
            )
        row = _index(line_number, tokens[0], rows, "row")
        column = _index(line_number, tokens[1], columns, "column")
        value = 1.0 if header.field == "pattern" else _value(
            line_number, tokens[2], header.field
        )
        entries.add(line_number, row, column, value)
    return rows, columns, entries.rows


def _positions(header: _Header, rows: int, columns: int) -> List[Tuple[int, int]]:
    """Where an array file's values go, in the order it writes them.

    Column by column; and for anything but ``general``, only the lower triangle,
    since that is the half the file stores.
    """
    places = []
    for column in range(columns):
        if header.symmetry == "general":
            first = 0
        elif header.symmetry == "skew-symmetric":
            first = column + 1  # its diagonal is not written, being zero
        else:
            first = column
        for row in range(first, rows):
            places.append((row, column))
    return places


def _array(
    header: _Header, body: List[Tuple[int, List[str]]]
) -> Tuple[int, int, List[Tuple[int, int, float]]]:
    number, words = body[0]
    rows, columns = _shape(number, words, 2)
    _square(header, number, rows, columns)

    data = body[1:]
    places = _positions(header, rows, columns)
    if len(data) != len(places):
        raise InstanceError(
            "line {}: a {} by {} {} array is {} value{}, but {} data line{} "
            "follow{}".format(
                number, rows, columns, header.symmetry, len(places),
                "" if len(places) == 1 else "s", len(data),
                "" if len(data) == 1 else "s", "s" if len(data) == 1 else "",
            )
        )

    entries = _Entries(header)
    for (line_number, tokens), (row, column) in zip(data, places):
        if len(tokens) != 1:
            raise InstanceError(
                "line {}: an array file writes one value per line, found {}: "
                "{!r}".format(line_number, len(tokens), " ".join(tokens)[:60])
            )
        value = _value(line_number, tokens[0], header.field)
        if value == 0.0:
            continue  # a dense file writes every position; a zero is not an entry
        entries.add(line_number, row, column, value)
    return rows, columns, entries.rows


def _square(header: _Header, number: int, rows: int, columns: int) -> None:
    if header.mirrored and rows != columns:
        raise InstanceError(
            "line {}: a {} matrix is square, but the size line says {} by "
            "{}".format(number, header.symmetry, rows, columns)
        )


# ---------------------------------------------------------------------------
# what the rest of the library calls
# ---------------------------------------------------------------------------

def parse(text: str, name: str = "", source: str = "") -> Matrix:
    """Read Matrix Market text into a :class:`Matrix`.

    Both triangles of a symmetric or skew-symmetric file are materialised, and
    :attr:`Matrix.symmetric` records only whether the mirrored entry equals the
    stated one -- a skew-symmetric file arrives with both triangles present and
    that flag false, because it is not symmetric.
    """
    header, body = _scan(text)
    if header.layout == "coordinate":
        rows, columns, entries = _coordinate(header, body)
    else:
        rows, columns, entries = _array(header, body)
    return Matrix(
        name=name,
        source=source,
        layout=LAYOUT,
        rows=rows,
        columns=columns,
        entries=tuple(entries),
        symmetric=header.symmetric,
    )


def read(path: Union[str, Path]) -> Matrix:
    """Read a Matrix Market file, named after the file it came from."""
    location = Path(path).expanduser()
    try:
        text = location.read_text(errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))
    return parse(text, name=location.stem, source=str(location))


def parse_vector(text: str, name: str = "", source: str = "") -> Tuple[float, ...]:
    """Read a one-column Matrix Market file as a right-hand side.

    These ship beside the matrices, in either format.  A coordinate vector states
    its non-zeros only, so every position it omits is a zero.
    """
    matrix = parse(text, name=name, source=source)
    if matrix.columns != 1:
        raise InstanceError(
            "{}: a right-hand side has one column, this file has {}".format(
                source or name or "vector", matrix.columns
            )
        )
    values = [0.0] * matrix.rows
    for row, _column, value in matrix.entries:
        values[row] = value
    return tuple(values)


def read_vector(path: Union[str, Path]) -> Tuple[float, ...]:
    """Read a right-hand-side file."""
    location = Path(path).expanduser()
    try:
        text = location.read_text(errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))
    return parse_vector(text, name=location.stem, source=str(location))
