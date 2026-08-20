"""Reading MPS, the format every solver accepts and none reads quite alike.

MPS is punched-card archaeology: a model is a sequence of named sections, and
inside them a fixed layout of card columns -- ``2-3``, ``5-12``, ``15-22``,
``25-36``, ``40-47``, ``50-61``.  Free MPS keeps the sections and abandons the
card columns, separating fields by whitespace instead.  Almost nobody honours
the card columns even for fixed files, because whitespace splitting reads every
file that column slicing reads, save one: a *name containing a space*.  This
reader splits on whitespace too, so **names with embedded spaces are not
supported** -- a fixed file that uses them will be misread, and there is no way
for us to notice.  Everything else is read identically in both dialects, which
is why there is one parser here and not two.

What comes out is an :class:`LinearProgram`, which is the model *as the file
states it*: rows keep their sense, columns keep their bounds, and nothing is
converted to ``Ax = b, x >= 0``.  That conversion invents columns, and doing it
in the reader would destroy the difference between a bound the modeller wrote
and a slack we added.

Where the format is disputed
----------------------------

Solvers genuinely disagree on the following, and every one of them has been
ruled here rather than left to chance:

* **Extra ``N`` rows.**  The first ``N`` row is the objective.  Further ``N``
  rows are free rows -- unconstrained expressions someone wanted evaluated.
  They are dropped, by universal convention, and their coefficients are dropped
  with them: they never reach :attr:`LinearProgram.objective`.
* **An ``RHS`` entry on the objective row.**  Every solver reads this as the
  *negated* objective constant, so a value of ``-10`` means the objective has
  ``+10`` added to it.  :attr:`LinearProgram.objective_constant` therefore
  holds the negation of what the file says.
* **A negative ``UP`` bound.**  ``UP`` states an upper bound.  If it is negative
  and no lower bound has yet been stated for that column, the default lower
  bound of zero would make the column infeasible on its own, which is plainly
  not what the modeller meant.  Most solvers -- CPLEX, GLPK, Gurobi -- silently
  drop the lower bound to minus infinity.  A minority leave it at zero and let
  the model be infeasible.  We follow the majority.  The rule reads the bounds
  *in file order*: a ``LO`` written before the ``UP`` protects the zero, one
  written after overrides the minus infinity.  That is also how the solvers
  behave, and it is why the order of ``BOUNDS`` lines is not cosmetic.
* **``MI``.**  Sets the lower bound to minus infinity and leaves the upper bound
  alone.  An older IBM reading also forced the upper bound to zero; nothing
  current does, and it silently changes the feasible set, so it is not done.
* **``BV``, ``LI``, ``UI``.**  These declare integrality just as an
  ``INTORG``/``INTEND`` marker does, so they set
  :attr:`LinearProgram.integer` as well as the bounds.
* **``1e30``.**  Used as a stand-in for infinity throughout the LP world.  A
  bound whose magnitude reaches ``1e30`` is read as infinite rather than as a
  very large number, since the latter would quietly bound an unbounded column.
  Only ``BOUNDS`` values are treated this way; a coefficient or a right-hand
  side of ``1e30`` is a number.
* **``RANGES``.**  A range turns one row into two bounds, and how it does so
  depends on the row's sense and on the *sign* of the value for ``E`` rows.
  That interpretation belongs where the model is put into standard form, not
  here, so :attr:`LinearProgram.ranges` holds the raw value per row.
* **A repeated coefficient.**  The last one written wins.  Some readers sum
  them, some refuse the file; last-wins is what the widely used solvers do.
* **An explicit zero coefficient.**  Kept.  The reader's job is to report what
  the file contains, and the number of non-zeros is a quantity this library
  logs; dropping zeros here would change a logged count without saying so.
* **A missing ``RHS``/``RANGES`` set name.**  The name is optional in practice.
  A line carries one or two row/value pairs, so an odd number of fields means
  the first is a set name and an even number means it is not.  (``COLUMNS`` is
  not disambiguated this way and accepts any number of pairs, because files
  every solver reads carry three.)  ``BOUNDS`` is
  disambiguated by looking the column name up instead, which is unambiguous
  because a bound on a column we never saw is an error either way.

Anything that cannot be read this way raises :class:`InstanceError` naming the
line, rather than being guessed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from . import InstanceError, LinearProgram

__all__ = ["parse", "read"]

#: Sections we know.  Anything else in column one is refused rather than
#: skipped: an unread section is a silently different model.
_SECTIONS = frozenset(
    ("NAME", "OBJSENSE", "OBJSENS", "ROWS", "COLUMNS", "RHS", "RANGES",
     "BOUNDS", "ENDATA")
)

#: How the objective sense may be spelt, British and American.
_SENSE_WORDS = {
    "MAX": True, "MAXIMIZE": True, "MAXIMISE": True,
    "MIN": False, "MINIMIZE": False, "MINIMISE": False,
}

_ROW_SENSES = frozenset(("N", "L", "G", "E"))

_BOUNDS_NEEDING_VALUE = frozenset(("UP", "LO", "FX", "LI", "UI"))
_BOUNDS_WITHOUT_VALUE = frozenset(("FR", "MI", "PL", "BV"))

#: The magnitude at which a bound stops being a number and becomes infinity.
_INFINITE = 1e30

_MINUS_INFINITY = float("-inf")


def parse(text: str, name: str = "", source: str = "") -> LinearProgram:
    """Read the text of an MPS file, fixed or free, into a linear program.

    The model's name is taken from the ``NAME`` line where there is one, and
    from *name* -- normally the file stem -- where there is not, since a
    surprising number of files in circulation have a bare ``NAME`` card or none
    at all.
    """
    return _Reader(text, name, source).run()


def read(path: Union[str, Path]) -> LinearProgram:
    """Read an MPS file from disk, falling back on its stem for the name."""
    location = Path(path).expanduser()
    try:
        text = location.read_text(errors="replace")
    except OSError as error:
        raise InstanceError("cannot read {}: {}".format(location, error))
    return parse(text, name=location.stem, source=str(location))


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

class _Reader:
    """One pass over the file, accumulating the model as it goes.

    Kept as a class only so the section handlers can share the half-built
    model without threading a dozen dictionaries through every call.
    """

    def __init__(self, text: str, name: str, source: str) -> None:
        self.text = text
        self.fallback_name = name
        self.source = source

        self.model_name = ""
        self.maximise = False

        self.objective_name = ""
        self.has_objective = False
        self.free_rows: Set[str] = set()
        self.row_index: Dict[str, int] = {}
        self.row_names: List[str] = []
        self.senses: List[str] = []

        self.column_index: Dict[str, int] = {}
        self.column_names: List[str] = []
        self.objective: List[float] = []
        self.lower: List[float] = []
        self.upper: List[Optional[float]] = []
        self.integer: List[bool] = []
        #: Whether a lower bound has been *stated* for the column, which is
        #: what the negative-``UP`` rule turns on.
        self.lower_stated: List[bool] = []

        self.entries: Dict[Tuple[int, int], float] = {}
        self.rhs: List[float] = []
        self.ranges: List[Optional[float]] = []
        self.objective_constant = 0.0

        self.marker_integer = False

    # -- machinery ---------------------------------------------------------

    def fail(self, number: int, detail: str) -> InstanceError:
        where = "{}: ".format(self.source) if self.source else ""
        return InstanceError("{}line {}: {}".format(where, number, detail))

    def number(self, token: str, line: int, what: str) -> float:
        try:
            return float(token)
        except ValueError:
            raise self.fail(line, "{} is {!r}, which is not a number".format(
                what, token))

    def run(self) -> LinearProgram:
        section = ""
        for number, raw in enumerate(self.text.splitlines(), 1):
            if not raw.strip():
                continue
            if raw.lstrip().startswith("*"):
                continue

            fields = _fields(raw)
            if not fields:
                continue  # the line held nothing but a `$` comment

            if not raw[0].isspace():
                keyword = fields[0].upper()
                if keyword not in _SECTIONS:
                    raise self.fail(number, "unknown section {!r}".format(
                        fields[0]))
                if keyword == "ENDATA":
                    break
                section = "OBJSENSE" if keyword == "OBJSENS" else keyword
                self.header(section, fields, number)
                continue

            if not section:
                raise self.fail(
                    number, "data before any section: {!r}".format(fields[0]))
            self.data(section, fields, number)

        if not self.has_objective:
            where = "{}: ".format(self.source) if self.source else ""
            raise InstanceError(
                "{}ROWS declares no N row, so the model has no objective"
                .format(where))

        return self.finish()

    def header(self, section: str, fields: List[str], number: int) -> None:
        """A section keyword, and whatever it carries on its own line."""
        if section == "NAME":
            if len(fields) > 1:
                self.model_name = fields[1]
        elif section == "OBJSENSE" and len(fields) > 1:
            self.set_sense(fields[1], number)
        elif section in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS"):
            if len(fields) > 1 and section != "RHS":
                # RHS is written `RHS` alone but occasionally carries the set
                # name; the others never carry anything.
                raise self.fail(number, "{} takes nothing on its line, found "
                                        "{!r}".format(section, fields[1]))

    def data(self, section: str, fields: List[str], number: int) -> None:
        if section == "NAME":
            raise self.fail(number, "the NAME section has no data lines")
        if section == "OBJSENSE":
            self.set_sense(fields[0], number)
        elif section == "ROWS":
            self.row(fields, number)
        elif section == "COLUMNS":
            self.column(fields, number)
        elif section == "RHS":
            self.right_hand_side(fields, number)
        elif section == "RANGES":
            self.range_(fields, number)
        elif section == "BOUNDS":
            self.bound(fields, number)

    # -- sections ----------------------------------------------------------

    def set_sense(self, word: str, number: int) -> None:
        try:
            self.maximise = _SENSE_WORDS[word.upper()]
        except KeyError:
            raise self.fail(number, "OBJSENSE is {!r}, expected MAX or MIN"
                                    .format(word))

    def row(self, fields: List[str], number: int) -> None:
        if len(fields) != 2:
            raise self.fail(number, "a ROWS line is a sense and a name, found "
                                    "{}".format(" ".join(fields)))
        sense, name = fields[0].upper(), fields[1]
        if sense not in _ROW_SENSES:
            raise self.fail(number, "row sense {!r} is not one of N, L, G, E"
                                    .format(fields[0]))
        if name == self.objective_name or name in self.free_rows \
                or name in self.row_index:
            raise self.fail(number, "row {!r} is declared twice".format(name))

        if sense == "N":
            if self.has_objective:
                # A second N row is a free row: an expression to be reported,
                # not optimised and not constrained.  Dropped, with everything
                # written into it.
                self.free_rows.add(name)
            else:
                self.has_objective = True
                self.objective_name = name
            return

        self.row_index[name] = len(self.row_names)
        self.row_names.append(name)
        self.senses.append(sense)
        self.rhs.append(0.0)
        self.ranges.append(None)

    def column(self, fields: List[str], number: int) -> None:
        if any(_unquote(field).upper() == "MARKER" for field in fields):
            self.marker(fields, number)
            return
        # Fixed-format MPS puts at most two pairs on a card, and this used to
        # refuse a third. Solvers do not: HiGHS reads them, and an instance
        # Binkowski's benchmark uses carries three. Refusing a file every
        # solver accepts is a reader bug, not strictness -- so any odd field
        # count is read as a column followed by that many pairs.
        if len(fields) < 3 or len(fields) % 2 == 0:
            raise self.fail(
                number, "a COLUMNS line is a column followed by row/value "
                        "pairs, found {} fields".format(len(fields)))

        column = self.column_for(fields[0])
        for row, token in zip(fields[1::2], fields[2::2]):
            value = self.number(token, number, "the coefficient of {} in {}"
                                               .format(fields[0], row))
            self.coefficient(row, column, value, number)

    def marker(self, fields: List[str], number: int) -> None:
        """``<name> 'MARKER' 'INTORG'`` opens an integer block, ``'INTEND'``
        closes it.  Columns declared inside are integer."""
        words = {_unquote(field).upper() for field in fields}
        if "INTORG" in words:
            self.marker_integer = True
        elif "INTEND" in words:
            self.marker_integer = False
        else:
            raise self.fail(number, "a MARKER line names neither INTORG nor "
                                    "INTEND")

    def coefficient(self, row: str, column: int, value: float,
                    number: int) -> None:
        if row == self.objective_name:
            self.objective[column] = value
        elif row in self.free_rows:
            return  # a free row: read and discarded, deliberately
        elif row in self.row_index:
            self.entries[(self.row_index[row], column)] = value
        else:
            raise self.fail(number, "coefficient on row {!r}, which ROWS does "
                                    "not declare".format(row))

    def right_hand_side(self, fields: List[str], number: int) -> None:
        for row, token in self.pairs(fields, number, "RHS"):
            value = self.number(token, number, "the RHS of {}".format(row))
            if row == self.objective_name:
                # Negated: the file states the constant moved to the other
                # side of the objective equation.
                self.objective_constant = -value
            elif row in self.free_rows:
                continue
            elif row in self.row_index:
                self.rhs[self.row_index[row]] = value
            else:
                raise self.fail(number, "RHS on row {!r}, which ROWS does not "
                                        "declare".format(row))

    def range_(self, fields: List[str], number: int) -> None:
        for row, token in self.pairs(fields, number, "RANGES"):
            value = self.number(token, number, "the range of {}".format(row))
            if row == self.objective_name or row in self.free_rows:
                continue  # a range on an unconstrained row means nothing
            if row not in self.row_index:
                raise self.fail(number, "range on row {!r}, which ROWS does "
                                        "not declare".format(row))
            self.ranges[self.row_index[row]] = value

    def pairs(self, fields: List[str], number: int,
              section: str) -> List[Tuple[str, str]]:
        """Row/value pairs from an RHS or RANGES line, set name or not.

        One or two pairs per line, optionally preceded by the name of the set
        they belong to.  Parity settles which: an odd count has the name, an
        even count does not.
        """
        if len(fields) in (3, 5):
            body = fields[1:]
        elif len(fields) in (2, 4):
            body = fields
        else:
            raise self.fail(
                number, "a {} line is one or two row/value pairs, optionally "
                        "after a set name, found {} fields".format(
                            section, len(fields)))
        return list(zip(body[0::2], body[1::2]))

    def bound(self, fields: List[str], number: int) -> None:
        kind = fields[0].upper()
        if kind not in _BOUNDS_NEEDING_VALUE and \
                kind not in _BOUNDS_WITHOUT_VALUE:
            raise self.fail(number, "bound type {!r} is not one of UP, LO, FX, "
                                    "FR, MI, PL, BV, LI, UI".format(fields[0]))

        rest = fields[1:]
        if not rest:
            raise self.fail(number, "a {} bound names no column".format(kind))
        # The set name is optional, so find the column by looking it up.  Its
        # usual place is second; where that is not a column, the writer left
        # the set name out and it is first.
        if len(rest) >= 2 and rest[1] in self.column_index:
            at = 1
        elif rest[0] in self.column_index:
            at = 0
        else:
            named = rest[1] if len(rest) >= 2 else rest[0]
            raise self.fail(number, "bound on column {!r}, which COLUMNS does "
                                    "not declare".format(named))
        column = self.column_index[rest[at]]

        token = rest[at + 1] if len(rest) > at + 1 else None
        if kind in _BOUNDS_NEEDING_VALUE:
            if token is None:
                raise self.fail(number, "a {} bound on {} states no value"
                                        .format(kind, rest[at]))
            value = self.number(token, number, "the {} bound on {}".format(
                kind, rest[at]))
        else:
            value = 0.0  # FR, MI, PL and BV carry a value only decoratively

        self.apply_bound(kind, column, value)

    def apply_bound(self, kind: str, column: int, value: float) -> None:
        if kind == "UP":
            self.upper[column] = None if value >= _INFINITE else value
            if value < 0.0 and not self.lower_stated[column]:
                # See the module docstring: an upper bound below the default
                # lower bound of zero is read as releasing that lower bound.
                self.lower[column] = _MINUS_INFINITY
        elif kind == "LO":
            self.lower[column] = (_MINUS_INFINITY if value <= -_INFINITE
                                  else value)
            self.lower_stated[column] = True
        elif kind == "FX":
            # The infinity convention applies here too, and it is the one place
            # it matters most: fixing a column *at* 1e30 is exactly the "quietly
            # bound an unbounded column" the rule exists to prevent, only worse,
            # because it pins the variable rather than merely capping it.
            if value >= _INFINITE:
                self.lower[column] = 0.0 if not self.lower_stated[column] \
                    else self.lower[column]
                self.upper[column] = None
            elif value <= -_INFINITE:
                self.lower[column] = _MINUS_INFINITY
                self.upper[column] = None
            else:
                self.lower[column] = value
                self.upper[column] = value
            self.lower_stated[column] = True
        elif kind == "FR":
            self.lower[column] = _MINUS_INFINITY
            self.upper[column] = None
            self.lower_stated[column] = True
        elif kind == "MI":
            self.lower[column] = _MINUS_INFINITY
            self.lower_stated[column] = True
        elif kind == "PL":
            self.upper[column] = None
        elif kind == "BV":
            self.lower[column] = 0.0
            self.upper[column] = 1.0
            self.lower_stated[column] = True
            self.integer[column] = True
        elif kind == "LI":
            self.lower[column] = (_MINUS_INFINITY if value <= -_INFINITE
                                  else value)
            self.lower_stated[column] = True
            self.integer[column] = True
        elif kind == "UI":
            self.upper[column] = None if value >= _INFINITE else value
            self.integer[column] = True

    # -- columns and the result -------------------------------------------

    def column_for(self, name: str) -> int:
        """The index of a column, declaring it with its defaults if new."""
        if name in self.column_index:
            return self.column_index[name]
        self.column_index[name] = len(self.column_names)
        self.column_names.append(name)
        self.objective.append(0.0)
        self.lower.append(0.0)
        self.upper.append(None)
        self.integer.append(self.marker_integer)
        self.lower_stated.append(False)
        return self.column_index[name]

    def finish(self) -> LinearProgram:
        matrix = tuple(
            (row, column, self.entries[(row, column)])
            for row, column in sorted(self.entries)
        )
        return LinearProgram(
            name=self.model_name or self.fallback_name,
            source=self.source,
            layout="mps",
            columns=tuple(self.column_names),
            rows=tuple(self.row_names),
            senses=tuple(self.senses),
            objective=tuple(self.objective),
            matrix=matrix,
            rhs=tuple(self.rhs),
            ranges=tuple(self.ranges),
            lower=tuple(self.lower),
            upper=tuple(self.upper),
            integer=tuple(self.integer),
            maximise=self.maximise,
            objective_name=self.objective_name,
            objective_constant=self.objective_constant,
        )


def _fields(line: str) -> List[str]:
    """The fields of a line, whitespace separated, comment removed.

    Fixed MPS lets a ``$`` in the field where a row name would go start a
    comment running to the end of the line.  Honouring that costs a name
    beginning with ``$``, which nothing writes; not honouring it costs the
    ability to read files that use it, which several generators do.
    """
    tokens = line.split()
    for position, token in enumerate(tokens):
        if token.startswith("$"):
            return tokens[:position]
    return tokens


def _unquote(token: str) -> str:
    """A field with the single quotes MARKER lines wear taken off."""
    return token.strip("'\"")
