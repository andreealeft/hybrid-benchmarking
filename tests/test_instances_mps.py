"""Reading MPS.

MPS is the one instance format in this library that is genuinely disputed:
two solvers reading the same file can build two different models, all of which
solve. So these tests assert what the *format* says the file means -- the
coefficients written on the cards, the conventions on free rows, on the
objective constant, on a negative upper bound -- rather than what the parser
happens to do with it. Where the reader had to rule between defensible
readings, the test names the ruling.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hybrid_benchmarking.instances import InstanceError, LinearProgram, mps

FIXTURES = Path(__file__).parent / "fixtures" / "mps"

EVERY_FIXTURE = sorted(path.name for path in FIXTURES.glob("*.mps"))


def load(stem: str) -> LinearProgram:
    return mps.read(FIXTURES / (stem + ".mps"))


def coefficients(program: LinearProgram):
    """The constraint matrix keyed by the names the file uses, so a failure
    reads like the file rather than like a pair of indices."""
    return {
        (program.rows[row], program.columns[column]): value
        for row, column, value in program.matrix
    }


def by_column(program: LinearProgram, values):
    return dict(zip(program.columns, values))


def by_row(program: LinearProgram, values):
    return dict(zip(program.rows, values))


# ---------------------------------------------------------------------------


class TestTheWorkedExample:
    """TESTPROB, the model every piece of MPS documentation prints, asserted
    against the numbers written on its cards."""

    def test_the_model_is_named_by_its_NAME_line(self):
        assert load("testprob").name == "TESTPROB"
        assert load("testprob").layout == "mps"

    def test_the_first_N_row_is_the_objective_and_is_not_a_constraint(self):
        program = load("testprob")
        assert program.objective_name == "COST"
        assert program.rows == ("LIM1", "LIM2", "MYEQN")
        assert program.senses == ("L", "G", "E")

    def test_columns_keep_the_order_the_file_declares_them_in(self):
        assert load("testprob").columns == ("XONE", "YTWO", "ZTHREE")

    def test_the_objective_row_is_read_coefficient_by_coefficient(self):
        program = load("testprob")
        assert by_column(program, program.objective) == {
            "XONE": 1.0, "YTWO": 2.0, "ZTHREE": 3.0,
        }

    def test_the_constraint_matrix_is_what_the_cards_say(self):
        program = load("testprob")
        assert coefficients(program) == {
            ("LIM1", "XONE"): 1.0,
            ("LIM2", "XONE"): 1.0,
            ("LIM1", "YTWO"): 1.0,
            ("MYEQN", "YTWO"): -1.0,
            ("LIM2", "ZTHREE"): 1.0,
            ("MYEQN", "ZTHREE"): 1.0,
        }

    def test_the_objective_row_never_enters_the_matrix(self):
        program = load("testprob")
        assert all(row < len(program.rows) for row, _, _ in program.matrix)

    def test_the_right_hand_side_is_read_per_row(self):
        program = load("testprob")
        assert by_row(program, program.rhs) == {
            "LIM1": 4.0, "LIM2": 1.0, "MYEQN": 7.0,
        }

    def test_bounds_are_read_and_the_unbounded_side_is_None(self):
        """UP 4 on XONE, LO -1 on YTWO, and nothing at all on ZTHREE, which
        therefore keeps the format's defaults of zero and infinity."""
        program = load("testprob")
        assert by_column(program, program.lower) == {
            "XONE": 0.0, "YTWO": -1.0, "ZTHREE": 0.0,
        }
        assert by_column(program, program.upper) == {
            "XONE": 4.0, "YTWO": None, "ZTHREE": None,
        }

    def test_nothing_the_file_does_not_say_is_invented(self):
        program = load("testprob")
        assert program.maximise is False
        assert program.objective_constant == 0.0
        assert program.ranges == (None, None, None)
        assert program.integer == (False, False, False)

    def test_the_file_stem_is_only_a_fallback_for_the_name(self):
        assert mps.parse("ROWS\n N COST\nENDATA\n", name="fallback").name \
            == "fallback"
        assert mps.parse("NAME STATED\nROWS\n N COST\nENDATA\n",
                         name="fallback").name == "STATED"

    def test_a_NAME_line_with_no_name_is_legal(self):
        assert mps.parse("NAME\nROWS\n N COST\nENDATA\n", name="stem").name \
            == "stem"


class TestTheTwoDialectsAgree:
    """Fixed and free MPS differ only in how the fields are found. Nothing
    downstream should be able to tell which one it was given."""

    def test_the_same_model_written_free_is_the_same_model(self):
        fixed = load("testprob")
        free = load("testprob-free")
        assert dataclasses.replace(free, source=fixed.source) == fixed

    def test_grouping_row_value_pairs_differently_changes_nothing(self):
        """One pair per line or two, in either order: the file states a set of
        coefficients, not a sequence of them."""
        assert coefficients(load("testprob-free")) == coefficients(
            load("testprob"))

    def test_a_field_beyond_the_fixed_card_columns_is_still_read(self):
        """Free format has no card columns at all, so a long name that would
        overrun field 3 of a fixed card must still be read."""
        program = mps.parse(
            "NAME LONG\n"
            "ROWS\n N COST\n L A_VERY_LONG_ROW_NAME_INDEED\n"
            "COLUMNS\n"
            " A_VERY_LONG_COLUMN_NAME COST 1.0 A_VERY_LONG_ROW_NAME_INDEED 2.0\n"
            "ENDATA\n"
        )
        assert program.columns == ("A_VERY_LONG_COLUMN_NAME",)
        assert program.matrix == ((0, 0, 2.0),)


class TestObjectiveSense:
    def test_minimising_is_the_default_because_the_format_says_so(self):
        assert load("testprob").maximise is False

    def test_OBJSENSE_on_its_own_section_line_is_read(self):
        assert load("objsense-max").maximise is True

    def test_OBJSENSE_on_the_same_line_is_read(self):
        assert load("objsense-max-free").maximise is True

    def test_the_two_spellings_state_the_same_model(self):
        own_line = load("objsense-max")
        same_line = load("objsense-max-free")
        assert dataclasses.replace(same_line, source=own_line.source) \
            == own_line

    def test_the_sense_is_a_flag_and_does_not_negate_the_objective(self):
        """A maximisation is not a minimisation of the negation until someone
        downstream decides to make it one; the file states + 7 and + 5."""
        program = load("objsense-max")
        assert by_column(program, program.objective) == {
            "TABLE": 7.0, "CHAIR": 5.0,
        }

    def test_an_unreadable_sense_is_refused(self):
        with pytest.raises(InstanceError, match="OBJSENSE"):
            mps.parse("OBJSENSE SIDEWAYS\nROWS\n N COST\nENDATA\n")


class TestFreeRows:
    """The second and later N rows are free rows. Every solver ignores them;
    the danger is that their coefficients leak into the objective, which
    produces a model that still solves and is not the one on the cards."""

    def test_the_first_N_row_is_the_objective(self):
        assert load("free-row").objective_name == "COST"

    def test_a_free_row_is_not_a_constraint(self):
        assert load("free-row").rows == ("CAP",)

    def test_a_free_row_s_coefficients_do_not_reach_the_objective(self):
        program = load("free-row")
        assert by_column(program, program.objective) == {"XA": 1.0, "XB": 0.0}

    def test_a_free_row_s_coefficients_do_not_reach_the_matrix(self):
        program = load("free-row")
        assert coefficients(program) == {("CAP", "XA"): 1.0, ("CAP", "XB"): 1.0}

    def test_a_right_hand_side_on_a_free_row_is_dropped_too(self):
        program = load("free-row")
        assert program.rhs == (9.0,)
        assert program.objective_constant == 0.0


class TestTheObjectiveConstant:
    """An RHS entry on the objective row is the constant, negated. This is
    unanimous among solvers and surprising to everyone else."""

    def test_an_RHS_on_the_objective_row_is_negated(self):
        program = mps.parse(
            "NAME K\nROWS\n N COST\n L CAP\n"
            "COLUMNS\n XA COST 1.0 CAP 1.0\n"
            "RHS\n RHS CAP 5.0 COST -10.0\n"
            "ENDATA\n"
        )
        assert program.objective_constant == 10.0
        assert program.rhs == (5.0,)

    def test_no_RHS_on_the_objective_means_no_constant(self):
        assert load("testprob").objective_constant == 0.0


class TestRanges:
    """A RANGES entry turns one row into two bounds, and how it does so
    depends on the row's sense and, for an E row, on the sign. That reading
    belongs downstream, so what is kept here is the number itself."""

    def test_the_raw_value_is_kept_per_row(self):
        program = load("ranges")
        assert by_row(program, program.ranges) == {
            "RLESS": 4.0, "RMORE": 6.0, "REQUAL": -3.0, "RPLAIN": None,
        }

    def test_a_negative_range_on_an_equality_row_keeps_its_sign(self):
        program = load("ranges")
        assert program.ranges[program.rows.index("REQUAL")] == -3.0

    def test_a_range_does_not_touch_the_right_hand_side(self):
        program = load("ranges")
        assert by_row(program, program.rhs) == {
            "RLESS": 10.0, "RMORE": 2.0, "REQUAL": 5.0, "RPLAIN": 8.0,
        }

    def test_a_range_does_not_add_a_row(self):
        program = load("ranges")
        assert program.rows == ("RLESS", "RMORE", "REQUAL", "RPLAIN")
        assert program.senses == ("L", "G", "E", "L")


class TestTheNegativeUpperBound:
    """The disputed one. UP with a negative value on a column whose lower
    bound is still the default zero: CPLEX, GLPK and Gurobi release the lower
    bound to minus infinity, on the grounds that the modeller cannot have meant
    an empty interval. A minority leave the zero and the model is infeasible.
    This reader follows the majority, and the rule turns on whether a lower
    bound has been stated *by that point in the file*."""

    def test_a_negative_UP_releases_an_unstated_lower_bound(self):
        program = load("negative-upper")
        assert by_column(program, program.lower)["XFREE"] == float("-inf")
        assert by_column(program, program.upper)["XFREE"] == -3.0

    def test_a_lower_bound_written_first_survives_the_negative_UP(self):
        program = load("negative-upper")
        assert by_column(program, program.lower)["XZERO"] == 0.0
        assert by_column(program, program.upper)["XZERO"] == -2.0

    def test_a_positive_UP_leaves_the_default_lower_bound_alone(self):
        program = load("negative-upper")
        assert by_column(program, program.lower)["XPLAIN"] == 0.0
        assert by_column(program, program.upper)["XPLAIN"] == 5.0

    def test_a_lower_bound_written_afterwards_wins(self):
        """Bounds are applied in file order, so the later card overrides."""
        program = mps.parse(
            "ROWS\n N COST\nCOLUMNS\n XA COST 1.0\n"
            "BOUNDS\n UP BND XA -3.0\n LO BND XA -9.0\n"
            "ENDATA\n"
        )
        assert program.lower == (-9.0,)


class TestBounds:
    def test_each_bound_type_means_what_the_format_says(self):
        program = mps.parse(
            "ROWS\n N COST\n"
            "COLUMNS\n"
            " XUP COST 1.0\n XLO COST 1.0\n XFX COST 1.0\n XFR COST 1.0\n"
            " XMI COST 1.0\n XPL COST 1.0\n XBV COST 1.0\n XLI COST 1.0\n"
            " XUI COST 1.0\n"
            "BOUNDS\n"
            " UP BND XUP 4.0\n LO BND XLO 2.0\n FX BND XFX 3.0\n"
            " FR BND XFR\n MI BND XMI\n PL BND XPL\n BV BND XBV\n"
            " LI BND XLI 1.0\n UI BND XUI 6.0\n"
            "ENDATA\n"
        )
        low = by_column(program, program.lower)
        high = by_column(program, program.upper)
        assert (low["XUP"], high["XUP"]) == (0.0, 4.0)
        assert (low["XLO"], high["XLO"]) == (2.0, None)
        assert (low["XFX"], high["XFX"]) == (3.0, 3.0)
        assert (low["XFR"], high["XFR"]) == (float("-inf"), None)
        assert (low["XMI"], high["XMI"]) == (float("-inf"), None)
        assert (low["XPL"], high["XPL"]) == (0.0, None)
        assert (low["XBV"], high["XBV"]) == (0.0, 1.0)
        assert (low["XLI"], high["XLI"]) == (1.0, None)
        assert (low["XUI"], high["XUI"]) == (0.0, 6.0)

    def test_MI_leaves_the_upper_bound_alone(self):
        """An older reading forced it to zero. Nothing current does, and it
        would silently shrink the feasible set."""
        program = mps.parse(
            "ROWS\n N COST\nCOLUMNS\n XA COST 1.0\n"
            "BOUNDS\n UP BND XA 5.0\n MI BND XA\n"
            "ENDATA\n"
        )
        assert program.lower == (float("-inf"),)
        assert program.upper == (5.0,)

    def test_the_integer_bound_types_declare_integrality(self):
        """BV, LI and UI say the column is integer just as a marker does."""
        program = mps.parse(
            "ROWS\n N COST\n"
            "COLUMNS\n XBV COST 1.0\n XLI COST 1.0\n XUI COST 1.0\n"
            " XPL COST 1.0\n"
            "BOUNDS\n BV BND XBV\n LI BND XLI 0.0\n UI BND XUI 3.0\n"
            " PL BND XPL\n"
            "ENDATA\n"
        )
        assert program.integer == (True, True, True, False)

    def test_a_bound_of_1e30_is_infinity_and_not_a_number(self):
        """1e30 is the LP world's spelling of infinity; taking it literally
        would bound a column that the file leaves unbounded."""
        program = mps.parse(
            "ROWS\n N COST\nCOLUMNS\n XA COST 1.0\n"
            "BOUNDS\n UP BND XA 1e30\n"
            "ENDATA\n"
        )
        assert program.upper == (None,)

    def test_the_bound_set_name_may_be_left_out(self):
        program = mps.parse(
            "ROWS\n N COST\nCOLUMNS\n XA COST 1.0\n"
            "BOUNDS\n UP XA 4.0\n"
            "ENDATA\n"
        )
        assert program.upper == (4.0,)


class TestIntegerMarkers:
    """INTORG and INTEND bracket the integer columns. The relaxation is what
    gets solved, so the flag is only there for the log to say the model was an
    integer program."""

    def test_columns_inside_the_markers_are_integer(self):
        program = load("integer")
        assert by_column(program, program.integer) == {
            "XCONT": False, "XINT": True, "YINT": True,
            "ZCONT": False, "WBIN": True,
        }

    def test_the_marker_lines_do_not_become_columns(self):
        assert "MARKER" not in load("integer").columns

    def test_INTEND_closes_the_block(self):
        program = load("integer")
        assert program.integer[program.columns.index("ZCONT")] is False

    def test_the_coefficients_are_read_across_the_markers(self):
        program = load("integer")
        assert by_column(program, program.objective) == {
            "XCONT": 1.0, "XINT": 2.0, "YINT": 3.0, "ZCONT": 4.0, "WBIN": 5.0,
        }
        assert coefficients(program) == {
            ("CAP", "XCONT"): 1.0, ("CAP", "XINT"): 1.0,
            ("CAP", "YINT"): 2.0, ("CAP", "ZCONT"): 1.0, ("CAP", "WBIN"): 1.0,
        }

    def test_the_program_says_it_was_relaxed(self):
        assert "integer program" in load("integer").describe()


class TestNothingIsShort:
    """The contract is that every per-column and per-row tuple comes back the
    length of columns and rows respectively, defaults filled in. A short tuple
    would read as a missing bound somewhere downstream, silently."""

    @pytest.mark.parametrize("filename", EVERY_FIXTURE)
    def test_every_fixture_comes_back_fully_populated(self, filename):
        program = mps.read(FIXTURES / filename)
        columns = len(program.columns)
        rows = len(program.rows)
        assert len(program.objective) == columns
        assert len(program.lower) == columns
        assert len(program.upper) == columns
        assert len(program.integer) == columns
        assert len(program.senses) == rows
        assert len(program.rhs) == rows
        assert len(program.ranges) == rows

    @pytest.mark.parametrize("filename", EVERY_FIXTURE)
    def test_every_matrix_entry_points_at_a_row_and_a_column(self, filename):
        program = mps.read(FIXTURES / filename)
        for row, column, _ in program.matrix:
            assert 0 <= row < len(program.rows)
            assert 0 <= column < len(program.columns)

    @pytest.mark.parametrize("filename", EVERY_FIXTURE)
    def test_every_row_carries_a_sense_the_format_defines(self, filename):
        program = mps.read(FIXTURES / filename)
        assert set(program.senses) <= {"L", "G", "E"}


class TestComments:
    def test_a_star_in_the_first_column_is_a_comment(self):
        program = mps.parse(
            "* this is a comment\n"
            "NAME K\n* and so is this\nROWS\n N COST\n L CAP\n"
            "COLUMNS\n XA COST 1.0 CAP 1.0\n* even here\nENDATA\n"
        )
        assert program.columns == ("XA",)
        assert program.rows == ("CAP",)

    def test_blank_lines_are_legal(self):
        program = mps.parse(
            "NAME K\n\nROWS\n\n N COST\n L CAP\n\nCOLUMNS\n"
            " XA COST 1.0 CAP 1.0\n\nENDATA\n"
        )
        assert program.rows == ("CAP",)

    def test_data_after_ENDATA_is_not_read(self):
        program = mps.parse(
            "ROWS\n N COST\n L CAP\nCOLUMNS\n XA COST 1.0 CAP 1.0\n"
            "ENDATA\nCOLUMNS\n XB COST 9.0\n"
        )
        assert program.columns == ("XA",)


class TestItRefusesRatherThanGuesses:
    """Every one of these produces a plausible model if guessed at, which is
    the failure this package exists to prevent. Each message has to name the
    line and what was wrong, because the person holding the file is the only
    one who can fix it."""

    def test_a_coefficient_on_an_undeclared_row(self):
        text = (
            "NAME BAD\n"          # 1
            "ROWS\n"              # 2
            " N COST\n"           # 3
            " L CAP\n"            # 4
            "COLUMNS\n"           # 5
            " XA COST 1.0\n"      # 6
            " XA NOSUCH 1.0\n"    # 7
            "ENDATA\n"            # 8
        )
        with pytest.raises(InstanceError, match="line 7") as raised:
            mps.parse(text)
        assert "NOSUCH" in str(raised.value)

    def test_a_right_hand_side_on_an_undeclared_row(self):
        text = (
            "ROWS\n"              # 1
            " N COST\n"           # 2
            " L CAP\n"            # 3
            "COLUMNS\n"           # 4
            " XA COST 1.0\n"      # 5
            "RHS\n"               # 6
            " RHS NOSUCH 4.0\n"   # 7
            "ENDATA\n"
        )
        with pytest.raises(InstanceError, match="line 7"):
            mps.parse(text)

    def test_a_bound_on_an_undeclared_column(self):
        text = (
            "NAME BAD\n"          # 1
            "ROWS\n"              # 2
            " N COST\n"           # 3
            " L CAP\n"            # 4
            "COLUMNS\n"           # 5
            " XA COST 1.0\n"      # 6
            " XA CAP 1.0\n"       # 7
            "BOUNDS\n"            # 8
            " UP BND NOSUCH 4.0\n"  # 9
            "ENDATA\n"
        )
        with pytest.raises(InstanceError, match="line 9") as raised:
            mps.parse(text)
        assert "NOSUCH" in str(raised.value)

    def test_a_section_keyword_we_do_not_know(self):
        """A quadratic objective section is a real thing that appears in real
        files, and reading the file without it gives a different model."""
        text = (
            "NAME BAD\n"          # 1
            "ROWS\n"              # 2
            " N COST\n"           # 3
            "QUADOBJ\n"           # 4
            " XA XA 2.0\n"
            "ENDATA\n"
        )
        with pytest.raises(InstanceError, match="line 4") as raised:
            mps.parse(text)
        assert "QUADOBJ" in str(raised.value)

    def test_no_N_row_at_all(self):
        text = (
            "NAME BAD\nROWS\n L CAP\n"
            "COLUMNS\n XA CAP 1.0\nENDATA\n"
        )
        with pytest.raises(InstanceError, match="no objective"):
            mps.parse(text)

    def test_a_row_sense_the_format_does_not_define(self):
        with pytest.raises(InstanceError, match="line 2"):
            mps.parse("ROWS\n Q CAP\n N COST\nENDATA\n")

    def test_a_coefficient_that_is_not_a_number(self):
        text = (
            "ROWS\n N COST\n L CAP\n"     # 1 2 3
            "COLUMNS\n"                    # 4
            " XA COST ONE\n"               # 5
            "ENDATA\n"
        )
        with pytest.raises(InstanceError, match="line 5") as raised:
            mps.parse(text)
        assert "not a number" in str(raised.value)

    def test_a_bound_type_the_format_does_not_define(self):
        text = (
            "ROWS\n N COST\n"              # 1 2
            "COLUMNS\n XA COST 1.0\n"      # 3 4
            "BOUNDS\n"                     # 5
            " ZZ BND XA 1.0\n"             # 6
            "ENDATA\n"
        )
        with pytest.raises(InstanceError, match="line 6") as raised:
            mps.parse(text)
        assert "ZZ" in str(raised.value)

    def test_a_bound_that_needs_a_value_and_has_none(self):
        with pytest.raises(InstanceError, match="line 6") as raised:
            mps.parse(
                "ROWS\n N COST\n"           # 1 2
                "COLUMNS\n XA COST 1.0\n"   # 3 4
                "BOUNDS\n"                  # 5
                " UP BND XA\n"              # 6 -- BND is the set, XA the column
                "ENDATA\n"
            )
        assert "no value" in str(raised.value)

    def test_a_COLUMNS_line_with_a_dangling_field(self):
        """A row named with no value beside it: the pair is incomplete, and
        assuming a zero would put a coefficient in the model that nobody
        wrote."""
        with pytest.raises(InstanceError, match="line 5") as raised:
            mps.parse(
                "ROWS\n N COST\n L CAP\n"   # 1 2 3
                "COLUMNS\n"                 # 4
                " XA COST 1.0 CAP\n"        # 5
                "ENDATA\n"
            )
        assert "pairs" in str(raised.value)

    def test_a_row_declared_twice(self):
        with pytest.raises(InstanceError, match="line 4"):
            mps.parse("ROWS\n N COST\n L CAP\n L CAP\nENDATA\n")

    def test_data_before_any_section(self):
        with pytest.raises(InstanceError, match="line 1"):
            mps.parse(" XA COST 1.0\nROWS\n N COST\nENDATA\n")

    def test_a_missing_file_says_so(self):
        with pytest.raises(InstanceError, match="cannot read"):
            mps.read("/nowhere/at/all.mps")

    def test_the_source_is_named_in_the_message(self):
        """Someone reading a batch of files needs to know which one failed."""
        with pytest.raises(InstanceError, match="batch/17.mps"):
            mps.parse("ROWS\n Q CAP\n N COST\nENDATA\n", source="batch/17.mps")
