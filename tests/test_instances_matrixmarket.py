"""Matrix Market files, and what the format claims about them.

The claims worth testing are the ones a reader can satisfy plausibly and
wrongly.  A symmetric file states one triangle; a reader that keeps only what is
written produces a matrix with half the non-zeros, the wrong row counts and a
condition number nobody can tell is wrong by looking at it.  An array file
writes column by column, and a reader that assumes rows transposes every
rectangular matrix it is given.  Both are asserted here against files, not
against what the code does with them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_benchmarking.instances import InstanceError, Matrix, detect
from hybrid_benchmarking.instances import matrixmarket as mm
from hybrid_benchmarking.instances import read as read_instance

FIXTURES = Path(__file__).parent / "fixtures" / "matrixmarket"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


SKEW = """\
%%MatrixMarket matrix coordinate real skew-symmetric
3 3 2
2 1 1.5
3 1 -2.0
"""

HERMITIAN = """\
%%MatrixMarket matrix coordinate real hermitian
2 2 3
1 1 2.0
2 1 -1.0
2 2 3.0
"""

SYMMETRIC_ARRAY = """\
%%MatrixMarket matrix array real symmetric
3 3
1.0
2.0
3.0
4.0
5.0
6.0
"""


class TestTheExampleFromTheSpecification:
    """The 5 by 5 matrix on the Matrix Market format page, entry by entry."""

    def matrix(self) -> Matrix:
        return mm.parse(fixture("spec-example.mtx"), name="spec-example")

    def test_the_shape_is_the_one_the_size_line_states(self):
        matrix = self.matrix()
        assert (matrix.rows, matrix.columns) == (5, 5)
        assert len(matrix.entries) == 8

    def test_every_entry_is_the_one_the_file_wrote(self):
        assert self.matrix().entries == (
            (0, 0, 1.0),
            (1, 1, 10.5),
            (2, 2, 0.015),
            (0, 3, 6.0),
            (3, 1, 250.5),
            (3, 3, -280.0),
            (3, 4, 33.32),
            (4, 4, 12.0),
        )

    def test_indices_written_from_one_are_held_from_zero(self):
        """The file's A(1,1) is entry (0, 0), and nothing reaches index 5."""
        matrix = self.matrix()
        assert (0, 0, 1.0) in matrix.entries
        assert max(row for row, _column, _value in matrix.entries) == 4
        assert max(column for _row, column, _value in matrix.entries) == 4

    def test_a_general_file_is_not_marked_symmetric(self):
        """It is not: A(1,4) is 6 and A(4,1) is absent."""
        matrix = self.matrix()
        assert matrix.symmetric is False
        assert (3, 0, 6.0) not in matrix.entries

    def test_the_comment_block_contributes_nothing(self):
        """Twenty lines of comment, including one quoting the banner."""
        text = fixture("spec-example.mtx")
        assert text.count("%") > 20
        assert len(self.matrix().entries) == 8

    def test_a_file_is_named_after_itself_and_cites_where_it_came_from(self):
        matrix = mm.read(FIXTURES / "spec-example.mtx")
        assert matrix.name == "spec-example"
        assert matrix.source.endswith("spec-example.mtx")
        assert matrix.layout == "matrix-market"

    def test_the_package_routes_an_mtx_file_here(self):
        assert detect(FIXTURES / "spec-example.mtx") == "matrix-market"
        assert isinstance(read_instance(FIXTURES / "spec-example.mtx"), Matrix)


class TestSymmetryIsMaterialised:
    """One triangle in the file, two in memory: the header is a statement about
    the matrix, not about how much of it to keep."""

    def test_one_triangle_and_two_describe_the_same_matrix(self):
        stated = mm.parse(fixture("triangle-symmetric.mtx"))
        written_out = mm.parse(fixture("triangle-general.mtx"))
        assert sorted(stated.entries) == sorted(written_out.entries)
        assert len(stated.entries) == 10

    def test_only_one_of_them_knows_it_is_symmetric(self):
        """The flag records the header's claim; the entries are the same either
        way, so nothing downstream depends on having read it."""
        assert mm.parse(fixture("triangle-symmetric.mtx")).symmetric is True
        assert mm.parse(fixture("triangle-general.mtx")).symmetric is False

    def test_the_diagonal_is_stated_once_and_kept_once(self):
        entries = mm.parse(fixture("triangle-symmetric.mtx")).entries
        diagonal = [e for e in entries if e[0] == e[1]]
        assert len(diagonal) == 4
        assert all(value == 4.0 for _row, _column, value in diagonal)

    def test_a_skew_symmetric_mirror_is_negated(self):
        matrix = mm.parse(SKEW)
        assert sorted(matrix.entries) == [
            (0, 1, -1.5), (0, 2, 2.0), (1, 0, 1.5), (2, 0, -2.0),
        ]

    def test_a_skew_symmetric_matrix_is_not_a_symmetric_one(self):
        assert mm.parse(SKEW).symmetric is False

    def test_a_skew_symmetric_diagonal_entry_is_refused(self):
        """Its diagonal is zero by definition, so a file stating one is not a
        skew-symmetric file and we do not get to choose which half it meant."""
        text = "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 2 1.0\n"
        with pytest.raises(InstanceError, match="line 3.*diagonal"):
            mm.parse(text)

    def test_hermitian_over_a_real_field_is_symmetric(self):
        matrix = mm.parse(HERMITIAN)
        assert matrix.symmetric is True
        assert (0, 1, -1.0) in matrix.entries and (1, 0, -1.0) in matrix.entries

    def test_stating_both_triangles_is_refused_and_names_both_lines(self):
        """The failure this catches is a matrix arriving with twice the
        off-diagonal non-zeros it has, which nothing later can detect."""
        text = (
            "%%MatrixMarket matrix coordinate real symmetric\n"
            "2 2 2\n"
            "2 1 1.0\n"
            "1 2 1.0\n"
        )
        with pytest.raises(InstanceError) as raised:
            mm.parse(text)
        assert "line 4" in str(raised.value) and "line 3" in str(raised.value)

    def test_a_symmetric_matrix_has_to_be_square(self):
        text = "%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 1.0\n"
        with pytest.raises(InstanceError, match="line 2.*square"):
            mm.parse(text)

    def test_the_banner_is_read_whatever_its_case(self):
        text = "%%MATRIXMARKET Matrix Coordinate Real Symmetric\n2 2 1\n2 1 3.0\n"
        matrix = mm.parse(text)
        assert matrix.symmetric is True
        assert sorted(matrix.entries) == [(0, 1, 3.0), (1, 0, 3.0)]


class TestTheArrayFormat:
    """Dense files carry no indices, so position is the only thing saying where
    a value belongs, and the specification says column by column."""

    def test_values_are_read_column_by_column(self):
        matrix = mm.parse(fixture("dense-array.mtx"))
        assert (matrix.rows, matrix.columns) == (3, 2)
        assert (2, 0, 3.0) in matrix.entries
        assert (0, 1, -2.5) in matrix.entries

    def test_reading_it_row_by_row_would_transpose_it(self):
        """Guards the ordering with a rectangular matrix, where the wrong
        reading cannot produce a valid matrix of the declared shape."""
        entries = dict(((row, column), value)
                       for row, column, value in mm.parse(
                           fixture("dense-array.mtx")).entries)
        assert entries[(2, 0)] == 3.0
        assert (0, 2) not in entries

    def test_a_written_zero_is_not_an_entry(self):
        """Every position of a dense file is written; only some hold anything."""
        assert len(mm.parse(fixture("dense-array.mtx")).entries) == 4

    def test_a_symmetric_array_writes_the_lower_triangle_only(self):
        matrix = mm.parse(SYMMETRIC_ARRAY)
        assert len(matrix.entries) == 9
        assert dict(((r, c), v) for r, c, v in matrix.entries)[(0, 2)] == 3.0
        assert dict(((r, c), v) for r, c, v in matrix.entries)[(1, 2)] == 5.0

    def test_too_few_values_for_the_declared_shape_is_refused(self):
        text = "%%MatrixMarket matrix array real general\n3 2\n1.0\n2.0\n"
        with pytest.raises(InstanceError, match="line 2.*6 values.*2 data lines"):
            mm.parse(text)

    def test_an_array_has_no_pattern_field(self):
        """Pattern states which entries exist; a dense file lists all of them."""
        with pytest.raises(InstanceError, match="line 1.*pattern"):
            mm.parse("%%MatrixMarket matrix array pattern general\n2 2\n")


class TestTheFields:
    def test_a_pattern_file_states_positions_and_no_values(self):
        matrix = mm.parse(fixture("pattern.mtx"))
        assert [(row, column) for row, column, _value in matrix.entries] == [
            (0, 0), (1, 2), (2, 0), (2, 2)
        ]
        assert all(value == 1.0 for _row, _column, value in matrix.entries)

    def test_an_integer_field_is_read(self):
        text = "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 7\n"
        assert mm.parse(text).entries == ((0, 0, 7.0),)

    def test_a_fraction_in_an_integer_file_is_refused(self):
        text = "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 7.5\n"
        with pytest.raises(InstanceError, match="line 3.*integer"):
            mm.parse(text)

    def test_a_complex_file_is_refused_rather_than_halved(self):
        """Its real part is a different matrix, and one that looks fine."""
        with pytest.raises(InstanceError, match="complex"):
            mm.parse(fixture("complex.mtx"))

    def test_the_refusal_comes_before_any_entry_is_read(self):
        """A complex file whose entries are otherwise sound still fails, so the
        message is about the field and not about a stray column."""
        with pytest.raises(InstanceError, match="line 1"):
            mm.parse(fixture("complex.mtx"))


class TestWhatIsMalformedNamesItsLine:
    def test_a_file_without_a_banner_is_not_a_matrix_market_file(self):
        with pytest.raises(InstanceError, match="line 1.*%%MatrixMarket"):
            mm.parse("5 5 1\n1 1 1.0\n")

    def test_a_banner_missing_its_symmetry_word_is_refused(self):
        """There is no default: general and symmetric differ by every mirrored
        entry, so a reader guessing here invents or loses half a matrix."""
        with pytest.raises(InstanceError, match="line 1"):
            mm.parse("%%MatrixMarket matrix coordinate real\n2 2 1\n1 1 1.0\n")

    def test_an_empty_file_says_so(self):
        with pytest.raises(InstanceError, match="empty"):
            mm.parse("\n\n")

    def test_a_declared_count_that_disagrees_with_the_lines_is_refused(self):
        text = (
            "%%MatrixMarket matrix coordinate real general\n"
            "3 3 4\n"
            "1 1 1.0\n"
            "2 2 2.0\n"
        )
        with pytest.raises(InstanceError, match="line 2.*4 entries.*2 data lines"):
            mm.parse(text)

    def test_an_index_outside_the_declared_shape_is_refused(self):
        text = "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1.0\n"
        with pytest.raises(InstanceError, match="line 3.*row index 3.*1..2"):
            mm.parse(text)

    def test_a_zero_index_is_refused_rather_than_read_as_zero_based(self):
        text = "%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 1.0\n"
        with pytest.raises(InstanceError, match="line 3"):
            mm.parse(text)

    def test_a_duplicate_position_names_both_lines(self):
        text = (
            "%%MatrixMarket matrix coordinate real general\n"
            "2 2 2\n"
            "1 1 1.0\n"
            "1 1 2.0\n"
        )
        with pytest.raises(InstanceError) as raised:
            mm.parse(text)
        assert "line 4" in str(raised.value) and "line 3" in str(raised.value)

    def test_a_missing_value_column_is_refused(self):
        text = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1\n"
        with pytest.raises(InstanceError, match="line 3"):
            mm.parse(text)

    def test_a_comment_below_the_size_line_is_tolerated(self):
        """The specification puts comments above it.  Files do not."""
        text = (
            "%%MatrixMarket matrix coordinate real general\n"
            "2 2 2\n"
            "1 1 1.0\n"
            "% written by something that did not read the specification\n"
            "2 2 2.0\n"
        )
        assert len(mm.parse(text).entries) == 2


class TestRightHandSides:
    """The vectors that ship beside the matrices, in whichever format."""

    def test_both_formats_give_the_same_vector(self):
        assert mm.parse_vector(fixture("rhs-array.mtx")) == (1.0, 0.0, -2.0, 3.5)
        assert mm.parse_vector(fixture("rhs-coordinate.mtx")) == (
            1.0, 0.0, -2.0, 3.5
        )

    def test_a_position_a_coordinate_vector_omits_is_a_zero(self):
        """Not a shorter vector: the size line says how long it is."""
        text = "%%MatrixMarket matrix coordinate real general\n5 1 1\n4 1 9.0\n"
        assert mm.parse_vector(text) == (0.0, 0.0, 0.0, 9.0, 0.0)

    def test_the_length_is_the_declared_row_count(self):
        assert len(mm.parse_vector(fixture("rhs-coordinate.mtx"))) == 4

    def test_a_file_read_from_disk_is_the_same_vector(self):
        assert mm.read_vector(FIXTURES / "rhs-array.mtx") == (
            1.0, 0.0, -2.0, 3.5
        )

    def test_a_matrix_is_not_a_right_hand_side(self):
        with pytest.raises(InstanceError, match="one column"):
            mm.parse_vector(fixture("spec-example.mtx"))

    def test_a_single_row_is_not_a_right_hand_side_either(self):
        """A 1 by 4 file is a row, and reading it as a column would silently
        transpose the system it belongs to."""
        text = "%%MatrixMarket matrix array real general\n1 4\n1.0\n2.0\n3.0\n4.0\n"
        with pytest.raises(InstanceError, match="one column"):
            mm.parse_vector(text)
