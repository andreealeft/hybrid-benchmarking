"""Reading multidimensional knapsack files: the two layouts, and their claims.

The claims belong to the files, not to the reader.  OR-Library states an
optimum for its ``mknap1`` and ``mknap2`` problems, so the optimum a brute
force finds in what we read back has to be the number the file states -- and it
is the check that matters here, because the two layouts differ in the order of
their first two numbers and in where the optimum sits, so a file read with the
wrong one of them comes back as a perfectly plausible knapsack with every value
in the wrong place.  Every fixture with more items than dimensions is that way
on purpose: a transposed cost matrix cannot pass a shape assertion it does not
fit.

The real specimens are lifted verbatim from files fetched from
``https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/``: the first problem of
``mknap1.txt`` and the first two of ``mknapcb1.txt``, with only their leading
problem counts rewritten to match what was kept, and ``PB7.DAT`` out of
``mknap2.txt`` with its name marker and banner intact.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from hybrid_benchmarking import instances
from hybrid_benchmarking.instances import InstanceError, MultidimensionalKnapsack
from hybrid_benchmarking.instances.mdkp import (
    LAYOUT,
    parse,
    parse_all,
    read,
    read_all,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mdkp"

#: The hand-written instance, stated once here and once in the fixture.  Four
#: items and three dimensions, so the cost matrix is not square.
PROFITS = (7, 5, 9, 3)
WEIGHTS = ((2, 3, 5, 1), (4, 1, 2, 6), (1, 7, 3, 2))
CAPACITIES = (6, 8, 9)


def best_value(instance: MultidimensionalKnapsack) -> int:
    """The optimum, by trying every subset.  Only fixture-sized instances."""
    best = 0
    for pattern in itertools.product((0, 1), repeat=len(instance.profits)):
        fits = all(
            sum(w for chosen, w in zip(pattern, row) if chosen) <= capacity
            for row, capacity in zip(instance.weights, instance.capacities)
        )
        if fits:
            best = max(
                best,
                sum(p for chosen, p in zip(pattern, instance.profits) if chosen),
            )
    return best


class TestTheCountedLayout:
    """``mknap1`` and every ``mknapcb``: a problem count, then ``n m z``."""

    def test_every_number_of_the_hand_written_instance_is_read_as_written(self):
        knapsack = read(FIXTURES / "hand_written.mdkp")
        assert knapsack.profits == PROFITS
        assert knapsack.weights == WEIGHTS
        assert knapsack.capacities == CAPACITIES
        assert knapsack.optimum == 12
        assert knapsack.layout == LAYOUT

    def test_the_matrix_has_one_row_per_dimension_not_one_per_item(self):
        """``weights[i][m]`` is what item ``m`` costs in dimension ``i``.  The
        fixture has four items and three dimensions so that the transposed
        reading, which is the intuitive one for anybody who thinks of an item
        as carrying a vector of costs, cannot fit these assertions."""
        knapsack = read(FIXTURES / "hand_written.mdkp")
        assert len(knapsack.weights) == 3 == len(knapsack.capacities)
        assert all(len(row) == 4 == len(knapsack.profits)
                   for row in knapsack.weights)
        assert knapsack.weights[1][3] == 6  # item 4 costs 6 in dimension 2
        assert knapsack.weights[2][1] == 7  # item 2 costs 7 in dimension 3

    def test_each_capacity_belongs_to_the_row_of_the_same_index(self):
        """A capacity vector read in reverse leaves every value present and
        every constraint wrong, which no shape check would catch."""
        knapsack = read(FIXTURES / "hand_written.mdkp")
        assert list(zip(knapsack.weights, knapsack.capacities)) == [
            ((2, 3, 5, 1), 6), ((4, 1, 2, 6), 8), ((1, 7, 3, 2), 9)
        ]

    def test_the_stated_optimum_is_the_optimum_of_what_we_read(self):
        """The file's claim about its own numbers.  Reading the cost matrix
        transposed, or pairing the capacities with the wrong rows, leaves the
        claim standing and makes it false."""
        knapsack = read(FIXTURES / "hand_written.mdkp")
        assert best_value(knapsack) == knapsack.optimum == 12

    def test_the_values_run_across_line_boundaries(self):
        """The fixture breaks its cost matrix mid-row, as the real files do.
        A reader taking one dimension per line would find three rows of six,
        seven and four numbers where there are three of four."""
        text = (FIXTURES / "hand_written.mdkp").read_text()
        assert [len(line.split()) for line in text.strip().splitlines()] == [
            1, 3, 4, 6, 2, 4, 3
        ]
        assert read(FIXTURES / "hand_written.mdkp").weights == WEIGHTS

    def test_a_real_mknapcb_problem_is_read_end_to_end(self):
        first, second = read_all(FIXTURES / "mknapcb1_two_problems.txt")
        assert len(first.profits) == 100
        assert len(first.weights) == 5
        assert all(len(row) == 100 for row in first.weights)
        assert first.profits[:3] == (504, 803, 667)
        assert first.profits[-1] == 632
        assert first.weights[0][:3] == (42, 41, 523)
        assert first.weights[4][-1] == 635
        assert first.capacities == (11927, 13727, 11551, 13056, 13460)
        assert second.profits[:3] == (869, 1043, 829)
        assert second.capacities == (12841, 13172, 12088, 12269, 13839)

    def test_a_real_specimen_wraps_its_hundred_profits_seven_to_a_line(self):
        """``mknapcb1`` writes seven numbers to a line regardless of what they
        belong to, so the eighth profit sits on the second line of them and
        each cost row ends two numbers into a line."""
        first, _ = read_all(FIXTURES / "mknapcb1_two_problems.txt")
        assert (first.profits[6], first.profits[7]) == (811, 856)
        assert (first.weights[0][-1], first.weights[1][0]) == (298, 509)

    def test_the_zero_optimum_of_the_chu_beasley_files_means_not_stated(self):
        """All 270 problems in ``mknapcb1`` to ``mknapcb9`` carry a ``0``
        there.  Every profit is positive, so a genuine optimum of zero would
        say that no single item fits any budget; the placeholder becomes
        ``None`` rather than a claim nobody made."""
        for instance in read_all(FIXTURES / "mknapcb1_two_problems.txt"):
            assert instance.optimum is None

    def test_a_placeholder_optimum_is_not_confused_with_a_real_one(self):
        """The third problem of the multi-problem fixture states ``0`` and is
        worth 10, so a reader taking the placeholder literally would report an
        optimum that the instance itself contradicts."""
        third = read_all(FIXTURES / "three_problems.mdkp")[2]
        assert third.optimum is None
        assert best_value(third) == 10


class TestTheMknap2Layout:
    """``mknap2``: no problem count, the header reversed, the optimum last."""

    def test_every_number_of_the_hand_written_instance_is_read_as_written(self):
        knapsack = read(FIXTURES / "hand_written_mknap2.txt")
        assert knapsack.profits == (5, 7, 9)
        assert knapsack.weights == ((1, 2, 3), (4, 5, 6))
        assert knapsack.capacities == (10, 12)
        assert knapsack.optimum == 16
        assert best_value(knapsack) == 16
        assert knapsack.layout == LAYOUT

    def test_the_header_states_the_dimensions_before_the_items(self):
        """The opposite of the counted layout, and the reason the two cannot
        share a reader: ``2 3`` here is two dimensions and three items, where
        in ``mknapcb`` it would be two items and three dimensions.  The fixture
        is not square, so only one of the two readings fits."""
        knapsack = read(FIXTURES / "hand_written_mknap2.txt")
        assert len(knapsack.weights) == 2
        assert len(knapsack.profits) == 3

    def test_the_capacities_come_before_the_cost_matrix_here(self):
        """In the counted layout they come after it.  Both orders leave every
        number in the file present, and they disagree about all of them."""
        knapsack = read(FIXTURES / "hand_written_mknap2.txt")
        assert knapsack.capacities == (10, 12)
        assert knapsack.weights[0] == (1, 2, 3)

    def test_a_real_mknap2_problem_is_read_end_to_end(self):
        knapsack = read(FIXTURES / "mknap2_pb7.txt")
        assert len(knapsack.profits) == 37
        assert len(knapsack.weights) == 30
        assert all(len(row) == 37 for row in knapsack.weights)
        assert knapsack.profits[:3] == (47, 77, 110)
        assert knapsack.capacities[:3] == (5875, 4351, 5221)
        assert knapsack.capacities[-1] == 3373
        assert knapsack.weights[0][:3] == (785, 774, 818)
        assert knapsack.weights[-1][-1] == 844

    def test_the_optimum_of_a_real_specimen_is_the_one_stated_at_its_end(self):
        """``PB7``'s 1035, which sits after the cost matrix rather than in the
        header.  A reader expecting it in the header would take the dimension
        count for it."""
        knapsack = read(FIXTURES / "mknap2_pb7.txt")
        assert knapsack.optimum == 1035
        assert knapsack.optimum < sum(knapsack.profits) == 1696

    def test_a_real_specimen_wraps_its_values_where_the_writer_ran_out(self):
        """``PB7``'s 37 profits arrive ten, six, ten, ten and one to a line.
        Nothing about the layout is recoverable from the line breaks, which is
        why the reader tokenises."""
        text = (FIXTURES / "mknap2_pb7.txt").read_text()
        widths = [len(line.split()) for line in text.splitlines()]
        assert widths[4:9] == [10, 6, 10, 10, 1]
        knapsack = read(FIXTURES / "mknap2_pb7.txt")
        assert (knapsack.profits[9], knapsack.profits[10]) == (63, 6)
        assert (knapsack.profits[15], knapsack.profits[16]) == (61, 85)
        assert knapsack.profits[-1] == 31

    def test_the_instance_takes_its_name_from_the_problem_marker(self):
        """Within a file of forty-eight the marker is what identifies one of
        them; the file stem identifies only the file."""
        knapsack = read(FIXTURES / "mknap2_pb7.txt")
        assert knapsack.name == "PB7.DAT"
        assert knapsack.source.endswith("mknap2_pb7.txt")

    def test_the_banner_rules_between_instances_are_not_numbers(self):
        """``mknap2`` separates its instances with rows of ``+``.  They are
        punctuation, and a reader that took them for data would refuse the
        file."""
        assert "+++" in (FIXTURES / "mknap2_pb7.txt").read_text()
        assert read(FIXTURES / "mknap2_pb7.txt").optimum == 1035


class TestChoosingBetweenTheLayouts:
    """Decided by the width of the first line, and by nothing else."""

    def test_one_number_on_the_first_line_means_the_counted_layout(self):
        knapsack = parse("1\n2 3 0\n5 7\n1 2\n3 4\n5 6\n10 11 12\n")
        assert len(knapsack.profits) == 2
        assert len(knapsack.weights) == 3

    def test_two_numbers_on_the_first_line_mean_the_mknap2_layout(self):
        """The same nine values under the other reading, to show that the two
        do not merely differ in their headers: dimensions and items swap, the
        capacities move, and the last number becomes the optimum."""
        knapsack = parse("3 2\n5 7\n1 2 3\n4 5\n6 7\n8 9\n10\n")
        assert len(knapsack.profits) == 2
        assert len(knapsack.weights) == 3
        assert knapsack.capacities == (1, 2, 3)
        assert knapsack.optimum == 10

    def test_a_first_line_of_any_other_width_is_refused_not_guessed(self):
        """Both readings consume every number and disagree about all of them,
        so picking one produces a plausible instance and no error at all."""
        with pytest.raises(InstanceError, match="cannot tell which layout"):
            parse("1 2 3\n0 5 7 1 2 3 4 10 11\n")

    def test_the_complaint_about_an_undecidable_first_line_names_both(self):
        with pytest.raises(InstanceError, match="mknap2"):
            parse("1 2 3 0\n5 7\n1 2\n3 4\n10 11\n")


class TestSeveralProblemsInOneFile:
    """These files hold seven, thirty and forty-eight; the readers say so."""

    def test_read_all_returns_them_in_file_order_with_numbered_names(self):
        found = read_all(FIXTURES / "three_problems.mdkp")
        assert [k.name for k in found] == [
            "three_problems-1", "three_problems-2", "three_problems-3"
        ]
        assert [len(k.profits) for k in found] == [2, 4, 3]
        assert [len(k.weights) for k in found] == [1, 3, 2]

    def test_each_stated_optimum_is_the_optimum_of_its_own_problem(self):
        for knapsack in read_all(FIXTURES / "three_problems.mdkp"):
            if knapsack.optimum is not None:
                assert best_value(knapsack) == knapsack.optimum

    def test_read_refuses_a_file_of_several_rather_than_dropping_any(self):
        """The choice this reader makes, and the one the 0-1 reader makes:
        ``read`` returns one instance and raises when the file holds more,
        naming ``read_all``.  Returning the first quietly is how the other
        twenty-nine go missing."""
        with pytest.raises(InstanceError, match="read_all"):
            read(FIXTURES / "three_problems.mdkp")

    def test_the_complaint_says_how_many_there_were(self):
        with pytest.raises(InstanceError, match="3 problems"):
            read(FIXTURES / "three_problems.mdkp")

    def test_a_real_multi_problem_file_is_refused_by_read_too(self):
        with pytest.raises(InstanceError, match="2 problems"):
            read(FIXTURES / "mknapcb1_two_problems.txt")

    def test_parse_all_of_a_single_problem_file_is_a_tuple_of_one(self):
        """So a caller who does not know which file they hold can always use
        it."""
        for stem in ("hand_written.mdkp", "hand_written_mknap2.txt",
                     "mknap2_pb7.txt"):
            assert len(parse_all((FIXTURES / stem).read_text())) == 1

    def test_a_single_problem_file_keeps_its_unnumbered_name(self):
        """The suffix says which of several; with one there is no which."""
        assert read(FIXTURES / "hand_written.mdkp").name == "hand_written"


class TestDeclaredCountsMustMatch:
    """Refuse a file whose counts disagree with the values present."""

    def test_a_problem_count_larger_than_the_problems_present_is_refused(self):
        with pytest.raises(InstanceError, match="ends before"):
            parse("2\n2 2 0\n5 7\n1 2\n3 4\n10 11\n")

    def test_the_shortfall_names_which_problem_ran_out(self):
        with pytest.raises(InstanceError, match="problem 2 of 2"):
            parse("2\n2 2 0\n5 7\n1 2\n3 4\n10 11\n")

    def test_numbers_left_over_after_the_last_problem_are_refused(self):
        """A trailing value means the counts we read the file by are not the
        counts it was written with, whatever else came out right."""
        with pytest.raises(InstanceError, match="further number"):
            parse("1\n2 2 0\n5 7\n1 2\n3 4\n10 11\n99\n")

    def test_an_item_count_larger_than_the_values_present_is_refused(self):
        with pytest.raises(InstanceError, match="ends before"):
            parse("1\n5 2 0\n5 7 9 11 13\n1 2 3 4 5\n6 7 8 9 10\n10\n")

    def test_the_shortfall_names_the_value_it_ended_before(self):
        with pytest.raises(InstanceError, match="capacity.*dimension 2 of 2"):
            parse("1\n5 2 0\n5 7 9 11 13\n1 2 3 4 5\n6 7 8 9 10\n10\n")

    def test_a_dimension_count_larger_than_the_rows_present_is_refused(self):
        with pytest.raises(InstanceError, match="dimension 3 of 3"):
            parse("1\n2 3 0\n5 7\n1 2\n3 4\n")

    def test_an_mknap2_instance_that_ends_before_its_optimum_is_refused(self):
        """Its optimum is the last number of the instance, so a file truncated
        by one value ends in a place the counted layout has no equivalent
        of."""
        with pytest.raises(InstanceError, match="ends before the stated optimum"):
            parse("2 3\n5 7 9\n10 12\n1 2 3\n4 5 6\n")

    def test_numbers_after_a_complete_mknap2_instance_start_another(self):
        """There is no problem count to check them against, so a trailing
        value is read as the beginning of a further instance and the file is
        refused for ending inside it."""
        with pytest.raises(InstanceError, match="problem 2"):
            parse("2 3\n5 7 9\n10 12\n1 2 3\n4 5 6\n16\n7\n")

    def test_a_problem_count_of_zero_is_refused(self):
        with pytest.raises(InstanceError, match="positive integer"):
            parse("0\n")

    def test_an_item_count_of_zero_is_refused(self):
        with pytest.raises(InstanceError, match="item count.*positive integer"):
            parse("1\n0 2 0\n1 2\n")


class TestValuesAreIntegersAndPositive:
    """Because the circuits read their binary representations."""

    def test_a_real_file_full_of_zero_costs_is_refused(self):
        """Every one of ``mknap1``'s seven problems has zeros in its cost
        matrix -- 99 across the file -- so the whole collection is unreadable
        here.  That is a fact about the file, and the message has to make it
        one: a zero cost has no gate in the circuit that reads it."""
        with pytest.raises(InstanceError, match="binary representation"):
            read(FIXTURES / "mknap1_first_problem.txt")

    def test_the_zero_cost_complaint_names_the_item_and_the_dimension(self):
        """So that whoever hit it can look at the number themselves rather
        than take the reader's word for it."""
        with pytest.raises(InstanceError,
                           match="weight of item 1 in dimension 7 is 0"):
            read(FIXTURES / "mknap1_first_problem.txt")

    def test_the_zero_cost_complaint_names_its_line(self):
        with pytest.raises(InstanceError, match="line 11"):
            read(FIXTURES / "mknap1_first_problem.txt")

    def test_a_fractional_profit_is_a_format_error_not_a_rounding(self):
        """``mknap1``'s second problem states profits like ``600.1``.  There is
        no gate count for it: the cost depends on where the ones sit."""
        with pytest.raises(InstanceError, match="binary representation"):
            parse("1\n2 1 0\n600.1 7\n1 2\n10\n")

    def test_a_negative_profit_is_refused(self):
        with pytest.raises(InstanceError, match="profit of item 2.*positive"):
            parse("1\n2 1 0\n5 -7\n1 2\n10\n")

    def test_a_zero_capacity_is_refused(self):
        """Every weight is positive, so a budget of zero admits nothing; no
        benchmark set writes one, and reading it would cost a circuit that
        cannot select an item."""
        with pytest.raises(InstanceError, match="capacity of dimension 1 is 0"):
            parse("1\n2 1 0\n5 7\n1 2\n0\n")

    def test_a_fractional_capacity_is_refused(self):
        with pytest.raises(InstanceError, match="capacity of dimension 1"):
            parse("1\n2 1 0\n5 7\n1 2\n10.5\n")

    def test_a_negative_stated_optimum_is_refused(self):
        with pytest.raises(InstanceError, match="negative"):
            parse("1\n2 1 -4\n5 7\n1 2\n10\n")

    def test_an_optimum_above_the_total_profit_is_refused(self):
        """No selection beats taking everything, so such a claim is evidence
        that the profits and the costs are not where we think they are."""
        with pytest.raises(InstanceError, match="exceeds 12"):
            parse("1\n2 1 400\n5 7\n1 2\n10\n")

    def test_a_fractional_optimum_is_refused(self):
        with pytest.raises(InstanceError, match="stated optimum"):
            parse("1\n2 1 8.5\n5 7\n1 2\n10\n")


class TestBadFilesSayWhere:
    def test_an_empty_file_says_it_is_empty(self):
        with pytest.raises(InstanceError, match="empty"):
            parse("\n\n")

    def test_a_file_of_nothing_but_banner_rules_is_empty_too(self):
        with pytest.raises(InstanceError, match="empty"):
            parse("+++++\n\n+++++\n")

    def test_a_missing_file_says_so(self):
        with pytest.raises(InstanceError, match="cannot read"):
            read(FIXTURES / "no_such_instance.mdkp")

    def test_a_word_where_a_number_belongs_names_its_line(self):
        """Outside the documentation that precedes a ``problem`` marker these
        files hold nothing but numbers, so a word is a sign that we are not
        reading what we think we are."""
        with pytest.raises(InstanceError, match="line 3.*expected a number"):
            parse("1\n2 1 0\n5 seven\n1 2\n10\n")

    def test_a_marker_with_no_numbers_under_it_is_refused(self):
        with pytest.raises(InstanceError, match="no numbers under it"):
            parse("problem EMPTY.DAT\n+++++\nproblem PB1.DAT\n1 1\n5\n3\n2\n5\n")

    def test_every_complaint_about_a_line_names_that_line(self):
        for text in ("1\n2 1 0\n5 0\n1 2\n10\n",
                     "1\n2 1 0\n5 7\n1 2\n-10\n",
                     "1\n2 1 0\n5 7\n1 x\n10\n",
                     "1 2 3\n4 5 6\n"):
            with pytest.raises(InstanceError, match="line [0-9]+"):
                parse(text)


class TestTheDispatcher:
    def test_the_layout_name_is_the_one_detect_returns(self):
        assert LAYOUT == "multidimensional-knapsack"
        assert instances.detect(FIXTURES / "hand_written.mdkp") == LAYOUT

    def test_an_mdkp_file_reaches_this_reader_through_the_package(self):
        path = FIXTURES / "hand_written.mdkp"
        assert instances.read(path) == read(path)
        assert isinstance(instances.read(path), MultidimensionalKnapsack)

    def test_a_named_layout_reaches_it_whatever_the_extension_says(self):
        """The real files are all ``.txt``, which names no format at all."""
        path = FIXTURES / "mknap2_pb7.txt"
        assert instances.read(path, layout=LAYOUT) == read(path)

    def test_parse_and_read_agree_apart_from_the_name_and_the_source(self):
        path = FIXTURES / "hand_written.mdkp"
        by_hand = parse(path.read_text(), name="hand_written", source=str(path))
        assert by_hand == read(path)

    def test_the_description_says_what_it_is(self):
        knapsack = read(FIXTURES / "mknap2_pb7.txt")
        assert knapsack.describe() == "PB7.DAT: 37 items, 30 dimensions"
