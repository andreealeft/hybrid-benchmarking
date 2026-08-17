"""Reading knapsack files: what each layout claims, and what it refuses.

The claims here belong to the file formats, not to the reader.  Pisinger's
generator writes ``index,profit,weight`` and records the optimum it found, so
the optimum a brute force finds in what we read back has to be the number the
file states -- a reader that transposed two columns would keep the claim and
break its truth, and would otherwise produce a perfectly plausible instance.
The plain Martello-Toth layouts state one instance three ways, so the three must
agree with each other; where the layout genuinely does not say which end the
capacity sits at, the reader is required to say so rather than pick.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from hybrid_benchmarking import instances
from hybrid_benchmarking.instances import InstanceError, Knapsack
from hybrid_benchmarking.instances.knapsack import (
    parse,
    parse_all,
    read,
    read_all,
)

FIXTURES = Path(__file__).parent / "fixtures" / "knapsack"

#: The instance the four single-instance fixtures each state, one per layout.
ITEMS = ((44, 23), (46, 31), (90, 52), (72, 44), (91, 54), (40, 26))
CAPACITY = 60


def best_value(instance: Knapsack) -> int:
    """The optimum, by trying every subset.  Only fixture-sized instances."""
    take = itertools.product((0, 1), repeat=len(instance.profits))
    values = [
        sum(p for chosen, p in zip(pattern, instance.profits) if chosen)
        for pattern in take
        if sum(w for chosen, w in zip(pattern, instance.weights) if chosen)
        <= instance.capacity
    ]
    return max(values)


class TestPisinger:
    """``knapPI_*`` as the generator writes it."""

    def test_the_header_and_every_item_row_are_read_as_written(self):
        knapsack = read(FIXTURES / "pisinger_six_items.kp")
        assert knapsack.profits == (44, 46, 90, 72, 91, 40)
        assert knapsack.weights == (23, 31, 52, 44, 54, 26)
        assert tuple(zip(knapsack.profits, knapsack.weights)) == ITEMS
        assert knapsack.capacity == CAPACITY
        assert knapsack.optimum == 91
        assert knapsack.layout == "pisinger"

    def test_the_stated_optimum_is_the_optimum_of_what_we_read(self):
        """``z`` is the file's claim about its own items.  Reading the profit
        column as the weight column leaves the claim standing and makes it
        false, which is exactly the failure a plausible number would hide."""
        knapsack = read(FIXTURES / "pisinger_six_items.kp")
        assert best_value(knapsack) == knapsack.optimum

    def test_the_name_comes_from_the_file_not_from_the_file_name(self):
        """Within a file of a hundred instances the stated name is what
        identifies one of them; the stem identifies only the file."""
        knapsack = read(FIXTURES / "pisinger_six_items.kp")
        assert knapsack.name == "knapPI_1_6_100_1"
        assert knapsack.source.endswith("pisinger_six_items.kp")

    def test_an_unrecognised_header_key_is_refused(self):
        """The header is ``n``, ``c``, ``z`` and ``time``.  A fifth key means
        the file is not the layout we took it for."""
        with pytest.raises(InstanceError, match="line 3"):
            parse("small\nn 2\nq 4\nc 9\n1,3,2\n2,5,4\n")

    def test_fewer_item_rows_than_n_says_is_refused(self):
        with pytest.raises(InstanceError, match="ends after 2"):
            parse("small\nn 3\nc 9\n1,3,2\n2,5,4\n-----\n")

    def test_more_item_rows_than_n_says_is_refused(self):
        with pytest.raises(InstanceError, match="line 6"):
            parse("small\nn 2\nc 9\n1,3,2\n2,5,4\n3,7,6\n4,8,7\n")

    def test_an_item_row_of_the_wrong_width_is_refused(self):
        """Some distributions add the solution vector as a fourth column.  It
        is a different file, and taking the third column as the weight of a
        four-column row would silently read the wrong number."""
        with pytest.raises(InstanceError, match="line 4"):
            parse("small\nn 2\nc 9\n1,3,2,1\n2,5,4,0\n")

    def test_an_optimum_above_the_total_profit_is_refused(self):
        """No selection beats taking everything, so such a ``z`` is evidence
        that the columns are not in the order we assumed."""
        with pytest.raises(InstanceError, match="exceeds 8"):
            parse("small\nn 2\nc 9\nz 400\n1,3,2\n2,5,4\n")


class TestSeveralInstancesInOneFile:
    """Pisinger ships a hundred instances per file; the readers say so."""

    def test_read_all_returns_them_in_file_order_with_their_own_names(self):
        found = read_all(FIXTURES / "pisinger_three_instances.kp")
        assert [k.name for k in found] == [
            "knapPI_1_3_1000_1", "knapPI_1_3_1000_2", "knapPI_1_4_1000_3"
        ]
        assert [len(k.profits) for k in found] == [3, 3, 4]
        assert [k.capacity for k in found] == [20, 15, 12]

    def test_each_instance_keeps_its_own_stated_optimum(self):
        for knapsack in read_all(FIXTURES / "pisinger_three_instances.kp"):
            assert best_value(knapsack) == knapsack.optimum

    def test_read_refuses_a_file_of_several_rather_than_dropping_any(self):
        """The choice this reader makes: ``read`` returns one instance and
        raises when the file holds more, naming ``read_all``.  Returning the
        first quietly is how ninety-nine instances go missing."""
        with pytest.raises(InstanceError, match="read_all"):
            read(FIXTURES / "pisinger_three_instances.kp")

    def test_the_complaint_says_how_many_there_were(self):
        with pytest.raises(InstanceError, match="3 instances"):
            read(FIXTURES / "pisinger_three_instances.kp")

    def test_parse_all_of_a_single_instance_file_is_a_tuple_of_one(self):
        """So a caller who does not know which file they hold can always use
        it."""
        assert len(parse_all(Path(FIXTURES / "capacity_second.txt").read_text())) == 1


class TestMartelloToth:
    """The plain layout, whose only difficulty is where the capacity sits."""

    def test_the_two_orderings_state_the_same_instance(self):
        """``n, c, then the pairs`` and ``n, the pairs, then c`` are one format
        written two ways, so they must read back identical -- name, layout and
        all."""
        second = parse(Path(FIXTURES / "capacity_second.txt").read_text())
        last = parse(Path(FIXTURES / "capacity_last.txt").read_text())
        assert second == last

    def test_both_orderings_give_the_items_the_file_lists(self):
        for stem in ("capacity_second", "capacity_last", "count_and_capacity"):
            knapsack = read(FIXTURES / "{}.txt".format(stem))
            assert tuple(zip(knapsack.profits, knapsack.weights)) == ITEMS
            assert knapsack.capacity == CAPACITY
            assert knapsack.optimum is None  # the layout states no optimum

    def test_the_count_and_the_capacity_may_share_the_first_line(self):
        knapsack = read(FIXTURES / "count_and_capacity.txt")
        assert len(knapsack.profits) == 6
        assert knapsack.capacity == CAPACITY

    def test_a_plain_file_takes_its_name_from_the_file(self):
        knapsack = read(FIXTURES / "capacity_second.txt")
        assert knapsack.name == "capacity_second"
        assert knapsack.layout == "martello-toth"

    def test_a_file_that_does_not_say_which_end_is_refused(self):
        """Every number on its own line fits both orderings, and the two
        readings disagree about every item: one reads the capacity as 20 and
        the first item as 10 for 7, the other reads the capacity as 5 and the
        first item as 20 for 10."""
        with pytest.raises(InstanceError, match="cannot tell"):
            read(FIXTURES / "ambiguous_one_per_line.txt")

    def test_the_ambiguity_is_refused_rather_than_resolved_by_position(self):
        """Guards against the tempting shortcut of taking the second line
        because it happens to come first."""
        text = Path(FIXTURES / "ambiguous_one_per_line.txt").read_text()
        with pytest.raises(InstanceError):
            parse(text)

    def test_a_count_that_disagrees_with_the_item_lines_is_refused(self):
        with pytest.raises(InstanceError, match="states 4 items"):
            parse("4\n60\n44 23\n46 31\n90 52\n")

    def test_an_item_line_of_three_numbers_is_refused(self):
        with pytest.raises(InstanceError, match="line 3"):
            parse("2 60\n44 23\n46 31 7\n")


class TestValuesAreIntegersAndPositive:
    """Because the circuits read their binary representations."""

    def test_a_zero_profit_says_why_it_cannot_be_one(self):
        with pytest.raises(InstanceError, match="binary representation"):
            read(FIXTURES / "pisinger_zero_profit.kp")

    def test_the_zero_profit_complaint_names_its_line(self):
        with pytest.raises(InstanceError, match="line 7"):
            read(FIXTURES / "pisinger_zero_profit.kp")

    def test_a_fractional_weight_is_a_format_error_not_a_rounding(self):
        """There is no gate count for a weight of 4.5: the cost depends on
        where the ones sit in the value."""
        with pytest.raises(InstanceError, match="binary representation"):
            parse("2\n60\n44 23\n46 4.5\n")

    def test_a_negative_profit_is_refused(self):
        with pytest.raises(InstanceError, match="positive integer"):
            parse("2\n60\n44 23\n-46 31\n")

    def test_a_capacity_of_zero_is_refused(self):
        with pytest.raises(InstanceError, match="capacity is 0"):
            parse("2\n0\n44 23\n46 31\n")

    def test_a_fractional_capacity_is_refused(self):
        with pytest.raises(InstanceError, match="capacity"):
            parse("small\nn 2\nc 9.5\n1,3,2\n2,5,4\n")


class TestBadFilesSayWhere:
    def test_an_empty_file_says_it_is_empty(self):
        with pytest.raises(InstanceError, match="empty"):
            parse("\n\n")

    def test_a_missing_file_says_so(self):
        with pytest.raises(InstanceError, match="cannot read"):
            read(FIXTURES / "no_such_instance.kp")

    def test_every_complaint_about_a_line_names_that_line(self):
        """Twelve ways to be malformed; all of them have to be locatable."""
        for text in ("small\nn 2\nc 9\n1,3\n2,5,4\n",
                     "small\nn 2\nc 9\n1,3,0\n2,5,4\n",
                     "2\n60\n44 23\n46\n",
                     "2 60\n44 23\n0 31\n"):
            with pytest.raises(InstanceError, match="line [0-9]+"):
                parse(text)


class TestTheDispatcher:
    def test_a_kp_file_reaches_this_reader_through_the_package(self):
        """``instances.read`` picks the reader from the extension, so a ``.kp``
        file has to come back as the same knapsack."""
        path = FIXTURES / "pisinger_six_items.kp"
        assert instances.read(path) == read(path)
        assert isinstance(instances.read(path), Knapsack)

    def test_the_description_says_what_it_is(self):
        knapsack = read(FIXTURES / "pisinger_six_items.kp")
        assert knapsack.describe() == "knapPI_1_6_100_1: 6 items, capacity 60"
