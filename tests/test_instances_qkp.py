"""Reading quadratic knapsack files: what the format claims, and what it refuses.

The specimen is Billionnet and Soutif's, from ``https://cedric.cnam.fr/~soutif/QKP/``:
``tests/fixtures/qkp/r_100_100_1_first_eight.txt`` is the leading corner of
``jeu_100_100_1.txt``, its numbers unedited.

One reading in this module is worth more than the rest, and it is the one no
exception would ever catch.  A pair's entry in the file is what that pair earns
**in total** -- the site's typeset objective prints ``+ 55 x_1x_2 + 23 x_1x_3``
once per unordered pair, and those coefficients are the first triangle row read
left to right -- so it goes into :attr:`pairs` as it stands.  Doubling it, which
is what summing a symmetric matrix's two halves would amount to on a file that
has only one half, leaves an instance that parses, sums and plots exactly as
well, and changes the position of the lowest set bit the circuits cost.  So the
tests here do not merely read the numbers back: they brute-force the best
packing of what was parsed and check it against an objective worked out by
hand, and they check that the doubled reading would fail that.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Tuple

import pytest

from hybrid_benchmarking import instances
from hybrid_benchmarking.instances import InstanceError, QuadraticKnapsack
from hybrid_benchmarking.instances.qkp import LAYOUT, parse, read

FIXTURES = Path(__file__).parent / "fixtures" / "qkp"

#: The hand-written instance, stated here independently of the file it is in.
#: Powers of two, so that any doubling, halving or transposition of the
#: triangle moves the objective rather than rounding it.
HAND_PROFITS = (10, 20, 30, 40, 50)
HAND_WEIGHTS = (3, 4, 5, 6, 7)
HAND_CAPACITY = 12
HAND_PAIRS = {
    (1, 0): 1, (2, 0): 2, (4, 0): 4,
    (2, 1): 8, (3, 1): 16,
    (3, 2): 32, (4, 2): 64,
    (4, 3): 128,
}
#: Worked out by hand: the packings within a capacity of 12 are
#: {2,4} at 30+50+64, {2,3} at 30+40+32, {0,1,2} at 60+1+2+8, {1,3} at
#: 20+40+16, and smaller.  The first wins.
HAND_OPTIMUM = 144


def value_of(instance: QuadraticKnapsack, chosen: Tuple[int, ...]) -> int:
    """What a selection earns: its items, plus every pair inside it once."""
    taken = set(chosen)
    return (
        sum(instance.profits[item] for item in taken)
        + sum(bonus for (high, low), bonus in instance.pairs.items()
              if high in taken and low in taken)
    )


def best_packing(instance: QuadraticKnapsack) -> int:
    """The optimum, by trying every subset.  Only fixture-sized instances."""
    items = range(len(instance.profits))
    best = 0
    for size in items:
        for chosen in itertools.combinations(items, size + 1):
            if sum(instance.weights[item] for item in chosen) <= instance.capacity:
                best = max(best, value_of(instance, chosen))
    return best


def doubled(instance: QuadraticKnapsack) -> QuadraticKnapsack:
    """The same instance under the rejected reading of the triangle.

    A reader that took the file for one half of a symmetric matrix would add
    ``c_ij`` to a ``c_ji`` that is not there, and arrive at exactly this.
    """
    twice: Dict[Tuple[int, int], int] = {
        pair: 2 * bonus for pair, bonus in instance.pairs.items()
    }
    return QuadraticKnapsack(
        name=instance.name, source=instance.source, layout=instance.layout,
        profits=instance.profits, weights=instance.weights,
        capacity=instance.capacity, pairs=twice,
    )


class TestAHandWrittenInstance:
    """Every number asserted one at a time, against the file as written."""

    def instance(self) -> QuadraticKnapsack:
        return read(FIXTURES / "hand_five_items.qkp")

    def test_the_linear_profits_are_the_third_line_in_order(self):
        assert self.instance().profits == HAND_PROFITS

    def test_the_weights_and_the_capacity_are_the_constraint(self):
        knapsack = self.instance()
        assert knapsack.weights == HAND_WEIGHTS
        assert knapsack.capacity == HAND_CAPACITY

    def test_each_triangle_entry_becomes_one_pair_keyed_higher_lower(self):
        """Row ``i`` of the block pairs item ``i`` with every item after it,
        and the contract keys a pair by its higher item first."""
        assert self.instance().pairs == HAND_PAIRS

    def test_every_pair_is_asserted_element_by_element(self):
        pairs = self.instance().pairs
        for pair, bonus in HAND_PAIRS.items():
            assert pairs[pair] == bonus

    def test_a_zero_entry_carries_no_pair_at_all(self):
        """"Absent pairs earn nothing" is the contract, so a pair worth
        nothing is absent rather than present and zero -- the circuit has no
        addition to perform for it."""
        pairs = self.instance().pairs
        assert (3, 0) not in pairs  # the file writes 0 for items 1 and 4
        assert (4, 1) not in pairs  # and for items 2 and 5
        assert len(pairs) == 8      # of the 10 unordered pairs of 5 items

    def test_no_pair_is_keyed_the_other_way_round(self):
        assert all(high > low for high, low in self.instance().pairs)

    def test_the_best_packing_of_what_we_read_is_the_hand_worked_objective(self):
        """The independent check.  144 was worked out from the file by hand;
        this brute-forces it from the parsed profits, pairs, weights and
        capacity, so a triangle read one row out, transposed or doubled does
        not agree with it."""
        assert best_packing(self.instance()) == HAND_OPTIMUM

    def test_that_objective_is_the_selection_it_is_claimed_to_be(self):
        """Items 3 and 5, earning 30 and 50 alone and 64 together."""
        knapsack = self.instance()
        assert value_of(knapsack, (2, 4)) == HAND_OPTIMUM
        assert sum(knapsack.weights[item] for item in (2, 4)) == 12


class TestThePairConvention:
    """The entry is the whole pair's profit, not half of it."""

    def test_the_rejected_reading_disagrees_about_the_optimum(self):
        """If the file's triangle were half of a symmetric matrix, every pair
        would be worth twice what is read here and the best packing would be
        208 rather than 144.  Both are perfectly plausible numbers; only one
        is the objective the file states."""
        knapsack = read(FIXTURES / "hand_five_items.qkp")
        assert best_packing(knapsack) == 144
        assert best_packing(doubled(knapsack)) == 208

    def test_the_specimens_first_triangle_row_is_the_typeset_objective(self):
        """The site's own picture of an instance from this generator prints
        ``+ 55 x_1x_2 + 23 x_1x_3 + 35 x_1x_4 + 44 x_1x_5 + 5 x_1x_6 +
        91 x_1x_7 + 95 x_1x_8``, once per unordered pair, and that is the
        first row of the block read left to right."""
        pairs = read(FIXTURES / "r_100_100_1_first_eight.txt").pairs
        row = [55, 23, 35, 44, 5, 91, 95]
        assert [pairs[(other, 0)] for other in range(1, 8)] == row

    def test_the_number_of_pairs_is_the_strict_upper_triangle(self):
        """``n(n-1)/2``, not ``n(n+1)/2``: the diagonal is the linear profits
        and is not in the block.  At 100 % density every one of them is
        non-zero, so every one is present."""
        knapsack = read(FIXTURES / "r_100_100_1_first_eight.txt")
        assert len(knapsack.pairs) == 8 * 7 // 2


class TestTheDistributedLayout:
    """A real specimen, trimmed: ``jeu_100_100_1.txt``'s leading corner."""

    def instance(self) -> QuadraticKnapsack:
        return read(FIXTURES / "r_100_100_1_first_eight.txt")

    def test_the_reference_line_names_the_instance_not_the_file(self):
        knapsack = self.instance()
        assert knapsack.name == "r_100_100_1"
        assert knapsack.source.endswith("r_100_100_1_first_eight.txt")

    def test_the_linear_profits_are_the_files_own_numbers(self):
        assert self.instance().profits == (91, 75, 50, 13, 91, 76, 55, 12)

    def test_the_capacity_is_read_before_the_weights(self):
        """The order the distributed files write, against the description
        that has the weights first: 616 is a capacity and the eight small
        numbers after it are weights, and reading them the other way round
        would make the capacity 10 and the first weight 616."""
        knapsack = self.instance()
        assert knapsack.capacity == 616
        assert knapsack.weights == (10, 2, 28, 10, 44, 14, 21, 10)

    def test_the_trailing_comment_block_is_not_read_as_data(self):
        """The layout ends with free-form comments -- a density, a seed, prose
        -- and none of it is an item."""
        assert len(self.instance().profits) == 8

    def test_the_file_states_no_optimum(self):
        """The optima of this set live in the ``N<n>D<density>.txt`` result
        tables beside the instances, keyed by the generator's seed rather than
        by the instance's own reference.  Nothing in this file claims one, so
        nothing is claimed here."""
        assert self.instance().optimum is None

    def test_the_layout_is_the_one_the_package_dispatches_on(self):
        assert self.instance().layout == LAYOUT == "quadratic-knapsack"

    def test_the_description_says_what_it_is(self):
        assert self.instance().describe() == (
            "r_100_100_1: 8 items, 28 paired, capacity 616"
        )

    def test_the_unconstrained_trim_takes_everything(self):
        """Its capacity is the full instance's and its eight weights sum to
        139, so the best packing is every item and every pair -- which is a
        second, independent statement that the pairs are the ones read."""
        knapsack = self.instance()
        assert best_packing(knapsack) == (
            sum(knapsack.profits) + sum(knapsack.pairs.values())
        )


class TestTheCapacityAndTheWeightsAreToldApartByCounting:
    """The two lines hold one number and ``n``; that decides, not the order."""

    def test_both_orders_state_the_same_instance(self):
        first = parse("r\n3\n10 20 30\n5 7\n9\n\n0\n9\n3 4 5\n")
        second = parse("r\n3\n10 20 30\n5 7\n9\n\n0\n3 4 5\n9\n")
        assert first == second
        assert first.capacity == 9 and first.weights == (3, 4, 5)

    def test_one_item_is_refused_because_counting_cannot_decide(self):
        """With a single item the capacity line and the weight line hold one
        number each, the two readings disagree about both, and there is
        nothing in the file to tell them apart."""
        with pytest.raises(InstanceError, match="cannot tell"):
            parse("r\n1\n10\n\n0\n9\n3\n")

    def test_a_constraint_of_the_wrong_width_is_refused(self):
        with pytest.raises(InstanceError, match="line 2 states 3 items"):
            parse("r\n3\n10 20 30\n5 7\n9\n\n0\n9\n3 4\n")


class TestValuesAreIntegersAndPositive:
    """Because the circuits read their binary representations."""

    def test_a_zero_linear_profit_says_why_it_cannot_be_one(self):
        with pytest.raises(InstanceError, match="binary representation"):
            read(FIXTURES / "zero_profit.qkp")

    def test_the_zero_profit_complaint_names_its_line(self):
        with pytest.raises(InstanceError, match="line 3"):
            read(FIXTURES / "zero_profit.qkp")

    def test_every_instance_below_full_density_is_refused_for_that_reason(self):
        """Worth stating, because it is a consequence rather than an accident:
        the density these files quote thins the diagonal too, so a 25 %
        instance has mostly zero linear profits and none of them can be
        costed.  The refusal happens here, as a format complaint naming a
        line, and not later inside a gate count."""
        with pytest.raises(InstanceError, match="line 3"):
            parse("r_3_25_1\n3\n0 0 30\n5 7\n9\n\n0\n9\n3 4 5\n")

    def test_a_fractional_weight_is_a_format_error_not_a_rounding(self):
        with pytest.raises(InstanceError, match="binary representation"):
            parse("r\n3\n10 20 30\n5 7\n9\n\n0\n9\n3 4.5 5\n")

    def test_a_negative_profit_is_refused(self):
        with pytest.raises(InstanceError, match="positive integer"):
            parse("r\n3\n10 -20 30\n5 7\n9\n\n0\n9\n3 4 5\n")

    def test_a_capacity_of_zero_is_refused(self):
        with pytest.raises(InstanceError, match="positive integer"):
            parse("r\n3\n10 20 30\n5 7\n9\n\n0\n0\n3 4 5\n")


class TestPairProfitsAreBonusesOnly:
    def test_a_negative_pair_profit_is_refused(self):
        with pytest.raises(InstanceError, match="no gate for a pair that costs"):
            read(FIXTURES / "negative_pair.qkp")

    def test_the_negative_pair_complaint_names_its_line_and_its_items(self):
        with pytest.raises(InstanceError, match="line 4"):
            read(FIXTURES / "negative_pair.qkp")
        with pytest.raises(InstanceError, match=r"item 1, item 3"):
            read(FIXTURES / "negative_pair.qkp")

    def test_a_zero_pair_profit_is_not_refused(self):
        """It is the ordinary case in a sparse instance and it means the pair
        earns nothing, which the contract writes as an absent key."""
        assert parse("r\n3\n10 20 30\n0 0\n0\n\n0\n9\n3 4 5\n").pairs == {}

    def test_a_fractional_pair_profit_is_refused(self):
        with pytest.raises(InstanceError, match="not an integer"):
            parse("r\n3\n10 20 30\n5 7.5\n9\n\n0\n9\n3 4 5\n")


class TestTheTriangleIsCheckedAgainstN:
    def test_a_block_with_too_few_rows_is_refused_naming_the_line(self):
        """Four items want rows of 3, 2 and 1 entries.  A block of two rows
        would otherwise read the constraint type as its third row and carry
        on, pairing items the file never paired."""
        with pytest.raises(InstanceError, match="line 5"):
            read(FIXTURES / "short_triangle.qkp")

    def test_the_complaint_says_how_many_rows_there_should_have_been(self):
        with pytest.raises(InstanceError, match="3 rows"):
            read(FIXTURES / "short_triangle.qkp")

    def test_a_block_with_too_many_rows_is_refused(self):
        with pytest.raises(InstanceError, match="2 rows"):
            parse("r\n3\n10 20 30\n5 7\n9\n1\n\n0\n9\n3 4 5\n")

    def test_a_row_of_the_wrong_width_is_refused_naming_the_line(self):
        """Row 2 of a four-item instance pairs item 2 with items 3 and 4, so
        it holds two entries; three would shift every pair after it."""
        with pytest.raises(InstanceError, match="line 5"):
            parse("r\n4\n10 20 30 40\n5 7 9\n11 13 15\n17\n\n0\n15\n3 4 5 6\n")

    def test_a_linear_line_disagreeing_with_n_is_refused(self):
        with pytest.raises(InstanceError, match="line 3"):
            parse("r\n4\n10 20 30\n5 7 9\n11 13\n17\n\n0\n15\n3 4 5 6\n")


class TestTheConstraintType:
    def test_zero_means_at_most_the_capacity_and_is_read(self):
        assert parse("r\n3\n10 20 30\n5 7\n9\n\n0\n9\n3 4 5\n").capacity == 9

    def test_one_means_an_equality_constraint_and_is_refused(self):
        """A file asking for a selection weighing exactly the capacity is a
        different problem; relaxing it to 'at most' would answer a question
        nobody asked and would look right doing it."""
        with pytest.raises(InstanceError, match="equality constraint"):
            parse("r\n3\n10 20 30\n5 7\n9\n\n1\n9\n3 4 5\n")

    def test_an_unrecognised_indicator_is_refused(self):
        with pytest.raises(InstanceError, match="line 7"):
            parse("r\n3\n10 20 30\n5 7\n9\n\n2\n9\n3 4 5\n")


class TestBadFilesSayWhere:
    def test_an_empty_file_says_it_is_empty(self):
        with pytest.raises(InstanceError, match="empty"):
            parse("\n\n")

    def test_a_missing_file_says_so(self):
        with pytest.raises(InstanceError, match="cannot read"):
            read(FIXTURES / "no_such_instance.qkp")

    def test_a_file_ending_before_the_constraint_says_so(self):
        with pytest.raises(InstanceError, match="ends before"):
            parse("r\n3\n10 20 30\n5 7\n9\n")

    def test_a_reference_line_of_several_words_is_refused(self):
        with pytest.raises(InstanceError, match="line 1"):
            parse("r 100 25 1\n3\n10 20 30\n5 7\n9\n\n0\n9\n3 4 5\n")

    def test_a_file_without_a_reference_line_is_refused_with_the_reason(self):
        """Its count is read as the reference and its linear profits as the
        count.  Both readings fit such a file and they disagree about every
        number in it, so this refuses and says which line to look at."""
        with pytest.raises(InstanceError, match="reference"):
            parse("3\n10 20 30\n5 7\n9\n\n0\n9\n3 4 5\n")

    def test_every_complaint_about_a_line_names_that_line(self):
        for text in ("r\n0\n\n0\n9\n3\n",
                     "r\nx\n10 20 30\n5 7\n9\n\n0\n9\n3 4 5\n",
                     "r\n3\n10 20 30\n5 7\n9\n\n0\n9\n3 4 x\n",
                     "r\n3\n10 20 30\n5 7\n9\n\n0 0\n9\n3 4 5\n"):
            with pytest.raises(InstanceError, match="line [0-9]+"):
                parse(text)


class TestTheDispatcher:
    def test_a_qkp_file_reaches_this_reader_through_the_package(self):
        path = FIXTURES / "hand_five_items.qkp"
        assert instances.read(path) == read(path)
        assert isinstance(instances.read(path), QuadraticKnapsack)

    def test_detect_names_the_layout_this_module_reports(self):
        assert instances.detect(FIXTURES / "hand_five_items.qkp") == LAYOUT
