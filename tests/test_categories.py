"""The headings the menu is grouped under.

Seventy-one names in one column is a list nobody reads to the end of.  The
grouping exists so a heading can be skipped, which means every name must sit
under exactly one, and the headings must be findable by someone who does not
know what their problem is called.
"""

from __future__ import annotations

import pytest

from hybrid_benchmarking import PROBLEMS, web
from hybrid_benchmarking.problems import CATEGORIES, CATEGORY, category_of


KEYS = {key for key, _ in CATEGORIES}


def test_every_problem_has_a_heading():
    for problem in PROBLEMS:
        assert problem.key in CATEGORY, problem.key


def test_no_heading_is_invented_for_a_problem_that_does_not_exist():
    known = {problem.key for problem in PROBLEMS}
    assert set(CATEGORY) <= known


def test_every_heading_is_used():
    used = {category_of(problem.key) for problem in PROBLEMS}
    assert used == KEYS


def test_headings_are_declared():
    assert set(CATEGORY.values()) <= KEYS


@pytest.mark.parametrize("key,label", CATEGORIES)
def test_a_heading_reads_as_a_task_not_as_a_classification(key, label):
    """No heading names a family, an algorithm or a piece of mathematics."""
    lowered = label.lower()
    for jargon in ("knapsack", "vertex", "clique", "independent set", "flow",
                   "linear program", "simplex", "interior point", "quadratic",
                   "multidimensional", "matrix", "quantum", "np-hard"):
        assert jargon not in lowered, (key, label)


def test_the_grouping_is_not_the_family_grouping():
    """The point of cutting across families rather than exposing them.

    Grouping by family would put siting a satellite beside siting a charging
    point under one heading and announce that they are the same problem, which
    is the classification the catalogue exists to spare people.  It is enough
    that the two do not line up: several headings must mix families, and no
    family may live under a single heading.
    """
    families = {}
    for problem in PROBLEMS:
        families.setdefault(category_of(problem.key), set()).add(problem.family)
    mixed = [key for key, seen in families.items() if len(seen) > 1]
    assert len(mixed) >= 3, families

    spread = {}
    for problem in PROBLEMS:
        spread.setdefault(problem.family, set()).add(category_of(problem.key))
    assert any(len(seen) > 1 for seen in spread.values()), spread


def test_the_two_siting_twins_are_not_the_whole_of_their_heading():
    """The specific pair that motivated the masking."""
    together = [problem.key for problem in PROBLEMS
                if category_of(problem.key) == category_of("satellite-siting")]
    assert "charging-stations" in together
    assert len(together) > 4, together
    assert len({problem.family for problem in PROBLEMS
                if problem.key in together}) > 1


def test_the_page_receives_the_headings_in_contiguous_runs():
    """How the page groups without knowing the headings or their order."""
    served = web.problems()
    assert len(served) == len(PROBLEMS)
    runs = []
    for entry in served:
        if not runs or runs[-1] != entry["category"]:
            runs.append(entry["category"])
    assert len(runs) == len(set(runs)) == len(KEYS)
    assert runs == [key for key, _ in CATEGORIES]


def test_every_served_problem_carries_its_heading_in_words():
    heading = dict(CATEGORIES)
    for entry in web.problems():
        assert entry["category_label"] == heading[entry["category"]]
