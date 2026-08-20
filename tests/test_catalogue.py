"""Seventy-one names for nine problems.

Someone siting charging points and someone siting satellites are solving the
same problem, and neither of them wants to be told that before they can get a
number. So the *family* owns the routes, the fields and the shapes, and a
problem is a name and a story attached to one. That arrangement is what these
tests hold: a name may be added for the cost of a line of prose, and it cannot
quietly acquire a route its family does not have.

The menu shows the names and nothing else. The technical identity stays in the
data -- it is still on every cost's provenance, which is the one place it must
never be dropped -- but it is not what someone browsing is made to read.
"""

from __future__ import annotations

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.problems import FAMILIES, family_of
from hybrid_benchmarking.web import problems as menu

BY_FAMILY = {}
for _problem in hb.PROBLEMS:
    BY_FAMILY.setdefault(_problem.family, []).append(_problem)


class TestTheCatalogue:
    def test_every_problem_belongs_to_a_family(self):
        for problem in hb.PROBLEMS:
            assert problem.family in FAMILIES, problem.key

    def test_every_family_is_used(self):
        assert set(BY_FAMILY) == set(FAMILIES)

    def test_names_and_keys_are_unique(self):
        keys = [p.key for p in hb.PROBLEMS]
        labels = [p.label for p in hb.PROBLEMS]
        assert len(set(keys)) == len(keys)
        assert len(set(labels)) == len(labels)

    def test_every_problem_has_a_story_of_its_own(self):
        # The blurb is what distinguishes two names for one problem, so two
        # problems sharing one would be two entries doing one entry's work.
        blurbs = [p.blurb for p in hb.PROBLEMS]
        assert len(set(blurbs)) == len(blurbs)
        for problem in hb.PROBLEMS:
            assert len(problem.blurb) > 40, problem.key

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_a_family_gives_every_one_of_its_names_the_same_routes(self, family):
        expected = FAMILIES[family][1]
        for problem in BY_FAMILY[family]:
            assert problem.routes == expected, problem.key

    def test_the_names_outnumber_the_problems_underneath(self):
        # Which is the point: nine problems, and enough names that someone
        # arriving with a real task finds one that sounds like it.
        assert len(hb.PROBLEMS) > 4 * len(FAMILIES)


class TestTheMenuShowsNamesNotClassifications:
    def test_a_menu_entry_carries_its_label_and_its_story(self):
        for entry in menu():
            assert entry["label"] and entry["blurb"]

    def test_two_names_for_one_problem_are_both_offered(self):
        labels = {p["label"] for p in menu()}
        assert "Where to put charging points" in labels
        assert "Which orbital slots to fill" in labels
        assert family_of("charging-stations") == family_of("satellite-siting")

    def test_the_technical_identity_survives_where_it_matters(self):
        # Not in the menu, but never off the answer: the cost still names the
        # lemma, the bound and every assumption behind it.
        route = hb.get_route("charging-stations", "tree-generator")
        cost = hb.compose({"routine": route.target, "unit": route.unit.name})
        assert cost.provenance.sources
        assert cost.provenance.assumptions


class TestTheKnapsackVariantsAreReachable:
    def test_the_quadratic_family_has_names_people_use(self):
        assert len(BY_FAMILY["quadratic-knapsack"]) >= 8

    def test_the_multidimensional_family_does_too(self):
        assert len(BY_FAMILY["multidimensional-knapsack"]) >= 8

    @pytest.mark.parametrize("key", ["satellite-siting", "charging-stations",
                                     "team-selection"])
    def test_a_quadratic_problem_costs_from_typed_values(self, key):
        route = hb.get_route(key, "tree-generator")
        data = hb.Dataset(instance={
            "profits": [6, 2, 1, 2], "weights": [2, 2, 1, 5],
            "capacity": 7, "profit_bound": 11,
            "pair_profits": {(0, 1): 6},
        })
        report = hb.run(route, data)
        assert report["total"] > 0
        assert report["unit_label"] == "cycles"

    @pytest.mark.parametrize("key", ["container-loading", "cloud-packing",
                                     "menu-planning"])
    def test_a_multidimensional_problem_costs_from_typed_values(self, key):
        route = hb.get_route(key, "tree-generator")
        data = hb.Dataset(instance={
            "profits": [6, 2, 1, 2],
            "weights": [[2, 2, 1, 5], [3, 1, 4, 2]],
            "capacities": [7, 6], "profit_bound": 11,
        })
        report = hb.run(route, data)
        assert report["total"] > 0

    def test_the_pair_bonuses_are_said_to_be_bonuses(self):
        # Nothing negative: the circuit has a gate for a pair that earns
        # something together and none for a pair that costs something.
        route = hb.get_route("charging-stations", "tree-generator")
        field = next(f for f in route.per_instance if f.name == "pair_profits")
        assert "Bonuses only" in field.help
        assert "not something this circuit has a gate for" in field.help

    def test_a_negative_pair_bonus_is_refused_rather_than_costed(self):
        from hybrid_benchmarking.routines.qkp import qkp_gates

        with pytest.raises(ValueError):
            qkp_gates([6, 2], {(0, 1): -4}, [2, 2], 7, 11)
