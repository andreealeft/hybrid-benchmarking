"""The level for someone who has a problem but no file.

Two things are being tested, and the second matters more.

The first is that plain-language answers turn into a real instance and a real
number. Nobody arrives holding a vertex count; they arrive with sixty people and
a hundred and eighty shifts. So each problem asks in its own nouns, an instance
of that shape is built, and the ordinary path -- the same solvers, the same log,
the same lemmas -- costs it.

The second is that the number says what it is. It is the cost of a *generated*
instance, which is a real thing and not an answer about the user's data, and
that sentence has to survive all the way onto every cost. And the columns of the
comparison must never turn into a ranking: they count different units and rest
on different assumptions, so they sit beside each other and the tool declines to
add them. Those two properties are the ones that quietly stop being true.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.classical import Budget, compare, generate_from_parameters
from hybrid_benchmarking.classical.synthesise import CAVEAT, build
from hybrid_benchmarking.instances import InstanceError
from hybrid_benchmarking.problems import BEGINNER, NOUNS, beginner_asks
from hybrid_benchmarking.web import beginner_form, compare_from_parameters

SAMPLE = {
    "maximum-flow": ("shift-assignment", {"things": 16, "links": 40}),
    "vertex-cover": ("camera-placement", {"things": 16, "links": 40}),
    "independent-set": ("transmitters", {"things": 16, "links": 40}),
    "clique": ("community", {"things": 12, "links": 40}),
    "linear-programming": ("blending", {"things": 24, "links": 8}),
    "knapsack": ("release-planning", {"things": 12, "budget": 100}),
    "quadratic-knapsack": ("charging-stations",
                           {"things": 10, "budget": 100, "pairs": 30}),
    "multidimensional-knapsack": ("cloud-packing",
                                  {"things": 12, "limits": 3, "budget": 100}),
    "linear-systems": ("heat-distribution", {"things": 30, "links": 4}),
}


class TestItAsksInTheProblemsOwnWords:
    def test_every_problem_has_nouns_of_its_own(self):
        for problem in hb.PROBLEMS:
            assert problem.key in NOUNS, problem.key

    def test_every_family_knows_what_to_ask_for(self):
        for problem in hb.PROBLEMS:
            assert problem.family in BEGINNER, problem.key

    @pytest.mark.parametrize("key", [p.key for p in hb.PROBLEMS])
    def test_no_question_leaks_a_placeholder_or_a_technical_name(self, key):
        for field in beginner_asks(key):
            assert "{a}" not in field.label and "{b}" not in field.label
            assert field.label.endswith("?")
            for jargon in ("vertex", "vertices", "sparsity", "condition number",
                           "kappa", "non-zero", "matrix", "knapsack", "LP"):
                assert jargon not in field.label.lower(), (key, field.label)

    def test_two_names_for_one_problem_ask_different_questions(self):
        # Which is the whole point of asking in the problem's own words.
        cameras = [f.label for f in beginner_asks("camera-placement")]
        monitors = [f.label for f in beginner_asks("network-monitors")]
        assert cameras != monitors

    def test_the_form_carries_the_routes_it_will_cost(self):
        form = beginner_form("shift-assignment")
        assert len(form["routes"]) == len(hb.get_problem("shift-assignment").routes)
        assert form["asks"]


class TestTheInstanceIsBuiltAndReal:
    @pytest.mark.parametrize("family", sorted(SAMPLE))
    def test_every_family_can_be_described_into_an_instance(self, family):
        key, values = SAMPLE[family]
        assert build(key, values) is not None

    @pytest.mark.parametrize("family", sorted(SAMPLE))
    def test_the_same_answers_give_the_same_instance_every_time(self, family):
        key, values = SAMPLE[family]
        first, second = build(key, values), build(key, values)
        assert first.describe() == second.describe()
        assert first == second

    @pytest.mark.parametrize("family", sorted(SAMPLE))
    def test_the_same_answers_give_the_same_number_every_time(self, family):
        key, values = SAMPLE[family]
        one = compare(key, values, budget=Budget(60))
        two = compare(key, values, budget=Budget(60))
        assert [r.get("total") for r in one["routes"]] == \
            [r.get("total") for r in two["routes"]]

    def test_a_bigger_description_costs_more(self):
        small = compare("camera-placement", {"things": 10, "links": 20},
                        budget=Budget(60))["routes"][0]["total"]
        large = compare("camera-placement", {"things": 20, "links": 60},
                        budget=Budget(60))["routes"][0]["total"]
        assert large > small

    def test_a_size_beyond_what_it_solves_says_so_rather_than_hanging(self):
        with pytest.raises(InstanceError, match="more than this tool solves"):
            build("camera-placement", {"things": 10 ** 6, "links": 10 ** 6})

    def test_nonsense_is_refused_by_name(self):
        with pytest.raises(InstanceError, match="whole number"):
            build("camera-placement", {"things": "lots", "links": 20})


class TestTheNumberSaysWhatItIs:
    @pytest.mark.parametrize("family", sorted(SAMPLE))
    def test_the_caveat_reaches_every_cost(self, family):
        key, values = SAMPLE[family]
        for entry in compare(key, values, budget=Budget(60))["routes"]:
            if "error" in entry:
                continue
            assert CAVEAT in entry["assumptions"], entry["route"]

    def test_it_survives_onto_the_run_itself(self):
        made = generate_from_parameters("camera-placement",
                                        {"things": 12, "links": 30},
                                        budget=Budget(60))
        assert CAVEAT in made.run.assumptions
        assert CAVEAT in made.data.generated["assumptions"]

    def test_the_caveat_says_it_is_not_your_data(self):
        assert "not read from your data" in CAVEAT

    def test_a_file_run_does_not_carry_it(self):
        # The caveat belongs to made-up instances alone; a real file is real.
        from hybrid_benchmarking.classical import cost, generate_from_file

        report = cost(generate_from_file("tests/fixtures/tiny.max",
                                         budget=Budget(60)))
        assert CAVEAT not in report["assumptions"]


class TestTheAnswerComesBeforeItsEvidence:
    """The chart is what somebody arriving with a problem actually asked for.

    They came to find out how long it would take. The counts are the evidence
    for that answer rather than the answer itself, and a row of totals in a
    unit they have never met is a poor first thing to meet. So the clock leads
    and the columns follow it, with only the caveat above both, since that is
    the difference between these numbers and an answer at all.
    """

    PAGE = (Path(hb.__file__).parent / "static" / "index.html").read_text()

    def _positions(self):
        chart = self.PAGE.index("html += timeChart(res.routes)")
        costs = self.PAGE.index("What each approach would cost")
        caveat = self.PAGE.index("This is a problem of your shape")
        return caveat, chart, costs

    def test_the_clock_comes_before_the_counts(self):
        caveat, chart, costs = self._positions()
        assert chart < costs, "the counts are being shown before the chart"

    def test_the_caveat_still_comes_before_both(self):
        caveat, chart, costs = self._positions()
        assert caveat < chart < costs

    def test_the_chart_is_rendered_once(self):
        assert self.PAGE.count("html += timeChart(res.routes)") == 1


class TestTheColumnsAreNotARace:
    def test_the_comparison_says_it_is_not_one(self):
        result = compare("shift-assignment", {"things": 16, "links": 40},
                         budget=Budget(60))
        assert result["comparable"] is False
        assert "never added or ranked" in result["why"]

    def test_the_units_are_reported_so_the_reader_can_see_they_differ(self):
        result = compare("shift-assignment", {"things": 16, "links": 40},
                         budget=Budget(60))
        assert set(result["units"]) == {"gates", "cycles"}

    def test_every_column_carries_its_own_bound_and_provenance(self):
        for entry in compare("camera-placement", {"things": 16, "links": 40},
                             budget=Budget(60))["routes"]:
            if "error" in entry:
                continue
            assert entry["bound"] and entry["provenance"] and entry["assumptions"]

    def test_the_totals_still_refuse_to_add(self):
        # Not merely a caption: the cost algebra will not do it either.
        from hybrid_benchmarking.cost import UnitMismatch, exact
        from hybrid_benchmarking.provenance import Unit

        with pytest.raises(UnitMismatch):
            exact(1, Unit.GATES) + exact(1, Unit.CYCLES)

    def test_one_instance_serves_every_column(self):
        # Two routes costed on two instances would differ for a reason nobody
        # asked about.
        result = compare("shift-assignment", {"things": 16, "links": 40},
                         budget=Budget(60))
        described = {r["instance"] for r in result["routes"] if "instance" in r}
        assert len(described) == 1


class TestOverTheInterface:
    def test_the_form_comes_back_in_plain_language(self):
        form = beginner_form("charging-stations")
        assert form["label"] == "Where to put charging points"
        assert any("candidate sites" in a["label"] for a in form["asks"])

    def test_the_comparison_leads_with_the_caveat(self):
        result = compare_from_parameters("charging-stations",
                                         {"things": 10, "budget": 100,
                                          "pairs": 30})
        assert result["caveat"] == CAVEAT
        assert result["snippet"].startswith("import hybrid_benchmarking")

    def test_the_snippet_reproduces_what_the_page_did(self):
        values = {"things": 12, "links": 30}
        shown = compare_from_parameters("camera-placement", values)
        namespace = {"hb": hb}
        exec("import hybrid_benchmarking as hb\nout = " +
             shown["snippet"].split("\n\n")[1], namespace)
        assert [r.get("total") for r in namespace["out"]["routes"]] == \
            [r.get("total") for r in shown["routes"]]


class TestTheSameAnswersGiveTheSameInstance:
    """Determinism has to survive leaving the process.

    It did not: the seed was ``hash`` of the answers, Python salts string
    hashing per process, and so the web server repeated itself while the
    command line did not.  A number that wobbles between two askings is worse
    than no number, because somebody will average them.
    """

    def test_within_one_process(self):
        from hybrid_benchmarking.classical.synthesise import build

        first = build("maximum-flow", {"things": "200", "links": "600"})
        second = build("maximum-flow", {"things": "200", "links": "600"})
        assert first.arcs == second.arcs

    def test_and_across_two_of_them(self):
        import subprocess
        import sys

        script = (
            "import warnings; warnings.filterwarnings('ignore');"
            "from hybrid_benchmarking.classical.synthesise import build;"
            "print(build('maximum-flow', {'things': '200', 'links': '600'})"
            ".arcs[:4])"
        )
        runs = {subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True).stdout
                for _ in range(3)}
        assert len(runs) == 1, runs

    def test_different_answers_give_different_instances(self):
        from hybrid_benchmarking.classical.synthesise import build

        first = build("maximum-flow", {"things": "200", "links": "600"})
        second = build("maximum-flow", {"things": "201", "links": "600"})
        assert first.arcs != second.arcs
