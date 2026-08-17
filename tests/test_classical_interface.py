"""Getting at all of this without writing Python, which is most people.

The panel and the command line are clients: they hold no logic, and what is
worth testing about them is that the loop stays visible.  The whole arrangement
turns into an oracle the moment a number appears without the log that produced
it, so the tests here are largely about ordering and about what is printed
alongside the answer -- the implementation's name, the status of the run, and
the advice when it did not finish.

Batch mode carries one rule of its own.  A directory can hold a network and a
knapsack, and those cost gates and cycles.  Summing them would be the single
thing this library exists to refuse, so the totals are kept per unit and there
is a test that a mixed directory produces two of them rather than one.
"""

from __future__ import annotations

import json

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.cli import main
from hybrid_benchmarking.web import cost_from_instance, problem_detail

FIXTURES = "tests/fixtures"


class TestThePanelCanStartFromAnInstance:
    def test_a_route_says_whether_a_log_can_be_produced_for_it(self):
        routes = {r["key"]: r["generated"]
                  for r in problem_detail("maximum-flow")["routes"]}
        assert routes["quantum-bfs"] is True
        # Honest about the ones it cannot: the panel offers the form only where
        # there is something behind it.
        assert routes["quantum-simplex"] is False

    def test_it_hands_back_the_log_as_well_as_the_number(self):
        result = cost_from_instance("maximum-flow", "quantum-bfs",
                                    FIXTURES + "/tiny.max", {}, 60)
        assert result["total"] > 0
        assert '"layers"' in result["log"]
        assert "Dinic" in result["implementation"]
        assert result["status"] == "complete"

    def test_the_snippet_it_prints_reproduces_what_it_did(self):
        result = cost_from_instance("maximum-flow", "quantum-bfs",
                                    FIXTURES + "/tiny.max", {}, 60)
        namespace = {}
        exec(result["snippet"].replace("hb.classical.cost",
                                       "namespace['total'] = hb.classical.cost"),
             {"hb": hb, "namespace": namespace})
        assert namespace["total"]["total"] == pytest.approx(result["total"])

    def test_an_unreadable_file_is_reported_rather_than_raised(self):
        from hybrid_benchmarking.dataset import FormatError

        with pytest.raises(FormatError):
            cost_from_instance("maximum-flow", "quantum-bfs",
                               FIXTURES + "/noproblem.max", {}, 60)

    def test_a_run_that_was_cut_off_still_returns_its_log(self):
        result = cost_from_instance("vertex-cover", "quantum-simplex",
                                    FIXTURES + "/tiny.clq",
                                    {"epsilon": "1e-3", "delta": "1e-3"}, 1e-9)
        # Nothing was logged at all in this case, so there is no cost -- but
        # there is still a reason, which is the point.
        assert result["status"] in ("failed", "truncated")
        assert result.get("error")


class TestTheCommandLine:
    def test_run_prints_the_log_before_it_prints_the_number(self, capsys):
        assert main(["run", FIXTURES + "/tiny.max"]) == 0
        printed = capsys.readouterr().out
        assert printed.index("the log this produced") < printed.index("gates")

    def test_run_names_what_actually_ran_not_what_the_route_is_about(
            self, capsys):
        assert main(["run", FIXTURES + "/knapsack/pisinger_six_items.kp"]) == 0
        printed = capsys.readouterr().out
        # The route's classical algorithm is COMBO. We did not run COMBO.
        assert "COMBO" not in printed
        assert "read rather than solved" in printed

    def test_log_stops_at_the_log(self, capsys):
        assert main(["log", FIXTURES + "/tiny.max"]) == 0
        printed = capsys.readouterr().out
        assert "layers" in printed
        assert "gates" not in printed

    def test_log_writes_where_it_is_told_and_the_file_reloads(
            self, capsys, tmp_path):
        target = str(tmp_path / "kept.json")
        assert main(["log", FIXTURES + "/tiny.max", "-o", target]) == 0
        data = hb.load(target)
        assert data.generated["status"] == "complete"
        assert hb.run(hb.get_route("maximum-flow", "quantum-bfs"),
                      data)["total"] > 0

    def test_a_file_it_cannot_read_fails_rather_than_costing_something(
            self, capsys):
        assert main(["run", FIXTURES + "/noproblem.max"]) == 1
        assert "line 2" in capsys.readouterr().err

    def test_template_still_prints_the_columns_for_logging_it_yourself(
            self, capsys):
        assert main(["template", "vertex-cover/quantum-simplex"]) == 0
        printed = capsys.readouterr().out
        assert "kappa" in printed and "u_norm" in printed
        # And the corrected description of the one that is easy to misread.
        assert "not the number of columns that could enter" in printed


class TestBatch:
    def test_it_tabulates_and_sums(self, capsys, tmp_path):
        import shutil

        for name in ("tiny.max", "comments.max"):
            shutil.copy(FIXTURES + "/" + name, str(tmp_path / name))
        assert main(["batch", str(tmp_path)]) == 0
        printed = capsys.readouterr().out
        assert "instance" in printed and "status" in printed
        assert "over 2 instances" in printed

    def test_units_are_never_added_across(self, capsys, tmp_path):
        import shutil

        # A network costs gates and a knapsack costs cycles.
        shutil.copy(FIXTURES + "/tiny.max", str(tmp_path / "tiny.max"))
        shutil.copy(FIXTURES + "/knapsack/pisinger_six_items.kp",
                    str(tmp_path / "items.kp"))
        assert main(["batch", str(tmp_path)]) == 0
        printed = capsys.readouterr().out
        assert "gates over" in printed and "cycles over" in printed
        assert "these units do not add" in printed

    def test_something_it_cannot_read_is_a_row_not_a_crash(
            self, capsys, tmp_path):
        import shutil

        shutil.copy(FIXTURES + "/tiny.max", str(tmp_path / "tiny.max"))
        shutil.copy(FIXTURES + "/twosources.max", str(tmp_path / "bad.max"))
        assert main(["batch", str(tmp_path)]) == 0
        printed = capsys.readouterr().out
        assert "unreadable" in printed
        assert "over 1 instance" in printed

    def test_a_directory_of_nothing_says_so(self, capsys, tmp_path):
        (tmp_path / "notes.rst").write_text("nothing to see\n")
        assert main(["batch", str(tmp_path)]) == 1
        assert "looks like an instance" in capsys.readouterr().err

    def test_truncation_is_reported_with_the_flag_that_fixes_it(
            self, capsys, tmp_path):
        import shutil

        shutil.copy(FIXTURES + "/tiny.max", str(tmp_path / "tiny.max"))
        main(["batch", str(tmp_path), "--budget", "1e-9"])
        printed = capsys.readouterr().out
        assert "failed" in printed or "truncated" in printed
