"""The local interface and the command line.

The interface is meant to hold no logic of its own, so most of what is worth
testing is that it faithfully relays the library -- including the parts that
refuse, warn, or explain themselves.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import hybrid_benchmarking as hb
from hybrid_benchmarking.cli import main
from hybrid_benchmarking.provenance import Unit
from hybrid_benchmarking.web import (
    catalogue,
    evaluate,
    serve,
    snippet,
    why_not,
)


class TestNothingLeavesTheMachine:
    def test_the_page_has_no_external_resources(self):
        """Someone working offline must get the same interface as everyone
        else, so nothing may be fetched from anywhere."""
        page = (Path(hb.__file__).parent / "static" / "index.html").read_text()
        for marker in ("http://", "https://", "//cdn", "@import"):
            assert marker not in page.replace("http://127.0.0.1", "")

    def test_it_binds_the_loopback_interface(self):
        url, httpd = serve(port=0, open_browser=False)
        try:
            assert url.startswith("http://127.0.0.1:")
        finally:
            httpd.server_close()


class TestTheCatalogue:
    def test_every_routine_appears(self):
        assert {entry["name"] for entry in catalogue()} == set(hb.names())

    def test_entries_carry_their_units_and_constructions(self):
        by_name = {entry["name"]: entry for entry in catalogue()}
        assert by_name["HamSim"]["constructions"] == ["qubitization", "berry"]
        assert by_name["QSearch"]["units"] == ["ITERATIONS"]


class TestItExplainsRatherThanGreysOut:
    def test_a_missing_gate_count_says_why(self):
        reason = why_not(hb.get("QLS-Chebyshev"), Unit.GATES)
        assert "oracle implementation is not fixed" in reason

    def test_a_missing_cycle_count_says_why(self):
        reason = why_not(hb.get("IsOptimal"), Unit.CYCLES)
        assert "no schedule" in reason or "bound" in reason

    def test_an_oracle_says_it_is_the_unit_not_a_cost(self):
        assert "unit other costs are counted in" in why_not(hb.get("O_F"),
                                                            Unit.QUERIES)


class TestEvaluation:
    def test_it_relays_a_value_and_its_provenance(self):
        result = evaluate("QSearch", "ITERATIONS", {"X": "1000000", "t": "1"})
        assert result["value"] == pytest.approx(
            hb.get("QSearch").evaluate(Unit.ITERATIONS, X=1e6, t=1).value
        )
        assert "exact" in result["provenance"]

    def test_vectors_arrive_as_json(self):
        """Some costs depend on the instance, not on summary statistics."""
        result = evaluate("QTG", "GATES", {
            "profits": "[6, 2, 1, 2]", "weights": "[2, 2, 1, 5]",
            "capacity": "7", "profit_bound": "11",
        })
        assert result["value"] > 0

    def test_nonsense_input_is_reported_not_guessed(self):
        with pytest.raises(ValueError, match="neither a number nor valid JSON"):
            evaluate("QSearch", "ITERATIONS", {"X": "lots", "t": "1"})

    def test_leaving_the_regime_warns_rather_than_refusing(self):
        result = evaluate("QLS-Chebyshev", "QUERIES", {
            "d": "4", "kappa": "10", "epsilon": "0.5", "x_norm": "1",
        })
        assert result["value"] > 0
        assert any("derived regime" in w for w in result["warnings"])

    def test_meaningless_input_is_refused(self):
        with pytest.raises(ValueError, match="no meaning"):
            evaluate("QSearch", "ITERATIONS", {"X": "10", "t": "50"})


class TestCopyAsCode:
    def test_the_snippet_reproduces_the_number(self):
        """Anything clicked here has to be pasteable into a script -- that is
        what makes 'too large for the packaged tool' a command, not an email."""
        result = evaluate("HamSim/berry", "GATES", {
            "d": "4", "A_max": "1", "A_1": "3", "t_sim": "10", "epsilon": "1e-3",
        })
        namespace = {}
        exec(compile(result["snippet"].replace("hb.get", "value = hb.get"),
                     "<snippet>", "exec"), namespace)
        assert namespace["value"].value == pytest.approx(result["value"])

    def test_it_names_the_implementation_when_there_are_several(self):
        text = snippet("HamSim/berry", "GATES", {"d": 4})
        assert "'HamSim/berry'" in text
        assert "hb.Unit.GATES" in text


class TestOverHttp:
    @classmethod
    def setup_class(cls):
        cls.url, cls.httpd = serve(port=0, open_browser=False)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def teardown_class(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path):
        with urllib.request.urlopen(self.url.rstrip("/") + path) as response:
            return response.status, response.read()

    def _post(self, payload):
        request = urllib.request.Request(
            self.url.rstrip("/") + "/api/evaluate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_the_page_is_served(self):
        status, body = self._get("/")
        assert status == 200 and b"hybrid-benchmarking" in body

    def test_the_catalogue_is_served(self):
        status, body = self._get("/api/routines")
        assert status == 200 and len(json.loads(body)) == len(hb.names())

    def test_a_routine_with_two_constructions_reports_both(self):
        status, body = self._get("/api/routine/QLS-Fourier")
        data = json.loads(body)
        assert {i["name"] for i in data["implementations"]} == {
            "via-qubitization", "via-berry"
        }
        assert "CYCLES" in data["missing"]

    def test_an_unknown_routine_is_a_404(self):
        with pytest.raises(urllib.error.HTTPError) as caught:
            self._get("/api/routine/Nonesuch")
        assert caught.value.code == 404

    def test_evaluation_round_trips(self):
        status, result = self._post({
            "path": "QSearch", "unit": "ITERATIONS",
            "values": {"X": "1000000", "t": "1"},
        })
        assert status == 200
        assert result["value"] == pytest.approx(1411.13, rel=1e-3)

    def test_a_refusal_comes_back_as_a_client_error(self):
        status, result = self._post({
            "path": "QSearch", "unit": "ITERATIONS",
            "values": {"X": "10", "t": "50"},
        })
        assert status == 400 and "no meaning" in result["error"]


class TestCommandLine:
    def test_list_prints_the_capability_table(self, capsys):
        assert main(["list"]) == 0
        assert "QLS-Chebyshev" in capsys.readouterr().out

    def test_show_prints_every_construction(self, capsys):
        assert main(["show", "HamSim"]) == 0
        out = capsys.readouterr().out
        assert "qubitization" in out and "berry" in out

    def test_show_explains_the_units_a_routine_lacks(self, capsys):
        assert main(["show", "QLS-Chebyshev"]) == 0
        assert "oracle implementation is not fixed" in capsys.readouterr().out

    def test_cost_evaluates(self, capsys):
        assert main(["cost", "QSearch", "-u", "ITERATIONS",
                     "-p", "X=1000000", "-p", "t=1"]) == 0
        assert "1411" in capsys.readouterr().out

    def test_cost_reports_a_refusal_and_fails(self, capsys):
        assert main(["cost", "QSearch", "-u", "ITERATIONS",
                     "-p", "X=10", "-p", "t=50"]) == 1
        assert "no meaning" in capsys.readouterr().err

    def test_formula_prints_the_expression(self, capsys):
        assert main(["formula", "QFT", "-u", "GATES"]) == 0
        assert "bits" in capsys.readouterr().out

    def test_badly_formed_parameters_are_rejected(self):
        with pytest.raises(SystemExit, match="name=value"):
            main(["cost", "QSearch", "-p", "X"])
