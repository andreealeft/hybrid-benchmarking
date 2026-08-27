"""The published results, redrawn for the front page.

A chart is a number wearing a picture, so it falls under the same rule as every
other number here: it says where it came from, and nothing is invented to make
it look complete.  What is tested is that each figure declares which kind it is,
that a schematic admits to being one, that a redrawing states its method, and
that the drawing itself fetches nothing and follows the page theme.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from hybrid_benchmarking import figures as F
from hybrid_benchmarking import web

FIGURES = F.all_figures()
KEYS = [entry["key"] for entry in FIGURES]

CONTRABAND = ("<script", "foreignobject", "<image", "http://", "https://",
              "<style", "<animate", "xlink:href", "rgb(", "hsl(")


def test_every_study_the_library_reimplements_has_one():
    assert set(KEYS) == set(F.ORDER)


def test_they_come_back_in_the_order_of_the_thesis():
    assert KEYS == list(F.ORDER)


def _is_a_key(node) -> bool:
    """A legend swatch or a reference rule, rather than a mark carrying data."""
    tag = node.tag.split("}")[-1]
    if tag == "line":
        return True
    if tag != "rect":
        return False
    try:
        return (float(node.get("width", 0)) <= 20
                and float(node.get("height", 0)) <= 20)
    except ValueError:
        return False


@pytest.mark.parametrize("entry", FIGURES, ids=KEYS)
class TestEachFigure:
    def test_it_says_which_kind_it_is(self, entry):
        assert entry["kind"] in ("redrawn", "schematic")

    def test_a_schematic_admits_it_in_the_first_word(self, entry):
        """So that nobody reads the shape of a result as its values."""
        if entry["kind"] == "schematic":
            assert entry["caption"].lower().startswith("schematic")

    def test_a_redrawing_states_how_it_was_obtained(self, entry):
        if entry["kind"] == "redrawn":
            assert len(entry.get("method", "")) > 40, entry["key"]

    def test_it_links_to_the_paper(self, entry):
        assert entry["paper"] and entry["paper_url"].startswith("https://")

    def test_it_is_a_chart_and_not_a_picture(self, entry):
        """Axes and labels: an illustration would not need this test, but a
        figure claiming to show results does."""
        root = ET.fromstring(entry["figure"])
        labels = [node for node in root.iter() if node.tag.endswith("text")]
        assert len(labels) >= 4, (entry["key"], len(labels))

    def test_it_is_well_formed_and_titled(self, entry):
        root = ET.fromstring(entry["figure"])
        assert root.tag.endswith("svg")
        assert "viewBox" in root.attrib
        assert any(child.tag.endswith("title") for child in root)

    def test_it_fetches_nothing_and_runs_nothing(self, entry):
        lowered = entry["figure"].lower()
        for token in CONTRABAND:
            assert token not in lowered, (entry["key"], token)

    def test_it_follows_the_page_theme(self, entry):
        assert "var(--" in entry["figure"]
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", entry["figure"])

    def test_the_prose_keeps_the_house_punctuation(self, entry):
        for field in ("title", "caption", "method"):
            text = entry.get(field, "")
            for dash in ("--", "—", "–"):
                assert dash not in text, (entry["key"], field)

    def test_a_dashed_stroke_is_only_ever_a_reference_line(self, entry):
        assert entry["figure"].lower().count("stroke-dasharray") <= 2

    def test_a_key_is_named_in_its_own_colour(self, entry):
        """Whatever introduces a coloured mark is written in that mark's
        colour, so that nothing ties a name to a line by position alone.

        This fault has been found and fixed three times in four figures, each
        time by rendering and looking, because markup where a legend reads
        "Dinic" in plain ink beside a rule that is orange parses exactly as
        well as markup where it does not.  A reader who cannot tell which
        curve is which is reading a picture, not a result.

        What counts as a key is a legend swatch or a reference rule -- a short
        line, or a rect small enough to be a marker.  The data marks
        themselves are excluded: the bar chart's group headings follow its
        last bar in document order and are headings, not series names.
        """
        series = ("var(--accent)", "var(--good)", "var(--warn)")
        nodes = list(ET.fromstring(entry["figure"]))
        for mark, label in zip(nodes, nodes[1:]):
            colour = mark.get("fill") or mark.get("stroke")
            if colour not in series or not _is_a_key(mark):
                continue
            if label.tag.split("}")[-1] != "text":
                continue
            assert label.get("fill") == colour, (
                entry["key"], colour, (label.text or "").strip())


class TestThePageReceivesThem:
    def test_the_endpoint_serves_them(self):
        assert len(web.figures()) == len(F.ORDER)

    def test_the_front_page_has_somewhere_to_put_them(self):
        from pathlib import Path

        import hybrid_benchmarking as hb

        page = (Path(hb.__file__).parent / "static" / "index.html").read_text()
        assert 'id="findings"' in page
        assert "What these studies found" in page
