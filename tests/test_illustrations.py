"""The paragraph and the drawing each problem is introduced with.

These are the first things a reader sees, and they are the two places where
the catalogue's masking could be undone by accident: a story that names the
mathematics, or a drawing that admits two problems share a shape.  They are
also the two places a page that fetches nothing could quietly start fetching
something.  Hence the checking.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from hybrid_benchmarking import PROBLEMS, web
from hybrid_benchmarking.illustrations import described, picture, story

KEYS = [p.key for p in PROBLEMS]

#: Vocabulary that would tell a reader what their problem is called, which is
#: the classification the catalogue exists to spare them.
JARGON = ("graph", "vertex", "vertices", "edge", "matrix", "condition number",
          "sparsity", "knapsack", "clique", "independent set", "vertex cover",
          "linear program", "simplex", "interior point", "relaxation",
          "objective function", "oracle", "qubit", "quantum", "polynomial")

#: Anything that would make the page reach the network, run code, or break a
#: theme.
CONTRABAND = ("<script", "foreignobject", "<image", "http://", "https://",
              "stroke-dasharray", "<style", "<animate", "xlink:href", "rgb(",
              "hsl(")


class TestEveryProblemIsIntroduced:
    def test_all_of_them_have_both(self):
        assert described() == len(PROBLEMS)

    @pytest.mark.parametrize("key", KEYS)
    def test_this_one_does(self, key):
        assert story(key) and picture(key), key


class TestTheStories:
    @pytest.mark.parametrize("key", KEYS)
    def test_it_says_nothing_about_the_mathematics(self, key):
        lowered = story(key).lower()
        for word in JARGON:
            assert not re.search(r"\b" + re.escape(word) + r"\b", lowered), \
                (key, word)

    @pytest.mark.parametrize("key", KEYS)
    def test_it_keeps_the_house_punctuation(self, key):
        """Colons and commas: Andreea's rule, and it applies to generated copy
        exactly as it applies to copy written by hand."""
        text = story(key)
        for dash in ("--", "—", "–"):
            assert dash not in text, (key, dash)
        assert "!" not in text, key

    @pytest.mark.parametrize("key", KEYS)
    def test_it_is_a_paragraph_and_not_an_essay(self, key):
        text = story(key)
        assert 2 <= text.count(".") <= 6, (key, text.count("."))
        assert 120 <= len(text) <= 700, (key, len(text))

    @pytest.mark.parametrize("key", KEYS)
    def test_it_gives_the_reader_a_number_to_recognise(self, key):
        """A quantity is what makes somebody see their own situation in it."""
        assert re.search(r"\d", story(key)), key

    def test_no_two_problems_share_a_story(self):
        seen = {}
        for key in KEYS:
            seen.setdefault(story(key), []).append(key)
        assert not [keys for keys in seen.values() if len(keys) > 1]


class TestTheDrawings:
    @pytest.mark.parametrize("key", KEYS)
    def test_it_is_well_formed_and_titled(self, key):
        root = ET.fromstring(picture(key))
        assert root.tag.endswith("svg"), key
        assert "viewBox" in root.attrib, key
        assert any(child.tag.endswith("title") for child in root), key

    @pytest.mark.parametrize("key", KEYS)
    def test_it_fetches_nothing_and_runs_nothing(self, key):
        lowered = picture(key).lower()
        for token in CONTRABAND:
            assert token not in lowered, (key, token)

    @pytest.mark.parametrize("key", KEYS)
    def test_it_follows_the_page_theme(self, key):
        """Literal colours would be unreadable in one theme or the other."""
        drawing = picture(key)
        assert "var(--" in drawing, key
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", drawing), key

    @pytest.mark.parametrize("key", KEYS)
    def test_it_stays_small_enough_to_ship(self, key):
        assert len(picture(key)) <= 3200, (key, len(picture(key)))

    def test_no_two_problems_share_a_drawing(self):
        """Seventy-one problems, seventy-one pictures: the whole point of
        drawing them one at a time rather than once per family."""
        seen = {}
        for key in KEYS:
            seen.setdefault(picture(key), []).append(key)
        assert not [keys for keys in seen.values() if len(keys) > 1]


class TestThePageReceivesThem:
    def test_the_problem_endpoint_carries_the_story_and_the_drawing(self):
        detail = web.problem_detail("shift-assignment")
        assert detail["story"] and detail["picture"].startswith("<svg")

    def test_the_menu_does_not(self):
        """The list stays text: seventy-one thumbnails would make it slower to
        scan, not faster."""
        assert all("picture" not in entry for entry in web.problems())
