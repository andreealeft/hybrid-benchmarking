"""The works and the people the page credits.

A number here is somebody's lemma, and the page that hands it over says whose.
This holds the introduction's reference list to the registry: a routine whose
provenance names a source the page has never heard of is a credit that went
missing, and the point of the provenance is that this cannot happen quietly.
"""

from __future__ import annotations

import re
from pathlib import Path

import hybrid_benchmarking as hb


PAGE = (Path(hb.__file__).parent / "static" / "index.html").read_text()


def fold(text: str) -> str:
    """Compare names without arguing about diacritics.

    The registry writes Hoyer and Gilyen, the page writes Høyer and Gilyén.
    The page is right and the registry is ASCII; neither is a missing credit.
    """
    return (text.replace("ø", "o").replace("é", "e").replace("ü", "u")
                .replace("á", "a").replace("í", "i"))


FOLDED_PAGE = fold(PAGE)

#: Words that appear in a source string without identifying a work.
FURNITURE = {
    "Lemma", "Lemmas", "Appendix", "Chapter", "Theorem", "Every", "Costed",
    "Quantum", "Evolution", "Maximum", "Classical", "Lowest", "Market", "Run",
    "This", "The", "Both", "Where", "Fast", "Practical",
}


def significant(source: str):
    """The names in a provenance string.

    Capitalised words that are not scaffolding, plus any arXiv identifier.
    Hyphenated runs are split, because a source says Boyer-Brassard-Hoyer-Tapp
    where the page says Boyer, Brassard, Høyer and Tapp -- the same four people.
    """
    words = {word for word in re.findall(r"[A-Z][A-Za-zø]{3,}", source)
             if word not in FURNITURE}
    words |= set(re.findall(r"arXiv:[\w.\-/]+", source))
    return {fold(word) for word in words}


def sources():
    seen = set()
    for implementation in hb.all_implementations():
        for cost in implementation.costs.values():
            seen.update(cost.provenance.sources)
    return seen


def test_there_are_sources_to_check():
    assert len(sources()) > 15


def test_every_source_the_registry_names_is_credited_on_the_page():
    """Whoever a cost points at, the introduction has to name."""
    missing = []
    for source in sources():
        names = significant(source)
        if not names:
            # "Lemma 13 of the thesis", "Lemma 2 of the quantum breadth-first
            # search study" -- no name in the string, so it must at least point
            # at a work the page describes.
            lowered, page = source.lower(), PAGE.lower()
            assert any(phrase in lowered and phrase in page
                       for phrase in ("thesis", "breadth-first search",
                                      "interior point", "knapsack")), source
            continue
        if not any(name in FOLDED_PAGE for name in names):
            missing.append((source, sorted(names)))
    assert not missing, missing


def test_the_four_primary_studies_are_named():
    for work in ("Hybrid benchmarking of quantum algorithms",
                 "Quantum breadth-first search for maximum flow",
                 "Practical lower bounds for hybrid quantum interior point",
                 "quadratic and multidimensional\n        knapsack problems"):
        assert work in PAGE, work


def test_the_people_behind_them_are_named():
    for person in ("Lefterovici", "Lelakowski", "Perk", "Binkowski",
                   "Wilkening", "Funck", "Karimov", "Fekete", "Osborne"):
        assert person in PAGE, person


def test_the_repositories_are_marked_as_fixtures_not_ingredients():
    """Rule four: reimplement from lemmas, never vendor."""
    assert "Checked against, never copied from" in PAGE
    for repository in ("QUBRABENCH", "simplex-benchmarks", "qls-comparison",
                       "qipm"):
        assert repository in PAGE, repository


def test_the_classical_solvers_that_actually_run_are_credited_too():
    for name in ("Dinic", "Dantzig", "DIMACS", "Pisinger", "Matrix Market",
                 "MPS"):
        assert name in PAGE, name
