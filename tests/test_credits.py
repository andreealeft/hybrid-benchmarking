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


def test_the_primary_studies_are_named():
    """Including the three the thesis is made of, and the paper the tree
    generator itself came from -- which is not the same paper as the variants
    that extend it."""
    for work in ("Hybrid benchmarking of quantum algorithms",
                 "Realistic runtime analysis for quantum simplex computation",
                 "comparing functional quantum linear\n        solvers",
                 "A quantum algorithm for solving 0-1 knapsack problems",
                 "quadratic and multidimensional\n        knapsack problems",
                 "Quantum breadth-first search for maximum flow",
                 "Practical lower bounds for hybrid quantum interior point"):
        assert work in PAGE, work


def test_the_identifiers_the_thesis_bibliography_gives():
    for identifier in ("arXiv:2311.09995", "arXiv:2503.21420",
                       "arXiv:2503.22325", "arXiv:2604.24362",
                       "npj Quantum Information 11, 146, 2025"):
        assert identifier in PAGE, identifier


def test_the_people_behind_them_are_named():
    """Everyone on the author list of a work this library reimplements."""
    for person in ("Ammann", "Binkowski", "Fekete", "Funck", "Goedicke",
                   "Gross", "Hess", "Karimov", "Lefterovici", "Lelakowski",
                   "Osborne", "Perk", "Ramacciotti", "Rotundo", "Skelton",
                   "Steinbach", "Stiller", "Wilkening", "de Wolff"):
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
