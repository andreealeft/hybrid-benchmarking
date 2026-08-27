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

#: Which listed work each provenance source belongs to.  The page names the
#: papers and no longer names their authors, so the correspondence has to be
#: written down: a distinctive fragment of the source string, and a fragment of
#: the title it points at.  A source matching nothing here is a citation with
#: no work behind it, which is what this table exists to catch.
WORKS = (
    ("Nannicini", "Fast quantum subroutines for the simplex method"),
    ("Boyer-Brassard-Hoyer-Tapp", "Tight bounds on quantum searching"),
    ("Brassard, Hoyer, Mosca and Tapp",
     "Quantum amplitude amplification and estimation"),
    ("Harrow-Hassidim-Lloyd", "Quantum algorithm for linear systems of equations"),
    ("Childs-Kothari-Somma", "exponentially improved dependence on precision"),
    ("Gilyen", "Quantum singular value transformation and beyond"),
    ("Low-Chuang", "Hamiltonian simulation by qubitization"),
    ("Berry et al.", "Exponential improvement in precision for simulating"),
    ("Cade, Folkertsma", "Quantifying Grover speed-ups beyond asymptotic"),
    ("Binkowski", "Practical lower bounds for hybrid quantum interior point"),
    ("Wilkening", "A quantum search method for quadratic and multidimensional"),
    ("breadth-first search",
     "benchmarking quantum breadth-first search for maximum flow problems"),
    ("thesis", "Hybrid benchmarking of quantum algorithms"),
)


#: Sources whose work is deliberately absent from the page.  The routine stays
#: in the registry and still cites it, so the credit survives on the number
#: even where the reference list no longer carries it.
UNLISTED = {
    "Dalzell": "dropped from the reference list at Andreea's request; "
               "Cade-linalg still implements it and still cites it",
}


def works_for(source: str):
    return [title for fragment, title in WORKS if fragment in source]


def unlisted(source: str):
    return [why for name, why in UNLISTED.items() if name in source]


def sources():
    seen = set()
    for implementation in hb.all_implementations():
        for cost in implementation.costs.values():
            seen.update(cost.provenance.sources)
    return seen


def test_there_are_sources_to_check():
    assert len(sources()) > 15


def test_every_source_the_registry_names_belongs_to_a_listed_work():
    """Whatever a cost points at, the introduction has to list.

    Adding a routine that cites something new fails here until the work goes on
    the page -- which is the only thing keeping the list true as the registry
    grows.
    """
    flat = " ".join(PAGE.split())
    orphans, absent = [], []
    for source in sources():
        if unlisted(source):
            continue
        titles = works_for(source)
        if not titles:
            orphans.append(source)
            continue
        if not any(" ".join(title.split()) in flat for title in titles):
            absent.append((source, titles))
    assert not orphans, orphans
    assert not absent, absent


def test_the_authors_are_named_on_the_page_and_in_the_provenance():
    """Both copies matter.

    The page credits whoever wrote the work; the provenance credits whoever
    proved the result, and travels with the number after it leaves the page.
    """
    for name in ("Ammann", "Binkowski", "Dantzig", "Dinitz",
                 "Fekete", "Funck", "Goedicke", "Gross", "Hess", "Karimov",
                 "Lefterovici", "Lelakowski", "Nannicini", "Osborne", "Perk",
                 "Ramacciotti", "Rotundo", "Skelton", "Steinbach", "Stiller",
                 "Wilkening", "de Wolff"):
        assert name in PAGE, name

    named = {name for source in sources() for name in
             ("Nannicini", "Cade", "Binkowski", "Wilkening", "Lefterovici",
              "Dalzell", "Brassard") if name in source}
    assert len(named) >= 5, named


def test_an_unlisted_work_is_still_cited_by_the_routine_that_uses_it():
    """An exemption that stopped being true would quietly become a gap.

    Dalzell is off the reference list by request, and the entry that
    implements that result still names it, which is where the credit has to
    survive.
    """
    for name in UNLISTED:
        assert any(name in source for source in sources()), name
        assert name not in PAGE, name


def test_the_primary_studies_are_named():
    """Titles as their authors publish them.

    Matched against the flattened page, because a title that carries a link
    wraps across lines inside its own anchor.
    """
    """Including the three the thesis is made of, and the paper the tree
    generator itself came from -- which is not the same paper as the variants
    that extend it."""
    flat = " ".join(PAGE.split())
    for work in ("Hybrid benchmarking of quantum algorithms",
                 "Realistic runtime analysis for quantum simplex computation",
                 "comparing functional quantum linear solvers",
                 "A quantum algorithm for solving 0-1 Knapsack problems",
                 "quadratic and multidimensional knapsack problems",
                 "benchmarking quantum breadth-first search for maximum "
                 "flow problems",
                 "Practical lower bounds for hybrid quantum interior point"):
        assert " ".join(work.split()) in flat, work


def test_a_preprint_is_cited_by_its_arxiv_number():
    for identifier in ("arXiv:2311.09995", "arXiv:2604.24362",
                       "arXiv:2604.24962"):
        assert identifier in PAGE, identifier


def test_a_published_work_is_cited_by_its_journal_and_not_by_arxiv():
    """Once a paper is out, the preprint number is noise beside the citation."""
    for journal in ("IEEE Transactions on Quantum Engineering",
                    "npj Quantum Information 11, 146, 2025",
                    "IEEE Quantum Computing and Engineering",
                    "Operations Research", "Quantum 7, 1133, 2023"):
        assert journal in PAGE, journal
    for superseded in ("arXiv:2503.21420", "arXiv:2503.22325"):
        assert superseded not in PAGE, superseded


def test_the_page_says_the_results_are_reimplemented_not_vendored():
    """Rule four, stated where a reader can see it.

    The list of reference repositories was dropped from the page at Andreea's
    request; the claim it was there to support is made in the lead instead, and
    QUBRABENCH is still credited beside the result it contributes.
    """
    flat = " ".join(PAGE.split())
    assert "reimplemented from the published statements" in flat
    assert "used to check the results against, never copied from" in flat
    assert "QUBRABENCH" in PAGE


def test_every_cited_work_that_has_an_address_is_linked():
    """A citation somebody has to retype is half a citation."""
    linked = re.findall(r'<a href="(https?://[^"]+)"[^>]*>(?:(?!</a>).)*</a>',
                        PAGE, flags=re.S)
    assert len(linked) >= 15, linked
    for address in linked:
        assert address.startswith(("https://arxiv.org/", "https://doi.org/",
                                   "https://www.nature.com/",
                                   "https://repo.uni-hannover.de/",
                                   "https://ieeexplore.ieee.org/")), address


def test_the_classical_solvers_that_actually_run_are_credited_too():
    for name in ("Dinic", "simplex method", "DIMACS", "Pisinger",
                 "Matrix Market", "MPS"):
        assert name in PAGE, name
