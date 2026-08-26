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
    ("Dalzell", "A shortcut to an optimal quantum linear system solver"),
    ("Binkowski", "Practical lower bounds for hybrid quantum interior point"),
    ("Wilkening", "A quantum search method for quadratic and multidimensional"),
    ("breadth-first search", "Quantum breadth-first search for maximum flow"),
    ("thesis", "Hybrid benchmarking of quantum algorithms"),
)


def works_for(source: str):
    return [title for fragment, title in WORKS if fragment in source]


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
    orphans, unlisted = [], []
    for source in sources():
        titles = works_for(source)
        if not titles:
            orphans.append(source)
            continue
        if not any(" ".join(title.split()) in flat for title in titles):
            unlisted.append((source, titles))
    assert not orphans, orphans
    assert not unlisted, unlisted


def test_the_authors_are_off_the_page_but_not_out_of_the_provenance():
    """Andreea asked for the papers alone on the page.

    The credit did not go anywhere: every cost still carries the names in its
    provenance, so a number handed to somebody still arrives with the people who
    proved it attached.  That is the copy that matters and it is tested here.
    """
    named = {name for source in sources() for name in
             ("Nannicini", "Cade", "Binkowski", "Wilkening", "Lefterovici",
              "Dalzell", "Brassard") if name in source}
    assert len(named) >= 5, named
    for name in ("Ammann", "Goedicke", "de Wolff", "Ramacciotti", "Steinbach"):
        assert name not in PAGE, name


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
                                   "https://ieeexplore.ieee.org/")), address


def test_the_classical_solvers_that_actually_run_are_credited_too():
    for name in ("Dinic", "simplex method", "DIMACS", "Pisinger",
                 "Matrix Market", "MPS"):
        assert name in PAGE, name
