"""The two DIMACS readers.

What is asserted here is what the format says, not what the reader happens to
do: that node numbers are one-based in the file and zero-based afterwards, that
an undirected edge is the same edge whichever way round it is written, that a
header is a claim about the file rather than the file itself -- and, above all,
that a line nobody understood stops the read.  A misread instance still produces
a graph, and a graph still produces a cost, which is the failure this package is
built to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_benchmarking.instances import Graph, InstanceError, Network
from hybrid_benchmarking.instances.dimacs import (
    parse_graph,
    parse_max_flow,
    read_graph,
    read_max_flow,
)

FIXTURES = Path(__file__).parent / "fixtures"

#: tiny.max, checked by hand: four nodes, five arcs, source 1 and sink 4 in the
#: file's numbering, so vertices 0 and 3 in ours.
TINY_ARCS = (
    (0, 1, 3.0),
    (0, 2, 2.0),
    (1, 2, 1.0),
    (1, 3, 2.0),
    (2, 3, 4.0),
)

#: tiny.clq: a triangle on 1, 2, 3 with 4 hanging off 3.
TINY_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3))


class TestAHandCheckedNetwork:
    """tiny.max, read against the file rather than against the code."""

    def test_it_holds_the_four_nodes_and_five_arcs_the_file_writes(self):
        network = read_max_flow(FIXTURES / "tiny.max")
        assert isinstance(network, Network)
        assert network.vertices == 4
        assert network.arcs == TINY_ARCS

    def test_the_designated_nodes_become_the_source_and_the_sink(self):
        """'n 1 s' and 'n 4 t', one-based, are vertices 0 and 3."""
        network = read_max_flow(FIXTURES / "tiny.max")
        assert network.source_vertex == 0
        assert network.sink_vertex == 3

    def test_every_arc_points_at_a_vertex_that_exists(self):
        network = read_max_flow(FIXTURES / "tiny.max")
        for tail, head, _ in network.arcs:
            assert 0 <= tail < network.vertices
            assert 0 <= head < network.vertices

    def test_it_says_where_it_came_from_and_which_reader_made_it(self):
        path = FIXTURES / "tiny.max"
        network = read_max_flow(path)
        assert network.name == "tiny"
        assert network.source == str(path)
        assert network.layout == "dimacs-max"

    def test_comments_blank_lines_and_a_missing_last_newline_change_nothing(
        self,
    ):
        """comments.max is tiny.max with everything the format permits done to
        it: comments before the problem line and between the arcs, indented
        lines, a capacity spelled 3.0, and no newline at the end."""
        plain = read_max_flow(FIXTURES / "tiny.max")
        awkward = read_max_flow(FIXTURES / "comments.max")
        assert awkward.arcs == plain.arcs
        assert awkward.vertices == plain.vertices
        assert awkward.source_vertex == plain.source_vertex
        assert awkward.sink_vertex == plain.sink_vertex

    def test_parallel_arcs_are_both_kept(self):
        """A flow solver is entitled to see two arcs where the file wrote two;
        merging them would change the capacity it may push."""
        network = parse_max_flow(
            "p max 2 2\nn 1 s\nn 2 t\na 1 2 3\na 1 2 4\n"
        )
        assert network.arcs == ((0, 1, 3.0), (0, 1, 4.0))

    def test_a_capacity_of_zero_is_still_an_arc(self):
        """Zero is a capacity, not an absence: the arc exists and may carry
        flow once the file is a residual network."""
        network = parse_max_flow("p max 2 1\nn 1 s\nn 2 t\na 1 2 0\n")
        assert network.arcs == ((0, 1, 0.0),)


class TestAnUndirectedGraph:
    def test_edges_written_once_read_as_the_file_writes_them(self):
        graph = read_graph(FIXTURES / "tiny.clq")
        assert isinstance(graph, Graph)
        assert graph.vertices == 4
        assert graph.edges == TINY_EDGES

    def test_writing_both_directions_gives_the_identical_graph(self):
        """The two conventions describe one object, so they must read as one --
        including the repeated edge and the self-loop tiny_both.col adds, which
        an undirected graph without weights cannot hold either."""
        once = parse_graph(
            (FIXTURES / "tiny.clq").read_text(), name="tiny", source="here"
        )
        twice = parse_graph(
            (FIXTURES / "tiny_both.col").read_text(), name="tiny", source="here"
        )
        assert twice == once

    def test_the_same_format_answers_to_all_of_its_names(self):
        """Collections spell the problem line edge, edges, col, clq and clique,
        in either case, for the one format."""
        graphs = [
            parse_graph("p {} 3 2\ne 1 2\ne 2 3\n".format(word))
            for word in ("edge", "edges", "col", "clq", "clique", "EDGE")
        ]
        assert all(graph == graphs[0] for graph in graphs)
        assert graphs[0].edges == ((0, 1), (1, 2))

    def test_a_weight_on_an_edge_is_dropped_rather_than_refused(self):
        """Weighted edge lists are read as the graphs underneath them: the
        third field is ignored, whatever it says, and an unweighted graph has
        nowhere to record it."""
        graph = read_graph(FIXTURES / "weighted.edges")
        assert graph.vertices == 5
        assert graph.edges == ((0, 1), (1, 2), (2, 3), (3, 4))

    def test_edges_come_out_sorted_and_with_the_smaller_end_first(self):
        graph = parse_graph("p edge 4 3\ne 4 2\ne 3 1\ne 2 1\n")
        assert graph.edges == ((0, 1), (0, 2), (1, 3))
        assert list(graph.edges) == sorted(graph.edges)
        assert all(u < v for u, v in graph.edges)

    def test_it_says_where_it_came_from_and_which_reader_made_it(self):
        path = FIXTURES / "tiny.clq"
        graph = read_graph(path)
        assert graph.name == "tiny"
        assert graph.source == str(path)
        assert graph.layout == "dimacs-edge"


class TestTheHeaderIsAClaimAboutTheFile:
    """Collected DIMACS files disagree with their own problem lines constantly.
    The data is the instance; the header is believed only as far as it can be a
    stale count of this file."""

    def test_ids_above_the_declared_count_widen_the_graph(self):
        """stalecount.clq declares six vertices and four edges and holds eight
        of each.  Dropping the vertices an edge names is not a reading of the
        file."""
        graph = read_graph(FIXTURES / "stalecount.clq")
        assert graph.vertices == 8
        assert len(graph.edges) == 8
        assert (6, 7) in graph.edges

    def test_a_declared_count_above_the_data_keeps_isolated_vertices(self):
        """A vertex with no edge exists only if the header says so, so there
        the header is the only evidence and is taken."""
        graph = parse_graph("p edge 10 1\ne 1 2\n")
        assert graph.vertices == 10

    def test_the_same_holds_of_a_network(self):
        network = parse_max_flow("p max 2 1\nn 1 s\nn 7 t\na 1 7 5\n")
        assert network.vertices == 7
        assert network.sink_vertex == 6

    def test_a_header_out_by_an_order_of_magnitude_is_refused(self):
        """Off by one, or by the factor of two both directions of every edge
        cost, is a stale header.  Off by ten is a truncated file or the wrong
        format, and reading it would give a plausible graph of the wrong size."""
        text = "p edge 3 2\n" + "".join(
            "e {} {}\n".format(i, i + 1) for i in range(1, 400)
        )
        with pytest.raises(InstanceError, match="line 1"):
            parse_graph(text)

    def test_a_header_promising_far_more_than_the_file_holds_is_refused(self):
        """The other direction: edges are never implicit, so a file holding one
        of a thousand declared edges has been cut short."""
        with pytest.raises(InstanceError, match="1000"):
            parse_graph("p edge 500 1000\ne 1 2\n")

    def test_a_count_that_is_not_a_count_is_refused(self):
        with pytest.raises(InstanceError, match="line 1"):
            parse_graph("p edge 4 many\ne 1 2\n")
        with pytest.raises(InstanceError, match="line 1"):
            parse_graph("p edge -4 1\ne 1 2\n")


class TestNothingIsGuessed:
    """Every refusal names the line and says what was wrong with it, because
    the alternative -- skipping the line -- returns an instance that is quietly
    not the file's."""

    def test_a_file_with_no_problem_line_is_refused_at_its_first_data_line(
        self,
    ):
        """noproblem.max states nodes and arcs with nothing declaring them; the
        format puts the problem line before either."""
        with pytest.raises(InstanceError, match="line 2") as caught:
            read_max_flow(FIXTURES / "noproblem.max")
        assert "problem line" in str(caught.value)

    def test_a_file_of_nothing_but_comments_says_there_is_no_problem_line(self):
        with pytest.raises(InstanceError, match="no problem line"):
            parse_max_flow("c a file that never gets to the point\n")

    def test_a_second_source_designator_is_refused_and_both_are_named(self):
        """Exactly one node is the source.  Two is not a network with a choice,
        it is a file we cannot read."""
        with pytest.raises(InstanceError, match="line 5") as caught:
            read_max_flow(FIXTURES / "twosources.max")
        message = str(caught.value)
        assert "source" in message
        assert "line 3" in message

    def test_a_second_sink_designator_is_refused(self):
        with pytest.raises(InstanceError, match="line 4"):
            parse_max_flow("p max 3 1\nn 1 s\nn 3 t\nn 2 t\na 1 3 1\n")

    def test_an_arc_out_of_node_zero_is_refused(self):
        """zeronode.max counts from 0.  The conversion to zero-based numbering
        would shift every vertex by one and still produce a network."""
        with pytest.raises(InstanceError, match="line 6") as caught:
            read_max_flow(FIXTURES / "zeronode.max")
        assert "from 1" in str(caught.value)

    def test_a_network_needs_both_terminals(self):
        with pytest.raises(InstanceError, match="no sink"):
            parse_max_flow("p max 2 1\nn 1 s\na 1 2 3\n")
        with pytest.raises(InstanceError, match="no source"):
            parse_max_flow("p max 2 1\nn 2 t\na 1 2 3\n")

    def test_one_node_cannot_be_both_terminals(self):
        with pytest.raises(InstanceError, match="line 3"):
            parse_max_flow("p max 2 1\nn 1 s\nn 1 t\na 1 2 3\n")

    def test_a_designator_that_is_neither_s_nor_t_is_refused(self):
        with pytest.raises(InstanceError, match="line 2"):
            parse_max_flow("p max 2 1\nn 1 x\nn 2 t\na 1 2 3\n")

    def test_a_negative_capacity_is_refused(self):
        with pytest.raises(InstanceError, match="line 4"):
            parse_max_flow("p max 2 1\nn 1 s\nn 2 t\na 1 2 -3\n")

    def test_an_arc_missing_its_capacity_is_refused(self):
        with pytest.raises(InstanceError, match="line 4") as caught:
            parse_max_flow("p max 2 1\nn 1 s\nn 2 t\na 1 2\n")
        assert "capacity" in str(caught.value)

    def test_a_line_of_an_unknown_kind_stops_the_read(self):
        """The dialects of this format are many; a reader that skipped what it
        did not know would answer confidently on a file it had not read."""
        with pytest.raises(InstanceError, match="line 3"):
            parse_graph("p edge 3 1\ne 1 2\nv 3 blue\n")
        with pytest.raises(InstanceError, match="line 4"):
            parse_max_flow("p max 2 1\nn 1 s\nn 2 t\nd 1 2 3\na 1 2 3\n")

    def test_two_problem_lines_are_refused(self):
        with pytest.raises(InstanceError, match="line 2"):
            parse_graph("p edge 3 1\np edge 3 1\ne 1 2\n")

    def test_each_reader_refuses_the_other_format_and_names_the_one_wanted(
        self,
    ):
        """A ``.max`` file and a ``.clq`` file are both DIMACS and are not the
        same object, so the mistake is worth catching by name."""
        with pytest.raises(InstanceError, match="parse_graph"):
            parse_max_flow("p edge 3 1\ne 1 2\n")
        with pytest.raises(InstanceError, match="parse_max_flow"):
            parse_graph("p max 3 1\nn 1 s\nn 3 t\na 1 3 2\n")

    def test_a_problem_line_for_no_format_at_all_is_refused(self):
        with pytest.raises(InstanceError, match="line 1"):
            parse_graph("p sat 3 1\ne 1 2\n")

    def test_an_edge_carrying_two_extra_fields_is_refused(self):
        """One extra field is a weight and is dropped; two is a format this
        does not know, and guessing which fields were the endpoints is exactly
        what a reader must not do."""
        with pytest.raises(InstanceError, match="line 2"):
            parse_graph("p edge 3 1\ne 1 2 7 9\n")

    def test_a_file_that_is_not_there_says_so(self):
        with pytest.raises(InstanceError, match="cannot read"):
            read_graph(FIXTURES / "no-such-file.clq")
