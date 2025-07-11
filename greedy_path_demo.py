#!/usr/bin/env python3
"""
Greedy path selection demo using NetworkX.

This script constructs a weighted directed graph, computes a greedy path
(starting from the cheapest outgoing edge at each step) from `START_NODE`
to `END_NODE`, and visualises both the full graph and the greedy path.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# Data & configuration
# ---------------------------------------------------------------------------

# Sample weighted directed edges: (source, target, weight)
SAMPLE_EDGES: List[Tuple[str, str, int]] = [
    ("A", "B", 4),
    ("A", "C", 2),
    ("B", "C", 4),
    ("C", "B", 1),
    ("B", "D", 2),
    ("C", "D", 4),
    ("B", "E", 3),
    ("E", "D", 1),
    ("C", "E", 5),
]

START_NODE: str = "A"
END_NODE: str = "E"

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_graph(edges: List[Tuple[str, str, int]]) -> nx.DiGraph:
    """Return a weighted directed graph constructed from *edges*."""
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges)
    return g


def greedy_path(
    graph: nx.DiGraph, start: str, end: str
) -> List[Tuple[str, str, int]]:
    """Return a greedy path from *start* to *end*.

    The algorithm chooses, at each step, the minimum‑weight outgoing edge
    from the current node that leads to an *unvisited* neighbour. It raises a
    ``ValueError`` if it gets stuck before reaching *end*.
    """
    if start not in graph or end not in graph:
        raise ValueError("Both start and end nodes must exist in the graph")

    path: List[Tuple[str, str, int]] = []
    current = start
    visited = {start}

    while current != end:
        outgoing = [
            (u, v, data["weight"])
            for u, v, data in graph.edges(current, data=True)
            if v not in visited
        ]
        if not outgoing:
            raise ValueError(
                f"Greedy algorithm stuck at node {current!r}. "
                "No unvisited outgoing edges."
            )
        # Choose the edge with the smallest weight
        u, v, w = min(outgoing, key=lambda t: t[2])
        path.append((u, v, w))
        current = v
        visited.add(v)

    return path

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _get_layout(graph: nx.DiGraph, layout: str):
    if layout == "dot":
        try:
            from networkx.drawing.nx_pydot import graphviz_layout

            return graphviz_layout(graph, prog="dot")
        except (ImportError, RuntimeError):
            print(
                "Graphviz layout requested but not available; "
                "falling back to spring layout.",
                file=sys.stderr,
            )
    # Default: spring layout
    return nx.spring_layout(graph)


def visualise(
    graph: nx.DiGraph,
    path_edges: Optional[List[Tuple[str, str, int]]] = None,
    layout: str = "spring",
) -> None:
    """Draw *graph* with matplotlib, optionally highlighting *path_edges*."""
    pos = _get_layout(graph, layout)

    # Draw full graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightgrey",
        node_size=1500,
        arrowsize=20,
        font_size=10,
    )

    # Edge labels (weights)
    edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Highlight greedy path if provided
    if path_edges:
        highlight = nx.DiGraph()
        highlight.add_edges_from([(u, v) for u, v, _ in path_edges])
        nx.draw_networkx_edges(
            highlight,
            pos,
            edge_color="red",
            width=2.5,
            arrowsize=25,
            connectionstyle="arc3,rad=0.1",
        )

    plt.show()

# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Greedy path demo")
    parser.add_argument(
        "--layout",
        choices=["spring", "dot"],
        default="spring",
        help="Graph layout engine (default: spring)",
    )
    args = parser.parse_args(argv)

    graph = build_graph(SAMPLE_EDGES)
    path = greedy_path(graph, START_NODE, END_NODE)

    # Pretty‑print path
    nodes_seq = [START_NODE] + [v for _, v, _ in path]
    print("Greedy path:", " -> ".join(nodes_seq))
    print("Edges :", path)

    visualise(graph, path_edges=path, layout=args.layout)


if __name__ == "__main__":
    main()
