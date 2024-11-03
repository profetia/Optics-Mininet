import enum
import networkx as nx
import numpy as np
import numpy.typing as npt

from typing import Optional, Set, Tuple


class Bytes(enum.IntEnum):
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    TB = 1024 * GB


def bipartite_matching(
    matrix: npt.NDArray[np.int32],
    n_tors: Optional[int] = None,
) -> Set[Tuple[int, int]]:
    if n_tors is None:
        n_tors = matrix.shape[0]

    edges = []
    for (i, j), value in np.ndenumerate(matrix):
        if i == j or value == 0:
            continue

        edges.append((i, j + n_tors, value))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    matching = nx.max_weight_matching(G, maxcardinality=True)

    topology = set()
    for src, dst in matching:
        if src >= n_tors:
            src, dst = dst, src

        topology.add((src, dst - n_tors))

    return topology
