import enum
import networkx as nx
import numpy as np
import numpy.typing as npt

from typing import List, Optional, Set, Tuple

from runtime.stub import consts


class Bytes(enum.IntEnum):
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    TB = 1024 * GB


def bipartite_matching(
    matrix: npt.NDArray[np.int32 | np.float64],
) -> Set[Tuple[int, int]]:
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


NETWORKX_WEIGHT_MAX = 1_000_000


def edmonds_karp_matching(
    matrix: npt.NDArray[np.int32 | np.float64],
    includes_weight: bool = False,
) -> Set[Tuple[int, int]]:
    n_tors = matrix.shape[0]

    G = nx.DiGraph()

    left_nodes, right_nodes = set(), set()
    for (i, j), value in np.ndenumerate(matrix):
        if i == j or value == 0:
            continue

        if matrix.dtype == np.float64:
            weight = NETWORKX_WEIGHT_MAX - int(value * NETWORKX_WEIGHT_MAX)
        else:
            weight = -value

        G.add_edge(
            i,
            j + n_tors,
            capacity=1,
            weight=weight,
            # networkx works poorly with float weights
        )

        left_nodes.add(i)
        right_nodes.add(j + n_tors)

    source = n_tors * 2
    for i in left_nodes:
        G.add_edge(
            source,
            i,
            capacity=1,
            weight=0,
        )

    sink = n_tors * 2 + 1
    for j in right_nodes:
        G.add_edge(
            j,
            sink,
            capacity=n_tors,
            weight=0,
        )

    flow_dict = nx.max_flow_min_cost(G, source, sink)

    result = set()
    for src in left_nodes:
        for dst, value in flow_dict[src].items():
            if dst == source or value == 0:
                continue

            if includes_weight:
                result.add((src, dst - n_tors, matrix[src, dst - n_tors]))
            else:
                result.add((src, dst - n_tors))

    return result


def weighted_b_matching(
    matrix: npt.NDArray[np.float64],
    b: int,
    max_iter: int = 1000,
    ignore_zero: bool = False,
) -> Set[Tuple[int, int]]:
    G = nx.Graph()
    for (i, j), value in np.ndenumerate(matrix):
        if i == j or (value == 0 and ignore_zero):
            continue

        G.add_edge(i, j, weight=int(value * NETWORKX_WEIGHT_MAX))

    vertex_degree = {node: 0 for node in G.nodes}
    matched_edges = set()
    for _ in range(max_iter):
        new_all_matched = nx.max_weight_matching(G, maxcardinality=True)
        if not new_all_matched:
            break

        new_matched_edges = set()
        for src, dst in new_all_matched:
            if vertex_degree[src] < b and vertex_degree[dst] < b:
                vertex_degree[src] += 1
                vertex_degree[dst] += 1
                new_matched_edges.add((src, dst))

        matched_edges.update(new_matched_edges)

        if all(degree >= b for degree in vertex_degree.values()):
            break

        for src, dst in new_matched_edges:
            G.remove_edge(src, dst)

    return matched_edges


F64_EPSILON = np.finfo(np.float64).eps


def sinkhorn_transform(
    matrix: npt.NDArray[np.int32],
    max_iter: int = 1000,
    epsilon: float = 1e-6,
) -> npt.NDArray[np.float64]:
    matrix_unified = matrix.astype(np.float64) / (
        consts.bandwidth_after_vma0 * Bytes.GB
    )

    matrix_unified[matrix_unified == 0] = F64_EPSILON
    np.fill_diagonal(matrix_unified, 0)

    for _ in range(max_iter):
        row_sum = matrix_unified.sum(axis=1)
        matrix_unified /= row_sum[:, np.newaxis]

        col_sum = matrix_unified.sum(axis=0)
        matrix_unified /= col_sum

        error = np.abs(row_sum - 1).max()
        if error < epsilon:
            break

    return matrix_unified


def birkhoff_von_neumann_decomposition(
    matrix: npt.NDArray[np.float64],
    max_iter: Optional[int] = None,
) -> List[Tuple[float, npt.NDArray[np.int32]]]:
    n_tors = matrix.shape[0]

    if max_iter is None:
        max_iter = n_tors

    remaining = matrix.copy()

    result = []
    for _ in range(max_iter):
        topology = bipartite_matching(remaining)
        if len(topology) == 0:
            break

        permutation = np.zeros((n_tors, n_tors), dtype=np.int32)
        coeficient = np.inf

        for src, dst in topology:
            coeficient = min(coeficient, remaining[src, dst])
            permutation[src, dst] = 1

        remaining -= coeficient * permutation

        result.append((coeficient, permutation))

    return result


def hedera_transform(
    matrix: npt.NDArray[np.int32],
    n_flows: npt.NDArray[np.int32],
    max_iter: int = 1000,
    epsilon: float = 1e-6,
) -> npt.NDArray[np.float64]:
    f_converged = np.zeros_like(matrix, dtype=bool)
    f_rl = np.zeros_like(matrix, dtype=bool)

    # f_demand = matrix.astype(np.float64)
    f_demand = matrix.astype(np.float64) / (consts.bandwidth_after_vma0 * Bytes.GB)
    for (i, j), n_flow in np.ndenumerate(n_flows):
        if n_flow == 0:
            continue

        f_demand[i, j] /= n_flow

    def est_src(src: int):
        d_F, n_U = 0.0, 0
        for j in range(f_demand.shape[1]):
            if src == j or n_flows[src, j] == 0:
                continue

            for _ in range(n_flows[src, j]):
                if f_converged[src, j]:
                    d_F += f_demand[src, j]
                else:
                    n_U += 1

        if n_U == 0:
            return

        e_S = (1.0 - d_F) / n_U
        for j in range(f_demand.shape[1]):
            if src == j or n_flows[src, j] == 0:
                continue

            if not f_converged[src, j]:
                f_demand[src, j] = e_S

    def est_dst(dst: int):
        d_T, d_S, n_R = 0.0, 0.0, 0
        for i in range(f_demand.shape[0]):
            f_rl[i, dst] = True

            if i == dst or n_flows[i, dst] == 0:
                continue

            for _ in range(n_flows[i, dst]):
                d_T += f_demand[i, dst]
                n_R += 1

        if d_T <= 1.0:
            return

        e_S = 1.0 / n_R
        while not np.all(f_rl[:, dst]):
            n_R = 0
            for i in range(f_demand.shape[0]):
                if i == dst or n_flows[i, dst] == 0:
                    continue

                if not f_rl[i, dst]:
                    continue

                for _ in range(n_flows[i, dst]):
                    if f_demand[i, dst] < e_S:
                        d_S += f_demand[i, dst]
                        f_rl[i, dst] = False
                    else:
                        n_R += 1

            e_S = (1.0 - d_S) / n_R

        for i in range(f_demand.shape[0]):
            if i == dst or n_flows[i, dst] == 0:
                continue

            f_demand[i, dst] = e_S
            f_converged[i, dst] = True

    for _ in range(max_iter):
        f_demand_old = np.copy(f_demand)

        for i in range(f_demand.shape[0]):
            est_src(i)

        for j in range(f_demand.shape[1]):
            est_dst(j)

        if np.allclose(f_demand, f_demand_old, atol=epsilon):
            break

    return f_demand


if __name__ == "__main__":
    # matrix = np.array(
    #     [
    #         [0, 20939400, 2478296, 0],
    #         [10485760, 0, 0, 0],
    #         [0, 10456561, 0, 0],
    #         [0, 0, 10428820, 0],
    #     ]
    # )

    # matrix = np.array(
    #     [
    #         [0, 0, 0, 9748460],
    #         [0, 0, 10429952, 0],
    #         [0, 10443421, 0, 0],
    #         [10485761, 0, 0, 0],
    #     ]
    # )

    # matrix = np.array(
    #     [
    #         [0, 9891541, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 15415480, 0, 20279482],
    #         [0, 10363120, 0, 0],
    #     ]
    # )

    # bdm = sinkhorn_transform(matrix)
    # print(bdm.astype(np.float32))
    # assert np.allclose(bdm.sum(axis=0), 1)
    # assert np.allclose(bdm.sum(axis=1), 1)

    # bvn = birkhoff_von_neumann_decomposition(bdm)
    # for coeficient, permutation in bvn:
    #     print(coeficient)
    #     print(permutation)

    # print(sum(perm for _, perm in bvn))

    # matrix = np.array(
    #     [
    #         [0, 10306980, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 10485760, 0, 0],
    #         [0, 10463860, 0, 0],
    #     ]
    # )
    # n_flows = np.array(
    #     [
    #         [0, 1, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 1, 0, 0],
    #     ]
    # )

    # matrix = np.array(
    #     [
    #         [0, 1 / 3, 1 / 3, 1 / 3],
    #         [1 / 3, 0, 1 / 3, 0],
    #         [1 / 2, 0, 0, 1 / 2],
    #         [0, 1 / 2, 0, 0],
    #     ],
    # )
    # n_flows = np.array(
    #     [
    #         [0, 1, 1, 1],
    #         [2, 0, 1, 0],
    #         [1, 0, 0, 1],
    #         [0, 2, 0, 0],
    #     ]
    # )

    # bdm = hedera_transform(matrix, n_flows)
    # matches = edmonds_karp_matching(bdm)
    # print(matches)

    bdm = np.array(
        [
            [0, 0, 0.05, 0.075, 0, 0, 0, 0],
            [0.05263158, 0, 0, 0, 0, 0, 0, 0],
            [0.05263158, 0, 0, 0, 0, 0, 0, 0],
            [0.05263158, 0, 0, 0, 0, 0, 0, 0],
            [0.05263158, 0, 0, 0, 0, 0, 0, 0],
            [0.05263158, 0, 0.05, 0, 0, 0, 0, 0],
            [0.05263158, 0, 0, 0, 0, 0, 0, 0],
            [0.05263158, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    print(weighted_b_matching(bdm, 4))
