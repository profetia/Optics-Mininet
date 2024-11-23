import argparse
import logging
import os
import time
import numpy as np
import numpy.typing as npt

from typing import Any, Dict, List, Optional, Set, Tuple

from runtime import core
from runtime.stub import consts
from runtime.algorithms import common, snoop


logger = logging.getLogger(__name__)


def split_connected_components(
    edges: Set[Tuple[int, int]],
) -> Tuple[
    List[Tuple[int, int]],
    Dict[Tuple[int, int], int],
]:
    parent = {}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)

        if root1 == root2:
            return

        parent[root1] = root2

    for src, dst in edges:
        if src not in parent:
            parent[src] = src
        if dst not in parent:
            parent[dst] = dst

    for src, dst in edges:
        union(src, dst)

    components, edge_map = [], {}
    parent_remap, remap_count = {}, 0
    for src, dst in edges:
        root = find(src)
        if root not in parent_remap:
            parent_remap[root] = remap_count
            remap_count += 1
            components.append([])

        component_index = parent_remap[root]
        components[component_index].append((src, dst))
        edge_map[(src, dst)] = component_index

    return components, edge_map


def shortest_path_routing(
    edges: Set[Tuple[int, int, np.float64]],
) -> Set[Tuple[int, int]]:
    # print(edges)
    nodes = list(range(4))

    dist = np.full((4, 4), np.inf, dtype=np.float64)
    next_hop = np.full((4, 4), -1, dtype=np.int32)

    for src, dst, weight in edges:
        dist[src][dst], dist[dst][src] = weight, weight
        next_hop[src][dst], next_hop[dst][src] = dst, src

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_hop[i][j] = next_hop[i][k]

    # print(dist, "\n", next_hop)

    topology = set()
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue

            next_node = next_hop[src][dst]
            topology.add((src, dst, next_node))

    return topology


class OSAScheduler:

    # WSS_B = 4  # 4 makes no sense if we only have 4 ToRs
    WSS_B = 2

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        auxiliary: Any,
    ) -> core.Topology:
        # Since we actually only have 4 ToRs, drop the unused ones
        real_tor_map = [0, 2, 3, 5]
        real_matrix = matrix[real_tor_map][:, real_tor_map]
        real_n_flows = n_flows[real_tor_map][:, real_tor_map]

        bdm = common.hedera_transform(real_matrix, real_n_flows)
        bdm = bdm + bdm.T  # Transform to undirected graph

        weight_map = {
            (i, j): value
            for (i, j), value in np.ndenumerate(bdm)
            if i != j  # and value != 0
        }

        topology = common.weighted_b_matching(
            bdm,
            self.WSS_B,
            ignore_zero=False,
        )

        connected_components, edge_map = split_connected_components(topology)
        assert len(connected_components) == 1
        # If we dont ignore zero, we should not worry about connectivity

        topology_weighted = set()
        for src, dst in topology:
            if (src, dst) in weight_map:
                topology_weighted.add((src, dst, weight_map[(src, dst)]))
            else:
                topology_weighted.add((dst, src, weight_map[(dst, src)]))

        # print(topology_weighted)

        real_topology = set()
        routes = shortest_path_routing(topology_weighted)
        for src, dst, next_hop in routes:
            real_topology.add(
                (
                    real_tor_map[src],
                    real_tor_map[dst],
                    real_tor_map[next_hop],
                )
            )

        return real_topology


class OSAEventHandler:

    AGING_SAMPLE_THRESHOLD = 8

    ELEPHANT_FLOW_THRESHOLD = 500 * common.Bytes.KB

    RATE_LIMIT_THRESHOLD = 50 * consts.SLICE_NUM * consts.SLICE_DURATION_US

    def __init__(self) -> None:
        self.aging_samples: List[npt.NDArray[np.int32]] = []
        self.rate_limit_time = time.time_ns()

    def __is_elephant_flow(self) -> bool:
        return np.any(
            np.sum(np.array(self.aging_samples), axis=0) > self.ELEPHANT_FLOW_THRESHOLD
        )

    def __call__(
        self,
        event: int,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        delta: npt.NDArray[np.int32],
    ) -> Optional[Tuple[()]]:
        if len(self.aging_samples) >= self.AGING_SAMPLE_THRESHOLD:
            self.aging_samples.pop(0)

        self.aging_samples.append(np.copy(matrix))

        now = time.time_ns()
        if now - self.rate_limit_time < self.RATE_LIMIT_THRESHOLD * 1_000:
            return None

        self.rate_limit_time = now

        if self.__is_elephant_flow() and np.any(delta > 0):
            logger.info("\n")
            logger.info("|" + "-" * 52 + "|")
            logger.info("| %-50s |" % f"OSA at {event}")
            logger.info("| %-50s |" % f"Matrix Max: {matrix.max()}")
            logger.info("| %-50s |" % f"Delta Max: {delta.max()}")
            logger.info("|" + "-" * 52 + "|")

            return ()

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSA")
    parser.add_argument(
        "-a", "--address", type=str, help="IPv4 address to bind to", default="0.0.0.0"
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Port number to bind to", default=1599
    )
    parser.add_argument(
        "-s", "--snoop", action="store_true", help="Enable snoop event handler"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    hosts = consts.host_ip
    tors = consts.host_ip
    relations = dict(zip(hosts, tors))

    runtime = core.Runtime(
        OSAScheduler(),
        core.Config(
            address=(args.address, args.port),
            report_kwargs=dict(
                hosts=hosts,
                tors=tors,
                relations=relations,
            ),
            clear_default=True,
            topology_type=core.TopologyType.MultiHop,
        ),
    )

    runtime.add_event_handler(OSAEventHandler())

    if args.snoop:
        runtime.add_event_handler(snoop.SnoopEventHandler())

    runtime.run()


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "WARNING")
    logging.basicConfig(level=getattr(logging, level, logging.WARNING))

    args = parse_args()
    main(args)
