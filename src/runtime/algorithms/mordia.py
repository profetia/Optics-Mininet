import argparse
import logging
import os
import numpy as np
import numpy.typing as npt
import math

from typing import Any, Optional, Tuple

from runtime import core
from runtime.stub import consts
from runtime.algorithms import common, snoop


logger = logging.getLogger(__name__)


class MordiaScheduler:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        auxiliary: Any,
    ) -> core.UnifiedTopology:
        bdm = common.sinkhorn_transform(matrix)
        composition = common.birkhoff_von_neumann_decomposition(bdm)

        composition.sort(key=lambda x: x[0], reverse=True)
        composition = composition[: consts.TOR_NUM]
        print(composition)

        topology, allocated_slices = [], 0
        for coefficient, permutation in composition:
            if allocated_slices >= consts.SLICE_NUM or coefficient < 0.01:
                break

            n_slices = math.ceil(coefficient * consts.SLICE_NUM)
            if allocated_slices + n_slices > consts.SLICE_NUM:
                n_slices = consts.SLICE_NUM - allocated_slices

            links = set()
            for (i, j), value in np.ndenumerate(permutation):
                if value > 0:
                    links.add((i, j))

            topology.append(
                (
                    (allocated_slices, allocated_slices + n_slices),
                    links,
                )
            )

            print(
                f"Alocating [{allocated_slices}, {allocated_slices + n_slices}]\n{links}"
            )

            allocated_slices += n_slices

        return topology


class MordiaTimingHandler:

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        monment: float,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        variance: Optional[npt.NDArray[np.float64]],
    ) -> Optional[Tuple[()]]:

        if np.any(matrix > 0):
            logger.info("\n")
            logger.info("|" + "-" * 52 + "|")
            logger.info("| %-50s |" % f"Mordia at {monment}")
            logger.info("| %-50s |" % f"Matrix Max: {matrix.max()}")
            logger.info("|" + "-" * 52 + "|")

            return ()

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="opsys_control")
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
        MordiaScheduler(),
        core.Config(
            address=(args.address, args.port),
            report_kwargs=dict(
                hosts=hosts,
                tors=tors,
                relations=relations,
            ),
            clear_default=True,
        ),
    )

    runtime.add_timing_handler(dict(interval=1.0), MordiaTimingHandler())

    if args.snoop:
        runtime.add_event_handler(snoop.SnoopEventHandler())

    runtime.run()


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", logging.WARNING)
    logging.basicConfig(level=getattr(logging, level, logging.WARNING))

    args = parse_args()
    main(args)
