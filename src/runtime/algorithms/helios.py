import argparse
import logging
import os
import time
import numpy as np
import numpy.typing as npt

from typing import Any, List, Optional, Tuple

from runtime import core
from runtime.stub import consts
from runtime.algorithms import common, snoop


logger = logging.getLogger(__name__)


class HeliosScheduler:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        auxiliary: Any,
    ) -> core.Topology:

        bdm = common.hedera_transform(matrix, n_flows)
        topology = common.edmonds_karp_matching(bdm)

        return topology


class HeliosEventHandler:

    AGING_SAMPLE_THRESHOLD = 8

    ELEPHANT_FLOW_THRESHOLD = 500 * common.Bytes.KB

    RATE_LIMIT_THRESHOLD = 40 * consts.SLICE_NUM * consts.SLICE_DURATION_US
    # Since our while system works much faster than the original one in Helios,
    # we need to adjust the rate limit threshold to avoid reconfiguring the
    # OCS too frequently.

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
            logger.info("| %-50s |" % f"Helios at {event}")
            logger.info("| %-50s |" % f"Matrix Max: {matrix.max()}")
            logger.info("| %-50s |" % f"Delta Max: {delta.max()}")
            logger.info("|" + "-" * 52 + "|")

            return ()

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helios")
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
        HeliosScheduler(),
        core.Config(
            address=(args.address, args.port),
            clear_default=True,
            report_kwargs=dict(
                hosts=hosts,
                tors=tors,
                relations=relations,
            ),
        ),
    )

    runtime.add_event_handler(HeliosEventHandler())

    if args.snoop:
        runtime.add_event_handler(snoop.SnoopEventHandler())

    runtime.run()


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "WARNING")
    logging.basicConfig(level=getattr(logging, level, logging.WARNING))

    args = parse_args()
    main(args)
