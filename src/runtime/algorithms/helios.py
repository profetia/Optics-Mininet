import argparse
import logging
import os
import numpy as np
import numpy.typing as npt

from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Optional, Tuple

from runtime import core
from runtime.stub import consts
from runtime.algorithms import common, snoop


logger = logging.getLogger(__name__)


class HeliosScheduler:
    def __init__(self, signal: Synchronized) -> None:
        self.signal = signal

    def __call__(
        self,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        auxiliary: Any,
    ) -> core.UnifiedTopology:
        self.signal.value = 1

        bdm = common.hedera_transform(matrix, n_flows)
        topology = common.max_flow_matching(bdm)

        self.signal.value = 0
        return topology


class HeliosEventHandler:

    def __init__(
        self,
        signal: Synchronized,
        rate_limit_us: Optional[int] = None,
    ) -> None:
        self.signal = signal
        self.rate_limit = rate_limit_us

        self.last_event = None

    def __call__(
        self,
        event: int,
        matrix: npt.NDArray[np.int32],
        n_flows: npt.NDArray[np.int32],
        delta: npt.NDArray[np.int32],
    ) -> Optional[Tuple[()]]:
        if self.signal.value > 0:
            return None

        if np.any(matrix > 0) and np.any(delta > 0):
            logger.info("\n")
            logger.info("|" + "-" * 52 + "|")
            logger.info("| %-50s |" % f"Helios at {event}")
            logger.info("| %-50s |" % f"Matrix Max: {matrix.max()}")
            logger.info("| %-50s |" % f"Delta Max: {delta.max()}")
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

    signal: Synchronized = Value("i", 0)

    runtime = core.Runtime(
        HeliosScheduler(signal),
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

    runtime.add_event_handler(HeliosEventHandler(signal, rate_limit_us=10_000))

    if args.snoop:
        runtime.add_event_handler(snoop.SnoopEventHandler())

    runtime.run()


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", logging.WARNING)
    logging.basicConfig(level=getattr(logging, level, logging.WARNING))

    args = parse_args()
    main(args)
