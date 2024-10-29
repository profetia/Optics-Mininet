import argparse
import numpy as np
import numpy.typing as npt

from typing import Any, Optional, Set, Tuple

from runtime import core, statistics
from runtime.stub import consts
from runtime.algorithms import common


class CThroughScheduler:
    def __init__(self) -> None:
        pass

    def __call__(self, matrix: np.array, auxiliary: Any) -> Set[Tuple[int, int]]:
        topology = common.bipartite_matching(matrix)

        # (counter,) = auxiliary
        # print(f"========== Event {counter} dispatched ==========")

        return topology


class CThroughEventHandler:

    BUFFER_SIZE_THRESHOLD = 204800
    BUFFER_VARIANCE_THRESHOLD = 204800 * 100

    def __init__(self, shape: tuple[int, int]):
        self.stat = statistics.RunningStatistics(shape=shape)

    def __call__(
        self, counter: int, matrix: npt.NDArray[np.int32], delta: npt.NDArray[np.int32]
    ) -> Optional[Tuple[int]]:
        self.stat.update(matrix)
        # print("Matrix:", matrix.max())
        # print("Variance:", self.stat.variance().max())

        if np.any(matrix > self.BUFFER_SIZE_THRESHOLD) and np.any(
            self.stat.variance() > self.BUFFER_VARIANCE_THRESHOLD
        ):
            # print(f"========== Event {counter} triggered ==========")
            # print("Matrix:", matrix.max())
            # print("Variance:", self.stat.variance().max())

            self.stat.reset()
            self.stat.update(matrix)

            return (counter,)

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="opsys_control")
    parser.add_argument(
        "-a", "--address", type=str, help="IPv4 address to bind to", default="0.0.0.0"
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Port number to bind to", default=1599
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    hosts = consts.host_ip
    tors = consts.host_ip
    relations = dict(zip(hosts, tors))

    runtime = core.Runtime(
        CThroughScheduler(),
        core.Config(
            (args.address, args.port),
            dict(
                hosts=hosts,
                tors=tors,
                relations=relations,
            ),
        ),
    )

    runtime.add_event_handler(CThroughEventHandler((len(tors), len(tors))))

    runtime.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
