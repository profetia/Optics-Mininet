import argparse
import numpy as np
import numpy.typing as npt

from typing import Any, Optional, Set, Tuple

from runtime import core
from runtime.stub import consts
from runtime.algorithms import common


class CThroughScheduler:
    def __init__(self) -> None:
        pass

    def __call__(self, matrix: np.array, auxiliary: Any) -> Set[Tuple[int, int]]:
        topology = common.bipartite_matching(matrix)

        return topology


class CThroughTimingHandler:

    BUFFER_SIZE_THRESHOLD = 200 * common.Bytes.KB
    # BUFFER_SIZE_THRESHOLD = 1 * common.Bytes.MB

    BUFFER_VARIANCE_THRESHOLD = 200 * 100 * common.Bytes.KB
    # BUFFER_VARIANCE_THRESHOLD = 1 * common.Bytes.MB

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        monment: float,
        matrix: npt.NDArray[np.int32],
        variance: npt.NDArray[np.float32],
    ) -> Optional[Tuple[()]]:
        # print("Matrix:", matrix.max())
        # print("Variance:", variance.max())

        if np.any(matrix > self.BUFFER_SIZE_THRESHOLD) and np.any(
            variance > self.BUFFER_VARIANCE_THRESHOLD
        ):
            # print("|" + "-" * 52 + "|")
            # print("| %-50s |" % f"C-Through at {monment}")
            # print("| %-50s |" % f"Buffer Size Threshold: {self.BUFFER_SIZE_THRESHOLD}")
            # print(
            #     "| %-50s |"
            #     % f"Buffer Variance Threshold: {self.BUFFER_VARIANCE_THRESHOLD}"
            # )
            # print("| %-50s |" % f"Matrix Max: {matrix.max()}")
            # print("| %-50s |" % f"Variance Max: {variance.max()}")

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

    runtime.add_timing_handler(dict(interval=0.03), CThroughTimingHandler())

    runtime.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
