import argparse
import time
import numpy as np
import numpy.typing as npt

from typing import Any, Optional

from runtime import core
from runtime.stub import consts


class CThroughScheduler:
    def __init__(self):
        pass

    def __call__(self, matrix: np.array, auxiliary: Any) -> np.array:
        pass


class CThroughEventHandler:

    def __init__(self):
        pass

    def __call__(self, counter: int, matrix: npt.NDArray[np.int32]) -> None:
        print(f"|------- Event {counter} -------|\n{matrix}\n")


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

    runtime.add_event_handler(CThroughEventHandler())

    runtime.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
