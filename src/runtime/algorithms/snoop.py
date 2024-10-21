import argparse
import numpy as np

from typing import Any

from runtime import core
from runtime.stub import consts


def NoopScheduler():
    def schedule(matrix: np.array, auxiliary: Any):
        pass

    return schedule


def PrintEventHandler():
    def handle_event(counter: int, matrix: np.ndarray):
        print(f"|------- Event {counter} -------|\n{matrix}\n")

    return handle_event


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
        NoopScheduler(),
        core.Config(
            (args.address, args.port),
            dict(hosts=hosts, tors=tors, relations=relations),
        ),
    )

    runtime.add_event_handler(PrintEventHandler())

    runtime.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
