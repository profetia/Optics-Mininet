import argparse
import time
import numpy as np
import numpy.typing as npt

from typing import Any, Optional, Set, Tuple

from runtime import core
from runtime.stub import consts


class RoundRobinScheduler:
    def __init__(self, n_tors: int):
        self.schedules = []

        for shift in range(1, n_tors):
            schedule = set()
            for i in range(n_tors):
                schedule.add((i, (i + shift) % n_tors))

            self.schedules.append(schedule)

    def __call__(self, matrix: np.array, auxiliary: Any) -> Set[Tuple[int, int]]:
        return self.schedules[auxiliary]


class RoundRobinEventHandler:

    def __init__(self):
        pass

    def __call__(
        self, counter: int, matrix: npt.NDArray[np.int32], delta: npt.NDArray[np.int32]
    ) -> None:
        print(f"|------- Event {counter} -------|\n{matrix}\n")


class RoundRobinTimingHandler:
    def __init__(self, n_tors: int, slice_duration_us: int):
        self.n_tors = n_tors
        self.slice_duration_us = slice_duration_us

        self.start = time.time()
        self.last_index = None

    def __call__(
        self,
        now: float,
        matrix: npt.NDArray[np.int32],
        variation: npt.NDArray[np.float64],
    ) -> Optional[int]:
        elapsed = (now - self.start) * 1000_000 / self.slice_duration_us
        new_index = int(elapsed) % (self.n_tors - 1)
        if self.last_index is not None and new_index == self.last_index:
            return None

        self.last_index = new_index
        return new_index


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

    TIME_SLICE_ROUND_ROBIN = core.SLICE_DURATION_US * consts.SLICE_NUM * 10 * 25

    runtime = core.Runtime(
        RoundRobinScheduler(8),
        core.Config(
            (args.address, args.port),
            dict(
                hosts=hosts,
                tors=tors,
                relations=relations,
            ),
        ),
    )

    runtime.add_timing_handler(
        dict(interval=TIME_SLICE_ROUND_ROBIN / 1e6),
        RoundRobinTimingHandler(8, TIME_SLICE_ROUND_ROBIN),
    )

    # runtime.add_event_handler(RoundRobinEventHandler())

    runtime.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
