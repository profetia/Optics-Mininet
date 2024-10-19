import argparse
import time
import numpy as np

from typing import Any, Optional

import impl
import stub.consts as consts


class RoundRobinScheduler:
    def __init__(self, n_tors: int):
        self.schedules = []

        for shift in range(1, n_tors):
            schedule = np.full((n_tors, n_tors), 0, dtype=np.int32)
            for i in range(n_tors):
                schedule[i][(i + shift) % n_tors] = 1
            self.schedules.append(schedule)

    def __call__(self, matrix: np.array, auxiliary: Any) -> np.array:
        return self.schedules[auxiliary]


class RoundRobinEventHandler:

    def __init__(self):
        pass

    def __call__(self, counter: int, matrix: np.ndarray) -> None:
        print(f"|------- Event {counter} -------|\n{matrix}\n")


class RoundRobinTimingHandler:
    def __init__(self, n_tors: int, slice_duration_us: int):
        self.n_tors = n_tors
        self.slice_duration_us = slice_duration_us

        self.start = time.time()
        self.last_index = None

    def __call__(self, now: float, matrix: np.array) -> Optional[int]:
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
    host_x10 = "10.29.1.110"
    host_x11 = "10.29.1.111"
    host_x12 = "10.29.1.120"
    host_x13 = "10.29.1.121"
    host_x14 = "10.29.1.130"
    host_x15 = "10.29.1.131"
    host_x16 = "10.29.1.140"
    host_x17 = "10.29.1.141"

    hosts = [
        host_x10,
        host_x11,
        host_x12,
        host_x13,
        host_x14,
        host_x15,
        host_x16,
        host_x17,
    ]

    tor_0 = "tor0"
    tor_1 = "tor1"
    tor_2 = "tor2"
    tor_3 = "tor3"
    tor_4 = "tor4"
    tor_5 = "tor5"
    tor_6 = "tor6"
    tor_7 = "tor7"

    tors = [tor_0, tor_1, tor_2, tor_3, tor_4, tor_5, tor_6, tor_7]

    relations = {
        host_x10: tor_0,
        host_x11: tor_1,
        host_x12: tor_2,
        host_x13: tor_3,
        host_x14: tor_4,
        host_x15: tor_5,
        host_x16: tor_6,
        host_x17: tor_7,
    }

    # host_x10_mapped = "172.16.11.10"
    # host_x11_mapped = "172.16.11.11"
    # host_x12_mapped = "172.16.12.10"
    # host_x13_mapped = "172.16.12.11"
    # host_x14_mapped = "172.16.13.10"
    # host_x15_mapped = "172.16.13.11"
    # host_x16_mapped = "172.16.14.10"
    # host_x17_mapped = "172.16.14.11"

    # host_map = {
    #     host_x10_mapped: host_x10,
    #     host_x11_mapped: host_x11,
    #     host_x12_mapped: host_x12,
    #     host_x13_mapped: host_x13,
    #     host_x14_mapped: host_x14,
    #     host_x15_mapped: host_x15,
    #     host_x16_mapped: host_x16,
    #     host_x17_mapped: host_x17,
    # }

    time_slice_us = impl.SLICE_DURATION_US * consts.SLICE_NUM * 10 * 25

    runtime = impl.Runtime(
        RoundRobinScheduler(8),
        impl.Config(
            (args.address, args.port),
            dict(
                hosts=hosts,
                tors=tors,
                relations=relations,
                # host_map=host_map,
            ),
        ),
    )

    runtime.add_timing_handler(
        dict(interval=time_slice_us / 1e6),
        RoundRobinTimingHandler(8, time_slice_us),
    )

    # runtime.add_event_handler(RoundRobinEventHandler())

    runtime.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
