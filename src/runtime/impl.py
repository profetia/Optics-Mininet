import asyncio
import dataclasses
import numpy as np
import socket
import struct
import time

from multiprocessing import Process, Queue
from typing import Any, Callable, List, Optional, Tuple

import common

from rpc import Client, Host
from runtime.report import Report, ReportHeader, ReportEntry
from stub import consts

EventHandler = Callable[[int, np.ndarray], Optional[Any]]
"""An event handler function takes the event counter and the
traffic matrix as input and returns an optional auxiliary data
structure. If the event handler returns None, a workload will not
be scheduled. Otherwise, the traffic matrix and the auxiliary data
structure will be passed to the schedule function.
"""

TimingHandler = Callable[[float, np.ndarray], Optional[Any]]
"""A timing handler function takes the invocation time and the
traffic matrix as input and returns an optional auxiliary data
structure. If the timing handler returns None, a workload will not
be scheduled. Otherwise, the traffic matrix and the auxiliary data
structure will be passed to the schedule function.
"""

Scheduler = Callable[[np.ndarray, Any], np.ndarray]
"""A scheduler function takes the traffic matrix and the auxiliary
data structure as input and returns a new schedule matrix.
"""


SLICE_DURATION_US = 50
"""The duration of a time slice in us."""


@dataclasses.dataclass
class Config:
    address: Tuple[str, int]
    """The address of the collector daemon."""

    report_kwargs: dict
    """The keyword arguments for the report object."""


class Runtime:
    def __init__(self, scheduler: Scheduler, config: Config) -> None:
        self.scheduler = scheduler
        self.config = config

        self.event_handlers: List[EventHandler] = []
        self.timing_handlers: List[Tuple[dict, TimingHandler]] = []

    def add_event_handler(self, handler: EventHandler) -> None:
        self.event_handlers.append(handler)

    def add_timing_handler(self, timing_kwargs: dict, handler: TimingHandler) -> None:
        self.timing_handlers.append((timing_kwargs, handler))

    def run(self) -> None:
        channel = Queue()

        collect = Process(
            target=collect_daemon,
            args=(channel, self.config, self.event_handlers, self.timing_handlers),
        )
        collect.start()

        schedule = Process(target=schedule_daemon, args=(channel, self.scheduler))
        schedule.start()

        collect.join()
        schedule.join()


ENTRY_SIZE = struct.calcsize("QII")


def collect_daemon_timing_impl(
    channel: Queue,
    report: Report,
    timing_handler: TimingHandler,
) -> None:
    now = time.time()

    auxiliary = timing_handler(now, report.matrix())
    if auxiliary is None:
        return

    channel.put((report.matrix(), auxiliary))


def collect_daemon(
    channel: Queue,
    config: Config,
    event_handlers: List[EventHandler],
    timing_handlers: List[Tuple[dict, TimingHandler]],
) -> None:
    report = Report(**config.report_kwargs)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(config.address)
    udp.setblocking(False)

    for timing_kwargs, handler in timing_handlers:
        common.Repeat(
            **timing_kwargs,
            function=collect_daemon_timing_impl,
            args=(channel, report, handler),
        ).start()

    while True:
        try:
            data, (source, _) = udp.recvfrom(1024 + len(ReportHeader))
        except BlockingIOError:
            continue

        # the first 16 bytes are the header, extract as a 16 of bytes
        header, payload = data[: len(ReportHeader)], data[len(ReportHeader) :]
        if header != ReportHeader:
            continue

        report_entries_len = len(payload) // ENTRY_SIZE
        report_entries = [
            ReportEntry(payload, i * ENTRY_SIZE) for i in range(report_entries_len)
        ]

        updated = report.update(source, report_entries)
        if not updated:
            continue

        matrix = report.matrix()
        for handler in event_handlers:
            auxiliary = handler(report.counter(), matrix)
            if auxiliary is None:
                continue

            channel.put((matrix, auxiliary))


def schedule_daemon(
    channel: Queue,
    scheduler: Scheduler,
) -> None:
    asyncio.run(schedule_daemon_impl(channel, scheduler))


def translate_matrix(matrix: np.ndarray) -> np.ndarray:
    n_tors, n_ports = matrix.shape

    schedule = np.full((consts.SLICE_NUM, n_tors * consts.PORT_NUM), -1, dtype=int)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if not value:
                continue

            column = i * consts.PORT_NUM
            schedule[:, column] = j

    return schedule


async def schedule_daemon_impl(
    channel: Queue,
    scheduler: Scheduler,
):
    clients = [Client(Host.Uranus), Client(Host.Neptune)]

    while True:
        matrix, auxiliary = channel.get()
        topology = scheduler(matrix, auxiliary)
        schedule = translate_matrix(topology)

        # with common.timing("schedule_daemon_impl"):
        await asyncio.gather(
            *clients[0].pause_and_resume_flow_unchecked(schedule),
            *clients[1].pause_and_resume_flow_unchecked(schedule),
        )
