import asyncio
import dataclasses
import numpy as np
import numpy.typing as npt
import socket
import struct
import time
import uvloop

from multiprocessing import Process, Queue
from typing import Any, Callable, Iterable, List, Optional, Tuple

from . import common

from .rpc import Client, Host
from .report import Report, ReportHeader, ReportEntry
from .stub import consts
from .statistics import RunningStatistics

EventHandler = Callable[
    [int, npt.NDArray[np.int32], npt.NDArray[np.int32]], Optional[Any]
]
"""An event handler function takes the event counter, the traffic matrix,
and the delta matrix as input and returns an optional auxiliary data
structure. If the event handler returns None, a workload will not be
scheduled. Otherwise, the traffic matrix and the auxiliary data structure
will be passed to the schedule function.
"""

TimingHandler = Callable[
    [float, npt.NDArray[np.int32], npt.NDArray[np.float32]], Optional[Any]
]
"""A timing handler function takes the invocation time, the traffic matrix,
and the variation matrix as input and returns an optional auxiliary data
structure. If the timing handler returns None, a workload will not be
scheduled. Otherwise, the traffic matrix and the auxiliary data structure
will be passed to the schedule function.
"""

Scheduler = Callable[[npt.NDArray[np.int32], Any], Iterable[Tuple[int, int]]]
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

        schedule = Process(target=schedule_daemon, args=(channel, self.scheduler))
        schedule.start()

        collect_daemon(channel, self.config, self.event_handlers, self.timing_handlers)

        schedule.join()


ENTRY_SIZE = struct.calcsize("QII")


def collect_daemon_timing_impl(
    channel: Queue,
    report: Report,
    stat: RunningStatistics,
    timing_handler: TimingHandler,
) -> None:
    now = time.time()

    variance = stat.variance()

    matrix = report.matrix()
    stat.reset(value=matrix)

    auxiliary = timing_handler(now, matrix, variance)
    if auxiliary is None:
        return

    channel.put((matrix, auxiliary))


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

    stats: List[RunningStatistics] = []
    for timing_kwargs, handler in timing_handlers:
        stat = RunningStatistics(shape=report.matrix().shape)
        common.Repeat(
            **timing_kwargs,
            function=collect_daemon_timing_impl,
            args=(channel, report, stat, handler),
        ).start()

        stats.append(stat)

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

        delta = report.update(source, report_entries)

        matrix = report.matrix()
        for stat in stats:
            stat.update(matrix)

        # if np.sum(np.abs(delta)) == 0:
        #     continue

        for handler in event_handlers:
            auxiliary = handler(report.counter(), matrix, delta)
            if auxiliary is None:
                continue

            channel.put((matrix, auxiliary))


def schedule_daemon(
    channel: Queue,
    scheduler: Scheduler,
) -> None:
    uvloop.install()
    with asyncio.Runner() as runner:
        clients = [Client(Host.Uranus), Client(Host.Neptune)]
        while True:
            matrix, auxiliary = channel.get()
            topology = scheduler(matrix, auxiliary)
            schedule = translate_matrix(matrix.shape[0], topology)

            # with common.Timer("schedule_daemon_dispatch_impl"):
            runner.run(schedule_daemon_dispatch_impl(clients, schedule))


def translate_matrix(
    n_tors: int, topology: Iterable[Tuple[int, int]]
) -> npt.NDArray[np.int32]:

    schedule = np.full(
        n_tors * consts.PORT_NUM * consts.SLICE_NUM,
        -1,
        dtype=np.int32,
    )

    for src, dst in topology:
        column = src * consts.PORT_NUM
        schedule[column * consts.SLICE_NUM : (column + 1) * consts.SLICE_NUM] = dst

    return schedule


async def schedule_daemon_dispatch_impl(
    clients: List[Client],
    schedule: npt.NDArray[np.int32],
):
    await asyncio.gather(
        *clients[0].pause_and_resume_flow_unchecked(schedule),
        *clients[1].pause_and_resume_flow_unchecked(schedule),
    )
