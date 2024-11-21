import asyncio
import dataclasses
import logging
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
import queue
import socket
import struct
import threading as th
import time
import uvloop

from typing import Any, Callable, List, Optional, Set, Tuple, Union

from runtime import common

from runtime.rpc import Client, Host
from runtime.report import Report, ReportHeader, ReportEntry
from runtime.stub import consts
from runtime.statistics import RunningStatistics

from runtime.proto.rpc_pb2 import ScheduleEntry, ScheduleEntryType

UnifiedTopology = Set[Tuple[int, int]]
"""A unified schedule is a set of tuples, where each tuple contains the source
and destination tor id.
The range for this schedule will be set to 0..=consts.TOR_NUM - 1.
"""

DiscreteTopology = List[Tuple[Tuple[int, int], Set[Tuple[int, int]]]]
"""A discrete schedule is a list of tuples, where each tuple contains a range
and a unified schedule for that range."""

EventHandler = Callable[
    [int, npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]],
    Optional[Any],
]
"""An event handler function takes the event counter, the traffic matrix, the
number of flows and the delta matrix as input and returns an optional auxiliary
data structure. If the event handler returns None, a workload will not be
scheduled. Otherwise, the traffic matrix and the auxiliary data structure will
be passed to the schedule function.
"""

TimingHandler = Callable[
    [
        float,
        npt.NDArray[np.int32],
        npt.NDArray[np.int32],
        Optional[npt.NDArray[np.float64]],
    ],
    Optional[Any],
]
"""A timing handler function takes the invocation time, the traffic matrix,
the number of flows and the variation matrix as input and returns an optional
auxiliary data structure. If the timing handler returns None, a workload will
not be scheduled. Otherwise, the traffic matrix and the auxiliary data
structure will be passed to the schedule function.
"""

Scheduler = Callable[
    [npt.NDArray[np.int32], npt.NDArray[np.int32], Any],
    UnifiedTopology | DiscreteTopology,
]
"""A scheduler function takes the traffic matrix, the number of flows and the
auxiliary data structure as input and returns a new schedule matrix.
"""


@dataclasses.dataclass
class Config:
    address: Tuple[str, int]
    """The address of the collector daemon."""

    report_kwargs: dict
    """The keyword arguments for the report object."""

    include_statistics: bool = False
    """Whether to collect statistics while scheduling."""

    clear_default: bool = False
    """Whether to clear the default flow tables on switches."""

    use_process: bool = True
    """Whether to run the scheduler in a separate process."""


logger = logging.getLogger(__name__)


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
        channel = mp.Queue() if self.config.use_process else queue.Queue()

        schedule = (
            mp.Process(
                target=schedule_daemon, args=(channel, self.config, self.scheduler)
            )
            if self.config.use_process
            else th.Thread(
                target=schedule_daemon, args=(channel, self.config, self.scheduler)
            )
        )

        schedule.start()

        collect_daemon(channel, self.config, self.event_handlers, self.timing_handlers)

        schedule.join()


REPORT_HEADER_LEN = len(ReportHeader)
REPORT_ENTRY_SIZE = struct.calcsize("III")


def collect_daemon_timing_impl(
    channel: Union[mp.Queue, queue.Queue],
    report: Report,
    timing_handler: TimingHandler,
    stat: Optional[RunningStatistics] = None,
) -> None:
    now = time.time()

    variance = stat.variance() if stat is not None else None

    matrix = report.matrix()
    n_flows = report.n_flows()

    if stat is not None:
        stat.reset()
        stat.update(matrix)

    auxiliary = timing_handler(now, matrix, n_flows, variance)
    if auxiliary is None:
        return

    # channel.put((now, time.time_ns(), matrix, n_flows, auxiliary))
    channel.put((matrix, n_flows, auxiliary))


def collect_daemon(
    channel: Union[mp.Queue, queue.Queue],
    config: Config,
    event_handlers: List[EventHandler],
    timing_handlers: List[Tuple[dict, TimingHandler]],
) -> None:
    report = Report(**config.report_kwargs)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(config.address)
    udp.setblocking(False)

    if config.include_statistics:
        stats: List[RunningStatistics] = [
            RunningStatistics(shape=report.matrix().shape) for _ in timing_handlers
        ]
    else:
        stats = [None for _ in timing_handlers]

    for (timing_kwargs, handler), stat in zip(timing_handlers, stats):
        common.Repeat(
            **timing_kwargs,
            function=collect_daemon_timing_impl,
            args=(channel, report, handler, stat),
        ).start()

    while True:
        try:
            data, (source, _) = udp.recvfrom(
                REPORT_HEADER_LEN + 1024 * REPORT_ENTRY_SIZE
            )
        except BlockingIOError:
            continue

        header = data[:REPORT_HEADER_LEN]  # the first 16 bytes are the header
        if header != ReportHeader:
            continue

        payload = data[REPORT_HEADER_LEN:]  # noqa: E203
        # the rest of the data is the payload

        report_entries_len = len(payload) // REPORT_ENTRY_SIZE
        report_entries = [
            ReportEntry(payload, i * REPORT_ENTRY_SIZE)
            for i in range(report_entries_len)
        ]

        delta = report.update(source, report_entries)

        matrix = report.matrix()
        n_flows = report.n_flows()

        if config.include_statistics:
            for stat in stats:
                stat.update(matrix)

        count = report.counter()
        for handler in event_handlers:
            auxiliary = handler(count, matrix, n_flows, delta)
            if auxiliary is None:
                continue

            # channel.put((count, time.time_ns(), matrix, n_flows, auxiliary))
            channel.put((matrix, n_flows, auxiliary))


def schedule_daemon(
    channel: Union[mp.Queue, queue.Queue],
    config: Config,
    scheduler: Scheduler,
) -> None:
    uvloop.install()
    with asyncio.Runner() as runner:

        old_topology = set()
        clients = [Client(Host.Uranus), Client(Host.Neptune)]

        if config.clear_default:
            runner.run(schedule_daemon_clear_default_impl(clients))

        while True:
            matrix, n_flows, auxiliary = channel.get()
            # event, moment, matrix, n_flows, auxiliary = channel.get()

            # print(f"Event {event} delayed by {(time.time_ns() - moment) / 1e3:.2f} us")
            # print(f"Incoming matrix:\n{matrix}")

            new_topology = scheduler(matrix, n_flows, auxiliary)

            if isinstance(new_topology, set) and new_topology.issubset(old_topology):
                logger.info(f"{old_topology} -> Skip")
                continue

            merged_topology, schedule_entries_all = translate_matrix(
                matrix.shape[0], old_topology, new_topology
            )

            # with common.Timer("schedule_daemon_dispatch_impl"):
            runner.run(schedule_daemon_dispatch_impl(clients, schedule_entries_all))

            logger.info(f"{old_topology} -> {merged_topology}")
            old_topology = merged_topology


def translate_matrix_unified(
    n_tors: int, old_topology: UnifiedTopology, new_topology: UnifiedTopology
) -> Tuple[UnifiedTopology, List[List[ScheduleEntry]]]:
    schedule_entries_all = [[] for _ in range(n_tors)]
    schedule_map = [-1 for _ in range(n_tors)]

    for src, dst in old_topology:
        schedule_map[src] = dst

    for src, dst in new_topology:
        old_dst = schedule_map[src]
        if old_dst == dst:
            continue

        schedule_map[src] = dst

        if old_dst != -1:
            schedule_entries_all[src].append(
                ScheduleEntry(
                    type=ScheduleEntryType.FixedRangeRemove,
                    start=0,
                    end=consts.SLICE_NUM - 1,
                    target_tor=old_dst,
                )
            )

        schedule_entries_all[src].append(
            ScheduleEntry(
                type=ScheduleEntryType.FixedRangeAdd,
                start=0,
                end=consts.SLICE_NUM - 1,
                target_tor=dst,
            )
        )

    merged_topology = set()
    for i in range(n_tors):
        if schedule_map[i] == -1:
            continue

        merged_topology.add((i, schedule_map[i]))

    return merged_topology, schedule_entries_all


def translate_matrix_discrete(
    n_tors: int, old_topology: DiscreteTopology, new_topology: DiscreteTopology
) -> Tuple[DiscreteTopology, List[List[ScheduleEntry]]]:
    schedule_entries_all = [[] for _ in range(n_tors)]

    for (start, end), topology in new_topology:
        for src, dst in topology:
            schedule_entries_all[src].append(
                ScheduleEntry(
                    type=ScheduleEntryType.GeneralUnspecified,
                    start=start,
                    end=end,
                    target_tor=dst,
                )
            )

    return new_topology, schedule_entries_all


def translate_matrix(
    n_tors: int,
    old_topology: UnifiedTopology | DiscreteTopology,
    new_topology: UnifiedTopology | DiscreteTopology,
) -> Tuple[UnifiedTopology | DiscreteTopology, List[List[ScheduleEntry]]]:

    if isinstance(new_topology, set):
        # old_topology = [((0, consts.SLICE_NUM - 1), old_topology)]
        # new_topology = [((0, consts.SLICE_NUM - 1), new_topology)]
        # return translate_matrix_discrete(n_tors, old_topology, new_topology)
        return translate_matrix_unified(n_tors, old_topology, new_topology)

    return translate_matrix_discrete(n_tors, old_topology, new_topology)


async def schedule_daemon_clear_default_impl(clients: List[Client]):
    await asyncio.gather(
        *clients[0].clear_unchecked(),
        *clients[1].clear_unchecked(),
    )


async def schedule_daemon_dispatch_impl(
    clients: List[Client],
    schedule_entries_all: List[List[ScheduleEntry]],
):
    await asyncio.gather(
        *clients[0].pause_and_resume_flow_unchecked(
            schedule_entries_all[: consts.TOR_NUM // 2]
        ),
        *clients[1].pause_and_resume_flow_unchecked(
            schedule_entries_all[consts.TOR_NUM // 2 :]  # noqa: E203
        ),
    )
