import numpy as np
import socket
import struct

from multiprocessing import Process
from typing import Callable

from runtime.report import Report, ReportEntry


TriggerFn = Callable[[np.array], bool]
ScheduleFn = Callable[[np.array], np.array]


def runtime_daemon(
    trigger_fn: TriggerFn,
    schedule_fn: ScheduleFn,
    host: str = "0.0.0.0",
    port: int = 1599,
    **kwargs,
):
    report = Report(**kwargs)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((host, port))

    while True:
        data, (source, _) = udp.recvfrom(1024)

        entry_size = struct.calcsize("QII")
        report_entries_len = len(data) // entry_size
        report_entries = [
            ReportEntry(data, i * entry_size) for i in range(report_entries_len)
        ]

        matrix = report.update(source, report_entries)
        if not trigger_fn(matrix):
            continue

        schedule_process = Process(
            target=schedule_daemon,
            args=(schedule_fn, np.copy(matrix)),
        )
        schedule_process.start()


def schedule_daemon(
    schedule_fn: ScheduleFn,
    matrix: np.array,
):
    topology = schedule_fn(matrix)

    # TODO


class Runtime(Process):
    def __init__(
        self,
        trigger_fn: TriggerFn,
        schedule_fn: ScheduleFn,
        host: str = "0.0.0.0",
        port: int = 1599,
        **kwargs,
    ):
        super().__init__(
            target=runtime_daemon,
            args=(trigger_fn, schedule_fn, host, port),
            kwargs=kwargs,
        )
