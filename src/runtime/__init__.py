import numpy as np
import socket
import struct
import threading
import xmlrpc.client

from multiprocessing import Process
from typing import Any, Callable, Optional


from runtime.report import Report, ReportEntry


TriggerFn = Callable[[np.array], Optional[Any]]
ScheduleFn = Callable[..., np.array]


N_PORTS = 4
N_SLICES = 16
SLICE_DURATION_US = 50


def runtime_daemon(
    trigger_fn: TriggerFn,
    schedule_fn: ScheduleFn,
    host: str = "0.0.0.0",
    port: int = 1599,
    event_only: bool = False,
    schedule_kwargs: dict = {},
    report_kwargs: dict = {},
):
    report = Report(**report_kwargs)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((host, port))

    if not event_only:
        udp.settimeout(N_SLICES * SLICE_DURATION_US / 1e6)

    schedule_process = None

    while True:
        try:
            data, (source, _) = udp.recvfrom(1024)
            entry_size = struct.calcsize("QII")
            report_entries_len = len(data) // entry_size
            report_entries = [
                ReportEntry(data, i * entry_size) for i in range(report_entries_len)
            ]
            matrix = report.update(source, report_entries)
        except socket.timeout:
            matrix = report.matrix()

        auxiliary = trigger_fn(matrix)
        if auxiliary is None:
            continue

        if schedule_process is not None and schedule_process.is_alive():
            continue

        schedule_process = Process(
            target=schedule_daemon,
            args=(schedule_fn, np.copy(matrix), auxiliary),
            kwargs=schedule_kwargs,
        )
        schedule_process.start()


def schedule_daemon(
    schedule_fn: ScheduleFn,
    matrix: np.array,
    auxiliary: Any,
    **kwargs,
):
    topology = schedule_fn(matrix, auxiliary, **kwargs)

    n_tors = len(topology)

    new_schedule = np.full((N_SLICES, n_tors * N_PORTS), -1, dtype=int)

    for i, row in enumerate(topology):
        for j, value in enumerate(row):
            if not value:
                continue

            column = i * N_PORTS
            new_schedule[:, column] = j

    def dispatch_fn(url: str, schedule: list[list[int]]):
        with xmlrpc.client.ServerProxy(
            uri=f"http://{url}/",
            allow_none=True,
        ) as proxy:
            proxy.pause_flow()
            proxy.resume_flow(schedule)

    import time

    start = time.time_ns()
    new_schedule_list = new_schedule.tolist()
    tor_servers = ["10.0.13.24:8989", "10.0.13.23:8989"]
    threads = [
        threading.Thread(
            target=dispatch_fn,
            args=(url, new_schedule_list),
        )
        for url in tor_servers
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    end = time.time_ns()
    print(f"Dispatch time: {(end - start) / 1e6} ms")

    # print(f"Update new schedule: {new_schedule}")


class Runtime(Process):
    def __init__(
        self,
        trigger_fn: TriggerFn,
        schedule_fn: ScheduleFn,
        host: str = "0.0.0.0",
        port: int = 1599,
        event_only: bool = False,
        schedule_kwargs: dict = {},
        report_kwargs: dict = {},
    ):
        super().__init__(
            target=runtime_daemon,
            args=(
                trigger_fn,
                schedule_fn,
                host,
                port,
                event_only,
                schedule_kwargs,
                report_kwargs,
            ),
        )
