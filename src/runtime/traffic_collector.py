import logging
import numpy as np
import socket
import struct
import threading

from collections import defaultdict


logger = logging.getLogger(__name__)


class ReportEntry:
    def __init__(self, buffer: bytes, offset: int):
        self.count, tor, _ = struct.unpack_from("QII", buffer, offset)
        self.target: str = socket.inet_ntoa(struct.pack("I", socket.ntohl(tor)))

    def __str__(self):
        return f"target: {self.target}, count: {self.count}"


class TrafficCollector(threading.Thread):
    def __init__(
        self,
        host_tor_map: dict[str, str],
        ipv4: str,
        port: int = 1599,
        buffer_size: int = 1024,
    ):
        self.__host_tor_map = host_tor_map
        self.__tor_host_map = defaultdict(list)
        for host, tor in self.__host_tor_map.items():
            self.__tor_host_map[tor].append(host)
        self.__tors = list(self.__tor_host_map.keys())

        self.__traffic_matrix = np.zeros((len(self.__tors), len(self.__tors)), dtype=int)
        self.__traffic_history = {
            host: defaultdict(int) for host in self.__host_tor_map.keys()
        }
        self.__traffic_lock = threading.Lock()

        super().__init__(target=self.__collect_daemon_fn, args=(ipv4, port, buffer_size))

    def __collect_daemon_fn(self, ipv4: str, port: int, buffer_size: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ipv4, port))

        while True:
            data, (source, _) = sock.recvfrom(buffer_size)
            entry_size = struct.calcsize("QII")
            report_len = len(data) // entry_size

            report = [ReportEntry(data, i * entry_size) for i in range(report_len)]
            self.__update_traffic_matrix(source, report)

    def __update_traffic_matrix(self, source: str, report: list[ReportEntry]):
        source_tor = self.__host_tor_map[source]
        with self.__traffic_lock:
            for entry in report:
                target_tor = self.__host_tor_map[entry.target]
                self.__traffic_matrix[self.__tors.index(source_tor)][
                    self.__tors.index(target_tor)
                ] += (entry.count - self.__traffic_history[source][entry.target])
                self.__traffic_history[source][entry.target] = entry.count

        logger.debug(f"Traffic Matrix:\n{self.__traffic_matrix}")

    def __enter__(self) -> bool:
        return self.__traffic_lock.__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        return self.__traffic_lock.__exit__(exc_type, exc_value, traceback)
    
    def __call__(self) -> np.ndarray:
        with self.__traffic_lock:
            return np.copy(self.__traffic_matrix)
