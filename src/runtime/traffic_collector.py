import anyio
import numpy as np
import socket
import struct

from collections import defaultdict
from typing import Sequence, Mapping


class ReportEntry:
    def __init__(self, buffer: bytes, offset: int):
        self.count, tor, _ = struct.unpack_from("QII", buffer, offset)
        self.target: str = socket.inet_ntoa(struct.pack("I", socket.ntohl(tor)))

    def __str__(self) -> str:
        return f"target: {self.target}, count: {self.count}"


class __TrafficCollectorImpl:

    def __init__(
        self, hosts: Sequence[str], tors: Sequence[str], relaions: Mapping[str, str]
    ):
        self.__hosts = hosts
        self.__tors = tors
        self.__relations = relaions

        self.__matrix = np.zeros((len(self.__tors), len(self.__tors)), dtype=int)
        self.__traffic = {host: defaultdict(int) for host in self.__hosts}

    def update(self, source: str, report: Sequence[ReportEntry]) -> np.ndarray:
        source_tor = self.__relations[source]
        for entry in report:
            target_tor = self.__relations[entry.target]
            self.__matrix[self.__tors.index(source_tor)][
                self.__tors.index(target_tor)
            ] += (entry.count - self.__traffic[source][entry.target])
            self.__traffic[source][entry.target] = entry.count

        return np.copy(self.__matrix)


async def TrafficCollector(
    hosts: Sequence[str],
    tors: Sequence[str],
    relations: Mapping[str, str],
    host: str = "0.0.0.0",
    port: int = 1599,
):
    traffic_collector_impl = __TrafficCollectorImpl(hosts, tors, relations)
    async with await anyio.create_udp_socket(
        family=socket.AF_INET, local_host=host, local_port=port
    ) as udp:
        async for packet, (source, _) in udp:
            entry_size = struct.calcsize("QII")
            report_len = len(packet) // entry_size

            report = [ReportEntry(packet, i * entry_size) for i in range(report_len)]
            yield traffic_collector_impl.update(source, report)
