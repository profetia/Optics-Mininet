import numpy as np
import socket
import struct

from collections import defaultdict
from typing import Optional, Sequence, Mapping


class ReportEntry:
    def __init__(self, buffer: bytes, offset: int):
        self.count, tor, _ = struct.unpack_from("QII", buffer, offset)
        self.target: str = socket.inet_ntoa(struct.pack("I", socket.ntohl(tor)))

    def __str__(self) -> str:
        return f"target: {self.target}, count: {self.count}"


class Report:

    def __init__(
        self,
        hosts: Sequence[str],
        tors: Sequence[str],
        relations: Mapping[str, str],
        host_map: Optional[Mapping[str, str]] = None,
    ):
        self.__hosts = hosts
        self.__tors = tors
        self.__relations = relations

        if host_map is None:
            host_map = {host: host for host in hosts}

        self.__host_map = host_map

        self.__matrix = np.zeros((len(self.__tors), len(self.__tors)), dtype=int)
        self.__traffic = {host: defaultdict(int) for host in self.__hosts}

        self.__counter = 0

    def update(self, source: str, report_entries: Sequence[ReportEntry]) -> int:
        real_source = self.__host_map[source]
        source_tor = self.__relations[real_source]

        modified: int = 0
        for entry in report_entries:
            target_tor = self.__relations[entry.target]
            delta = entry.count - self.__traffic[real_source][entry.target]
            self.__matrix[self.__tors.index(source_tor)][
                self.__tors.index(target_tor)
            ] += delta
            modified += abs(delta)
            self.__traffic[real_source][entry.target] = entry.count

        if modified:
            self.__counter += 1

        return modified

    def matrix(self) -> np.ndarray:
        return self.__matrix

    def counter(self) -> int:
        return self.__counter
