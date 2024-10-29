import numpy as np
import numpy.typing as npt
import socket
import struct

from collections import defaultdict
from typing import Sequence, Mapping


ReportHeader = bytes(
    [
        0xAB,
        0xBC,
        0xCD,
        0x00,
        0xDE,
        0xEF,
        0xFA,
        0x00,
        0xFE,
        0xED,
        0xDC,
        0x00,
        0xCA,
        0xBA,
        0xAF,
        0x00,
    ]
)


class ReportEntry:
    def __init__(self, buffer: bytes, offset: int = 0):
        self.count, tor, _ = struct.unpack_from("QII", buffer, offset)
        self.target: str = socket.inet_ntoa(struct.pack("I", socket.ntohl(tor)))

    def __str__(self) -> str:
        return f"target: {self.target}, count: {self.count}"


class ReportFlags:

    def __init__(self, buffer: bytes, offset: int = 0):
        self.flags, *self.reserved = struct.unpack_from("B15x", buffer, offset)

    def __str__(self) -> str:
        return f"flags: {self.flags}"


class Report:

    def __init__(
        self,
        hosts: Sequence[str],
        tors: Sequence[str],
        relations: Mapping[str, str],
    ):
        self.__hosts = hosts
        self.__tors = tors
        self.__relations = relations

        self.__matrix = np.zeros((len(self.__tors), len(self.__tors)), dtype=np.int32)
        self.__traffic = {host: defaultdict(int) for host in self.__hosts}

        self.__counter = 0

    def update(
        self, source: str, report_entries: Sequence[ReportEntry]
    ) -> npt.NDArray[np.int32]:
        source_tor = self.__relations[source]
        source_tor_index = self.__tors.index(source_tor)

        delta = np.zeros((len(self.__tors), len(self.__tors)), dtype=np.int32)

        for entry in report_entries:
            target_tor = self.__relations[entry.target]
            target_tor_index = self.__tors.index(target_tor)

            delta_local = entry.count - self.__traffic[source][entry.target]

            delta[source_tor_index][target_tor_index] = delta_local
            self.__matrix[source_tor_index][target_tor_index] += delta_local

            self.__traffic[source][entry.target] = entry.count

        self.__counter += 1

        return delta

    def matrix(self) -> npt.NDArray[np.int32]:
        return self.__matrix

    def counter(self) -> int:
        return self.__counter
