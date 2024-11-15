import numpy as np
import numpy.typing as npt
import socket
import struct

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
        self.value, dst_ip, self.count = struct.unpack_from("III", buffer, offset)
        self.tor: str = socket.inet_ntoa(struct.pack("I", dst_ip))

    def __str__(self) -> str:
        return f"tor: {self.tor}, value: {self.value}, count: {self.count}"


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
        self.__n_flows = np.zeros((len(self.__tors), len(self.__tors)), dtype=np.int32)

        self.__counter = 0

    def update(
        self, source: str, report_entries: Sequence[ReportEntry]
    ) -> npt.NDArray[np.int32]:
        source_tor = self.__relations[source]
        source_tor_index = self.__tors.index(source_tor)

        old_matrix = np.copy(self.__matrix)

        for entry in report_entries:
            target_tor = self.__relations[entry.tor]
            target_tor_index = self.__tors.index(target_tor)

            self.__matrix[source_tor_index][target_tor_index] = entry.value
            if entry.value > 0:
                self.__n_flows[source_tor_index][target_tor_index] = entry.count
            else:
                self.__n_flows[source_tor_index][target_tor_index] = 0

        self.__counter += 1

        delta = self.__matrix - old_matrix
        return delta

    def matrix(self) -> npt.NDArray[np.int32]:
        return self.__matrix

    def n_flows(self) -> npt.NDArray[np.int32]:
        return self.__n_flows

    def counter(self) -> int:
        return self.__counter
