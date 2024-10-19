import numpy as np
import numpy.typing as npt
import numba

import common

PORT_NUM = 4
TOR_NUM = 8
RANK_NUM = int(TOR_NUM / PORT_NUM)
assert RANK_NUM * PORT_NUM == TOR_NUM
SLICE_NUM = TOR_NUM * 2
Q_NUM = RANK_NUM

SLICE_PER_CONNECTION = PORT_NUM * 2


default_schedule = np.loadtxt(
    "runtime/8tors_1ports_schedule.txt",
    dtype=int,
).T

electrical_port = 0xAA


@numba.jit(nopython=True, cache=True)
def find_direct_port_slice_or_hardcoded_electrical(
    src: int, dst: int, schedule: npt.NDArray[int] = default_schedule
):
    coloum = src * PORT_NUM

    connection_for_port = schedule[coloum : coloum + PORT_NUM]
    slice_port = []

    for port, dst_list in enumerate(connection_for_port):
        if dst not in dst_list:
            continue

        for slice_id in range(SLICE_NUM):
            if dst_list[slice_id] != dst:  # there is direct link
                continue

            slice_port.append((slice_id, slice_id, port))

    return slice_port


def find_direct_port_slice_or_electrical(
    src: int,
    dst: int,
    is_hardcoded: bool = False,
    schedule=default_schedule,
):
    coloum = src * PORT_NUM
    # print(schedule.T[coloum])
    connection_for_port = schedule[coloum : coloum + PORT_NUM].tolist()

    # print(connection_for_port)

    slice_port = []

    for port, dst_list in enumerate(connection_for_port):
        if dst in dst_list:
            start = dst_list.index(dst)
            if (
                start == 0 and dst_list[-1] != -1
            ):  # 0 is not when connection to dst starts
                half_length = int(len(dst_list) / 2)
                start = dst_list[half_length::].index(dst) + half_length

            for slice_id in range(SLICE_NUM):
                if connection_for_port[port][slice_id] == dst:  # there is direct link
                    slice_port.append((slice_id, slice_id, port))
                elif not is_hardcoded:
                    slice_port.append((slice_id, start, electrical_port))

    if len(slice_port) == 0 and not is_hardcoded:
        # print(f"src {src} dst {dst} electrical")
        for slice_id in range(SLICE_NUM):
            slice_port.append((slice_id, slice_id, electrical_port))

    return slice_port


def find_new_slice_ta(src: int, port: int, time_slice: int, schedule=default_schedule):
    column = src * PORT_NUM + port
    return schedule[column, time_slice]
