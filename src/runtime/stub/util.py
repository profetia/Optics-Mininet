import numpy as np

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
)

electrical_port = 0xAA


def find_direct_port_slice_or_electrical(
    src: int, dst: int, schedule: np.ndarray = default_schedule
):
    coloum = src * PORT_NUM
    # print(schedule.T[coloum])
    connection_for_port = schedule.T[coloum : coloum + PORT_NUM].tolist()

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
                else:
                    slice_port.append((slice_id, start, electrical_port))

    if len(slice_port) == 0:
        # print(f"src {src} dst {dst} electrical")
        for slice_id in range(SLICE_NUM):
            slice_port.append((slice_id, slice_id, electrical_port))

    return slice_port


def find_pause_slice(
    src: int, port: int, queue: int, schedule: np.ndarray = default_schedule
):
    assert 0 <= src <= TOR_NUM
    assert 0 <= port <= PORT_NUM
    assert 0 <= queue <= Q_NUM

    coloum = src * PORT_NUM + port
    # print(schedule.T[coloum])
    connection_for_port = schedule.T[coloum].tolist()

    # print(connection_for_port)
    start_slice = SLICE_PER_CONNECTION * queue
    end_slice = start_slice + SLICE_PER_CONNECTION

    # print(f"start_slice is {start_slice}, end_slice is {end_slice}")
    # print(f"connection for port is {connection_for_port}")

    paused_slice = connection_for_port.index(-1, start_slice, end_slice)

    # print(f"Find paused slice {paused_slice}")

    return paused_slice


def find_new_slice_ta(src: int, port: int, time_slice: int, schedule=default_schedule):
    column = src * PORT_NUM + port
    return schedule[time_slice][column]
