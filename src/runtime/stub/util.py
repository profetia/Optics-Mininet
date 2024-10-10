import numpy as np

PORT_NUM = 4
TOR_NUM = 8
RANK_NUM = int(TOR_NUM / PORT_NUM)
assert RANK_NUM * PORT_NUM == TOR_NUM
SLICE_NUM = TOR_NUM * 2
Q_NUM = RANK_NUM

SLICE_PER_CONNECTION = PORT_NUM * 2

# column is port, row is slice
schedule = np.loadtxt(
    "runtime/8tors_1ports_schedule.txt",
    dtype=int,
)

active_slices = range(0, SLICE_NUM, 2)


def previous_slice(slice_id: int):
    return (slice_id + SLICE_NUM - 1) % SLICE_NUM


def find_dst_with_valid_slice(src: int, port: int, slice: int):
    # print(schedule)

    # print(f"input {src} {port} {slice}")
    coloum = src * 4 + port
    # print(f"coloum {coloum} slice {slice}")
    dst = schedule[slice][coloum]
    if dst == -1:
        slice = (slice + 1) % SLICE_NUM
        dst = schedule[slice][coloum]
    # print(f"dst {dst}")
    assert dst != -1
    # return 0, 0
    return slice, dst


def dst_to_port_slice(src: int, dst: int):

    coloum = src * PORT_NUM
    # print(schedule.T[coloum])
    connection_for_port = schedule.T[coloum : coloum + PORT_NUM].tolist()

    # print(connection_for_port)

    slice_port = []

    for port, dst_list in enumerate(connection_for_port):
        if dst in dst_list:
            # if src == 0:
            #    print(f"dst {dst}, port {port}")
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
                    slice_port.append((slice_id, start, port))

    # print(f"connection {connection_for_port}, dst {dst}")
    # print(slice_port)

    return slice_port


electrical_port = 0xAA


def find_direct_port_slice_or_electrical(src: int, dst: int):
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


def find_pause_slice(src: int, port: int, queue: int):
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


def find_new_slice(src: int, port: int, queue: int):
    pause_slice = find_pause_slice(src, port, queue)
    return (pause_slice + 1) % SLICE_NUM


def find_new_slice_tort(src: int, port: int, queue: int):
    pause_slice = find_pause_slice(src, port, queue)
    new_slice = (pause_slice + 1) % SLICE_NUM

    coloum = src * PORT_NUM + port
    connection_for_port = schedule.T[coloum].tolist()

    to_tor = connection_for_port[new_slice]
    # print(f"new slice {new_slice}, to tor {to_tor}")
    return (new_slice, to_tor)


def find_new_slice_ta(src: int, port: int, time_slice: int):
    column = src * PORT_NUM + port
    return schedule[time_slice][column]


def slice_to_rank(src: int, port: int, slice: int):

    for rank in range(Q_NUM):
        rank_start = find_pause_slice(src, port, rank)
        rank_end = (rank_start + SLICE_PER_CONNECTION) % SLICE_NUM
        # print(f"rank starts {rank_start} ends {rank_end} slice {slice}")

        if rank_start < rank_end:
            if rank_start <= slice and slice <= rank_end:
                # print(f"Slice {slice} to rank {rank}")
                return rank
        else:
            if rank_start <= slice or slice <= rank_end:
                # print(f"Slice {slice} to rank {rank}")
                return rank

    assert 0, f"Bug: src {src} port {port} slice {slice} is not mapped to any rank"


def dst_to_rank(src, dst) -> int:

    # return a rank.
    assert dst != src

    coloum = src * PORT_NUM
    # print(schedule.T[coloum])
    connection_for_port = schedule.T[coloum : coloum + PORT_NUM].tolist()

    # print(connection_for_port)

    rank = 0
    for port, dst_list in enumerate(connection_for_port):
        if dst in dst_list:
            for index in range(len(dst_list)):
                if dst_list[index] == dst:
                    break
                if dst_list[index] == -1:
                    rank += 1

    # print(rank)
    return rank
