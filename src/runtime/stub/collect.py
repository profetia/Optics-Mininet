import numpy as np
import socket
import struct

from . import afc
from . import consts
from . import util

import common

from runtime.proto.rpc_pb2 import (
    SetAfcTableEntry,
    ScheduleEntry,
    # HohoLookupSendSliceTableEntry,
    # SliceToDirectTorIpTableEntry,
)


def __ipv4_to_int(ipv4: str):
    # Convert IPv4 string to a packed binary format
    packed_ip = socket.inet_aton(ipv4)
    # Unpack the binary format into an integer
    return struct.unpack("!I", packed_ip)[0]


def __pause_port_queue_at_slice(tor_id, port, queue, slice_id, app_id):
    pause_afc_msg = afc.gen_pause_afc_msg(tor_id, port, queue)
    return SetAfcTableEntry(
        app_id=app_id,
        slice_id=slice_id,
        packet_id=1,
        afc_msg=int.from_bytes(bytes(pause_afc_msg), "big"),
    )


__afc_entries_cache = []
for tor_id in range(consts.TOR_NUM):
    set_afc_entries = []
    set_afc_entries.append(
        __pause_port_queue_at_slice(tor_id, port=0, queue=0, slice_id=0, app_id=1)
    )
    set_afc_entries.append(
        __pause_port_queue_at_slice(tor_id, port=0, queue=0, slice_id=0, app_id=3)
    )
    __afc_entries_cache.append(set_afc_entries)


def pause_flow_impl(tor_id: int):
    # set_afc_entries = []
    # set_afc_entries.append(
    #     __pause_port_queue_at_slice(tor_id, port=0, queue=0, slice_id=0, app_id=1)
    # )
    # set_afc_entries.append(
    #     __pause_port_queue_at_slice(tor_id, port=0, queue=0, slice_id=0, app_id=3)
    # )

    set_afc_entries = __afc_entries_cache[tor_id]

    return set_afc_entries


__host_ipv4_cache = []
for ip in consts.host_ip:
    host_ipv4 = __ipv4_to_int(ip)
    __host_ipv4_cache.append(host_ipv4)


def resume_flow_impl(tor_id: int, schedule: np.ndarray):
    return [ScheduleEntry(schedule_columns=row) for row in schedule]

    # hoho_lookup_send_slice_entries = []
    # for dst in range(consts.TOR_NUM):
    #     if dst == tor_id:
    #         continue

    #     port_slice_id = util.find_direct_port_slice_or_hardcoded_electrical(
    #         tor_id, dst, schedule=schedule
    #     )
    #     for cur_slice, send_slice, port in port_slice_id:
    #         hoho_lookup_send_slice_entries.append(
    #             HohoLookupSendSliceTableEntry(
    #                 cur_slice=cur_slice,
    #                 dst_group=dst + 0x10,
    #                 port=port,
    #                 next_tor=dst + 0x10,
    #                 slot=send_slice,
    #             )
    #         )

    # slice_to_direct_tor_ip_entries = [
    #     SliceToDirectTorIpTableEntry(
    #         cur_slice=slice_id,
    #         tor_ip=__host_ipv4_cache[
    #             util.find_new_slice_ta(
    #                 src=tor_id, port=0, time_slice=slice_id, schedule=schedule
    #             )
    #         ],
    #     )
    #     for slice_id in range(consts.SLICE_NUM)
    # ]

    # return hoho_lookup_send_slice_entries, slice_to_direct_tor_ip_entries
