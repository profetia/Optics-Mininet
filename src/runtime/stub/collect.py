import numpy as np
import socket
import struct

from . import afc
from . import consts
from . import util

from runtime.proto.rpc_pb2 import (
    SetAfcTableEntry,
    HohoLookupSendSliceTableEntry,
    SliceToDirectTorIpTableEntry,
)


def __ipv4_to_int(ipv4: str):
    # Convert IPv4 string to a packed binary format
    packed_ip = socket.inet_aton(ipv4)
    # Unpack the binary format into an integer
    return struct.unpack("!I", packed_ip)[0]


def pause_flow_impl(tor_id: int, schedule: np.ndarray):

    def pause_port_queue_at_slice(port, queue, slice_id, app_id):
        pause_afc_msg = afc.gen_pause_afc_msg(tor_id, port, queue)
        return SetAfcTableEntry(
            app_id=app_id,
            slice_id=slice_id,
            packet_id=1,
            afc_msg=int.from_bytes(bytes(pause_afc_msg), "big"),
        )

    set_afc_entries = []
    set_afc_entries.append(
        pause_port_queue_at_slice(port=0, queue=0, slice_id=0, app_id=1)
    )
    set_afc_entries.append(
        pause_port_queue_at_slice(port=0, queue=0, slice_id=0, app_id=3)
    )
    # print(f"Pause port 0 queue 0 at slice {0}")

    return set_afc_entries


def resume_flow_impl(tor_id: int, schedule: np.ndarray):
    hoho_lookup_send_slice_entries = []
    for dst in range(consts.TOR_NUM):
        if dst == tor_id:
            continue

        port_slice_id = util.find_direct_port_slice_or_electrical(
            tor_id, dst, schedule=schedule
        )
        for cur_slice, send_slice, port in port_slice_id:
            hoho_lookup_send_slice_entries.append(
                HohoLookupSendSliceTableEntry(
                    cur_slice=cur_slice,
                    dst_group=dst + 0x10,
                    port=port,
                    next_tor=dst + 0x10,
                    slot=send_slice,
                    alternate_port=port,
                    alternate_next_tor=dst + 0x10,
                    alternate_slot=send_slice,
                )
            )

    slice_to_direct_tor_ip_entries = []
    for slice_id in range(consts.SLICE_NUM):
        target_id = util.find_new_slice_ta(
            src=tor_id, port=0, time_slice=slice_id, schedule=schedule
        )
        host_ipv4 = __ipv4_to_int(consts.host_ip[target_id])
        slice_to_direct_tor_ip_entries.append(
            SliceToDirectTorIpTableEntry(cur_slice=slice_id, tor_ip=host_ipv4)
        )

    # print(f"Resume flow for tor {tor_id}")

    return hoho_lookup_send_slice_entries, slice_to_direct_tor_ip_entries
