from scapy.all import Packet, BitField


from . import consts


class AFC(Packet):
    name = "adv_flow_ctl"
    fields_desc = [
        BitField("qfc", 0, 1),
        BitField("tm_pipe_id", 0, 2),
        BitField("tm_mac_id", 0, 4),
        BitField("pad", 0, 3),
        BitField("tm_mac_qid", 0, 7),
        BitField("credit", 0, 15),
    ]


def get_pg_id(dev_port: int):
    # each pipe has 64 dev_ports + divide by 8 to get the pg_id
    pg_id = (dev_port % 128) >> 3
    return pg_id


def get_pg_queue(dev_port: int, qid: int):
    lane = dev_port % 8
    pg_queue = lane * 16 + qid  # there are 16 queues per lane
    return pg_queue


def gen_pause_afc_msg(tor_id: int, port: int, q: int):
    return AFC(
        qfc=1,
        tm_pipe_id=tor_id % 4,
        tm_mac_id=get_pg_id(consts.to_ocs[tor_id][port]),
        tm_mac_qid=get_pg_queue(consts.to_ocs[tor_id][port], q),
        credit=1,  # stop
    )


def gen_resume_afc_msg(tor_id: int, port: int, q: int):
    return AFC(
        qfc=1,
        tm_pipe_id=tor_id % 4,
        tm_mac_id=get_pg_id(consts.to_ocs[tor_id][port]),
        tm_mac_qid=get_pg_queue(consts.to_ocs[tor_id][port], q),
        credit=0,  # resume
    )
