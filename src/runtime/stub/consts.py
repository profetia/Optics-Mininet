SLICE_NUM = 32
Q_NUM = 2
PORT_NUM = 4

GUARDBAND = 0

REC_LIMIT = 1
# SLICE_DUATION_US = 50 - 1
SLICE_DURATION_US = 50
UPDATE_INETRVAL_NS = 50
# max_qdepth = (SLICE_DUATION_US * 1000) * PORT_NUM * 100 / 8

# qdiff = 0
ETHERTYPE_ROTATION = 0x3001

IP_PROTOCOLS_UDP = 17

TOR_NUM = 8

self_mac = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD

electrical_port = 0xAA

to_ocs = [
    [56, 64, 40, 48],  # 25/0-28/0
    [136, 144, 152, 160],  # 1/0 server8-nic0
    [320, 312, 304, 296],  # 9/0
    [400, 392, 416, 408],  # 17/0
    [56, 64, 40, 48],  # 25/0-28/0
    [136, 144, 152, 160],  # 1/0 server8-nic0
    [320, 312, 304, 296],  # 9/0
    [400, 392, 416, 408],  # 17/0
]

to_server = [[32], [176], [280], [424], [32], [176], [280], [424]]
to_electrical = [16, 192, 264, 440, 16, 192, 264, 440]

to_elec_port = 0xAA
to_server_logic_port = 0xFF

# 9-1, 10-1, 7-1, 8-1,
# 9-0, 10-0, 7-0, 8-0
servers_mac = [
    0x08C0EBD9E11B,
    0x08C0EBD9E10B,
    0x08C0EBD9E0AB,
    0x08C0EBC628DB,
    0x08C0EBD9E11A,
    0x08C0EBD9E10A,
    0x08C0EBD9E0AA,
    0x08C0EBC628DA,
]

# 11-0, 11-1, 12-0, 12-1,
# 13-0, 13-1, 14-0, 14-1,
servers_mac = [
    0xE8EBD3EDC5EE,
    0xE8EBD3EDC5EF,
    0xE8EBD3E8B6A4,
    0xE8EBD3E8B6A5,
    0xE8EBD3E8B6C4,
    0xE8EBD3E8B6C5,
    0xE8EBD3EDC53E,
    0xE8EBD3EDC53F,
]

host_ip = [
    "10.29.1.91",
    "10.29.1.101",
    "10.29.1.71",
    "10.29.1.81",
    "10.29.1.90",
    "10.29.1.100",
    "10.29.1.70",
    "10.29.1.80",
]

host_ip = [
    "10.29.1.110",
    "10.29.1.111",
    "10.29.1.120",
    "10.29.1.121",
    "10.29.1.130",
    "10.29.1.131",
    "10.29.1.140",
    "10.29.1.141",
]


bandwidth0 = 40
bandwidth1 = 40
bandwidth2 = 40
bandwidth3 = 40


bandwidth0 = 100
bandwidth1 = 100
bandwidth2 = 10
bandwidth3 = 10


bandwidth_elect = 100


bandwidth0 = 25
bandwidth1 = 25
bandwidth2 = 25
bandwidth3 = 25


bandwidth0 = 100
bandwidth1 = 100
bandwidth2 = 100
bandwidth3 = 100


# bandwidth0 = 10
# bandwidth1 = 10
# bandwidth2 = 10
# bandwidth3 = 10


fp_port_configs = [
    # ToR0
    ("25/0", f"{bandwidth0}G", "NONE", 2),  # port0, ocs
    ("26/0", f"{bandwidth1}G", "NONE", 2),  # port1, ocs
    ("27/0", f"{bandwidth2}G", "NONE", 2),  # port2, ocs
    ("28/0", f"{bandwidth3}G", "NONE", 2),  # port3, ocs
    ("30/0", "100G", "RS", 2),  # sv9-1
    # ('31/0', f'{bandwidth3}G', 'NONE', 2), #Liam
    ("32/0", f"{bandwidth_elect}G", "NONE", 2),  # electrical
    # ('31/0', '100G', 'NONE', 2), #sw3-port60, rotation trigger port
    # tor1
    ("1/0", f"{bandwidth0}G", "NONE", 2),  # port0, ocs
    ("2/0", f"{bandwidth1}G", "NONE", 2),  # port1, ocs
    ("3/0", f"{bandwidth2}G", "NONE", 2),  # port2, ocs
    ("4/0", f"{bandwidth3}G", "NONE", 2),  # port3, ocs
    ("6/0", "100G", "RS", 2),  # sv10-1
    ("8/0", f"{bandwidth_elect}G", "NONE", 2),  # electrical
    # ('7/0', '100G', 'NONE', 2), #sw3-port57, rotation trigger port
    # tor2
    ("9/0", f"{bandwidth0}G", "NONE", 2),  # port0, ocs
    ("10/0", f"{bandwidth1}G", "NONE", 2),  # port1, ocs
    ("11/0", f"{bandwidth2}G", "NONE", 2),  # port2, ocs
    ("12/0", f"{bandwidth3}G", "NONE", 2),  # port3, ocs
    ("14/0", "100G", "RS", 2),  # sv10-1
    ("16/0", f"{bandwidth_elect}G", "NONE", 2),  # electrical
    # ('15/0', '100G', 'NONE', 2), #sw3-port58, rotation trigger port
    # tor3
    ("17/0", f"{bandwidth0}G", "NONE", 2),  # port0, ocs
    ("18/0", f"{bandwidth1}G", "NONE", 2),  # port1, ocs
    ("19/0", f"{bandwidth2}G", "NONE", 2),  # port2, ocs
    ("20/0", f"{bandwidth3}G", "NONE", 2),  # port3, ocs
    ("22/0", "100G", "RS", 2),  # sv8-1
    ("24/0", f"{bandwidth_elect}G", "NONE", 2),  # electrical
    # ('23/0', '100G', 'NONE', 2), #sw3-port59, rotation trigger port
]

bandwidth_after_vma0 = 18
bandwidth_after_vma1 = 18
bandwidth_after_vma2 = 18
bandwidth_after_vma3 = 18
