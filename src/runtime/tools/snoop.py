import asyncio
import argparse

from runtime.traffic_collector import TrafficCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="opsys_control")
    parser.add_argument(
        "-a", "--address", type=str, help="IPv4 address to bind to", default="0.0.0.0"
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Port number to bind to", default=1599
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    host_x10 = "10.29.1.110"
    host_x11 = "10.29.1.111"
    host_x12 = "10.29.1.120"
    host_x13 = "10.29.1.121"
    host_x14 = "10.29.1.130"
    host_x15 = "10.29.1.131"
    host_x16 = "10.29.1.140"
    host_x17 = "10.29.1.141"

    hosts = [
        host_x10,
        host_x11,
        host_x12,
        host_x13,
        host_x14,
        host_x15,
        host_x16,
        host_x17,
    ]

    tor_0 = "tor0"
    tor_1 = "tor1"
    tor_2 = "tor2"
    tor_3 = "tor3"
    tor_4 = "tor4"
    tor_5 = "tor5"
    tor_6 = "tor6"
    tor_7 = "tor7"

    tors = [tor_0, tor_1, tor_2, tor_3, tor_4, tor_5, tor_6, tor_7]

    relations = {
        host_x10: tor_0,
        host_x11: tor_1,
        host_x12: tor_2,
        host_x13: tor_3,
        host_x14: tor_4,
        host_x15: tor_5,
        host_x16: tor_6,
        host_x17: tor_7,
    }

    host_x10_mapped = "172.16.11.10"
    host_x11_mapped = "172.16.11.11"
    host_x12_mapped = "172.16.12.10"
    host_x13_mapped = "172.16.12.11"
    host_x14_mapped = "172.16.13.10"
    host_x15_mapped = "172.16.13.11"
    host_x16_mapped = "172.16.14.10"
    host_x17_mapped = "172.16.14.11"

    host_map = {
        host_x10_mapped: host_x10,
        host_x11_mapped: host_x11,
        host_x12_mapped: host_x12,
        host_x13_mapped: host_x13,
        host_x14_mapped: host_x14,
        host_x15_mapped: host_x15,
        host_x16_mapped: host_x16,
        host_x17_mapped: host_x17,
    }

    async for matrix in TrafficCollector(
        hosts, tors, relations, host_map=host_map, host=args.address, port=args.port
    ):
        print(matrix)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
