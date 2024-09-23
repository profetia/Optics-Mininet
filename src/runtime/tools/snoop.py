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
    tor4 = "sw4"
    host_131 = "10.29.1.131"

    tor6 = "sw6"
    host_141 = "10.29.1.141"

    hosts = [
        host_131,
        host_141,
    ]
    tors = [tor4, tor6]
    relations = {
        host_131: tor4,
        host_141: tor6,
    }

    async for matrix in TrafficCollector(
        hosts, tors, relations, args.address, args.port
    ):
        print(matrix)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
