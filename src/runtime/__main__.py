import argparse
import socket

from traffic_collector import TrafficCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="opsys_control")
    parser.add_argument(
        "-a", "--address", type=str, help="IPv4 address to bind to", default="0.0.0.0"
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Port number to bind to", default=1599
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    t = TrafficCollector({"10.29.1.141": "sw6", "10.29.1.131": "sw4"}, args.address, args.port)
    t.start()
    t.join()


if __name__ == "__main__":
    args = parse_args()
    main(args)
