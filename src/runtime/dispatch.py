import argparse
import numpy as np

import xmlrpc.client


def pause_flow(url: str):
    with xmlrpc.client.ServerProxy(
        uri=f"http://{url}/",
        allow_none=True,
    ) as proxy:
        proxy.pause_flow()


def __pause_flow_fn(args: argparse.Namespace) -> None:
    pause_flow(args.url)


def resume_flow(url: str, schedule: np.array):
    with xmlrpc.client.ServerProxy(
        uri=f"http://{url}/",
        allow_none=True,
    ) as proxy:
        proxy.resume_flow(schedule.tolist())


def __resume_flow_fn(args: argparse.Namespace) -> None:
    schedule = np.loadtxt(args.schedule)
    resume_flow(args.url, schedule)


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--url", type=str, help="URL of the XML-RPC server", required=True
    )

    subparsers = parser.add_subparsers()

    pause_flow_parser = subparsers.add_parser("pause_flow")
    pause_flow_parser.set_defaults(func=__pause_flow_fn)

    resume_flow_parser = subparsers.add_parser("resume_flow")
    resume_flow_parser.add_argument(
        "schedule", type=str, help="Path to the schedule file"
    )
    resume_flow_parser.set_defaults(func=__resume_flow_fn)

    return parser.parse_args()


if __name__ == "__main__":
    args = __parse_args()
    args.func(args)
