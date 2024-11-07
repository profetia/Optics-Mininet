import argparse
import asyncio
import logging

from runtime import rpc


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    clients = [rpc.Client(rpc.Host.Neptune), rpc.Client(rpc.Host.Uranus)]

    await asyncio.gather(
        *clients[0].reset_unchecked(),
        *clients[1].reset_unchecked(),
    )

    await asyncio.gather(*[client.close() for client in clients])


if __name__ == "__main__":
    import uvloop

    uvloop.install()

    args = parse_args()
    asyncio.run(main(args))
