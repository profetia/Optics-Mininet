import argparse
import asyncio
import uvloop

from runtime import core
from runtime.stub import consts

from runtime.rpc import Client, Host


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    clients = [
        Client(host=Host.Uranus),
        Client(host=Host.Neptune),
    ]

    n_tors = 8
    schedules = []
    for shift in range(n_tors):
        schedule = set()
        for i in range(n_tors):
            schedule.add((i, (i + shift) % n_tors))

        schedules.append(
            (
                (
                    shift * consts.SLICE_NUM // n_tors,
                    (shift + 1) * consts.SLICE_NUM // n_tors - 1,
                ),
                schedule,
            )
        )

    # print(schedules)

    _, schedule_entries_all = core.translate_matrix_discrete(
        n_tors,
        set(),
        schedules,
    )

    await asyncio.gather(
        *clients[0].clear_unchecked(),
        *clients[1].clear_unchecked(),
    )

    await asyncio.gather(
        *clients[0].pause_and_resume_flow_unchecked(
            schedule_entries_all[: n_tors // 2],
        ),
        *clients[1].pause_and_resume_flow_unchecked(
            schedule_entries_all[n_tors // 2 :],  # noqa: E203
        ),
    )


if __name__ == "__main__":
    uvloop.install()

    args = parse_args()
    asyncio.run(main())
