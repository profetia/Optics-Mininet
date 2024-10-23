import asyncio
from typing import Awaitable, List
import grpc
import enum
import numpy as np
import numpy.typing as npt


from . import common
from .stub import collect


from .proto.rpc_pb2 import PauseFlowRequest, ResumeFlowRequest
from .proto.rpc_pb2_grpc import RpcStub


class Host(enum.Enum):
    Uranus = "10.0.13.23"
    Neptune = "10.0.13.24"


class Client:

    def __init__(self, host: Host):
        self.host = host
        self.tor_range = range(4) if host == Host.Uranus else range(4, 8)

        self.channels: list[grpc.aio.Channel] = []
        self.stubs: list[RpcStub] = []
        for i in range(4):
            channel = grpc.aio.insecure_channel(f"{host.value}:{4000 + i}")
            self.channels.append(channel)
            self.stubs.append(RpcStub(channel))

    async def close(self):
        for channel in self.channels:
            await channel.close()

    async def __pause_flow_impl(self, tor_id: int) -> None:
        set_afc_entries = collect.pause_flow_impl(tor_id)
        request = PauseFlowRequest(set_afc_table_entries=set_afc_entries)

        await self.stubs[tor_id % 4].PauseFlow(request)

    def pause_flow_unchecked(self) -> List[Awaitable[None]]:
        return [self.__pause_flow_impl(tor_id) for tor_id in self.tor_range]

    async def pause_flow(self) -> None:
        await asyncio.gather(*self.pause_flow_unchecked())

    async def __resume_flow_impl(
        self, tor_id: int, schedule: npt.NDArray[np.int32]
    ) -> None:
        # hoho_lookup_send_slice_entries, slice_to_direct_tor_ip_entries = (
        #     collect.resume_flow_impl(tor_id, schedule)
        # )
        request = ResumeFlowRequest(
            # hoho_lookup_send_slice_table_entries=hoho_lookup_send_slice_entries,
            # slice_to_direct_tor_ip_table_entries=slice_to_direct_tor_ip_entries,
            schedule_entries=schedule,
        )

        await self.stubs[tor_id % 4].ResumeFlow(request)

    def resume_flow_unchecked(
        self, schedule: npt.NDArray[np.int32]
    ) -> List[Awaitable[None]]:
        return [self.__resume_flow_impl(tor_id, schedule) for tor_id in self.tor_range]

    async def resume_flow(self, schedule: npt.NDArray[np.int32]) -> None:
        await asyncio.gather(*self.resume_flow_unchecked(schedule))

    async def __pause_and_resume_flow_impl(
        self, tor_id: int, schedule: npt.NDArray[np.int32]
    ) -> None:
        await self.__pause_flow_impl(tor_id)
        await self.__resume_flow_impl(tor_id, schedule)

    def pause_and_resume_flow_unchecked(
        self, schedule: npt.NDArray[np.int32]
    ) -> List[Awaitable[None]]:
        return [
            self.__pause_and_resume_flow_impl(tor_id, schedule)
            for tor_id in self.tor_range
        ]

    async def pause_and_resume_flow(self, schedule: npt.NDArray[np.int32]) -> None:
        await asyncio.gather(*self.pause_and_resume_flow_unchecked(schedule))


async def main():
    schedule = (
        np.loadtxt(
            "runtime/8tors_1ports_test.txt",
            dtype=int,
        )
        .astype(np.int32)
        .T.flatten()
    )

    neptune = Client(host=Host.Neptune)
    with common.Timer("Neptune"):
        await neptune.pause_flow()
        await neptune.resume_flow(schedule)
    await neptune.close()

    uranus = Client(host=Host.Uranus)
    with common.Timer("Uranus"):
        await uranus.pause_flow()
        await uranus.resume_flow(schedule)
    await uranus.close()


if __name__ == "__main__":
    import uvloop

    uvloop.install()
    asyncio.run(main())

    # import cProfile

    # cProfile.run("asyncio.run(main())", sort="cumtime")
