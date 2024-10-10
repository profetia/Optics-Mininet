import asyncio
import grpc
import numpy as np


from proto.rpc_pb2 import PauseFlowRequest, ResumeFlowRequest
from proto.rpc_pb2_grpc import RpcStub

import stub.collect as collect
import stub.util as util

import time


class Client:

    def __init__(self, url: str):
        self.channels: list[grpc.aio.Channel] = []
        self.stubs: list[RpcStub] = []
        for i in range(4):
            channel = grpc.aio.insecure_channel(f"{url}:{4000 + i}")
            self.channels.append(channel)
            self.stubs.append(RpcStub(channel))

    def __del__(self):
        for channel in self.channels:
            asyncio.create_task(channel.close())

    async def __pause_flow_impl(self, tor_id: int, schedule: np.ndarray) -> None:
        set_afc_entries = collect.pause_flow_impl(tor_id, schedule)
        request = PauseFlowRequest(set_afc_table_entries=set_afc_entries)

        await self.stubs[tor_id].PauseFlow(request)

    async def pause_flow(self, schedule: np.ndarray) -> None:
        await asyncio.gather(
            *[self.__pause_flow_impl(tor_id, schedule) for tor_id in range(4)]
        )

    async def __resume_flow_impl(self, tor_id: int, schedule: np.ndarray) -> None:

        hoho_lookup_send_slice_entries, slice_to_direct_tor_ip_entries = (
            collect.resume_flow_impl(tor_id, schedule)
        )
        request = ResumeFlowRequest(
            hoho_lookup_send_slice_table_entries=hoho_lookup_send_slice_entries,
            slice_to_direct_tor_ip_table_entries=slice_to_direct_tor_ip_entries,
        )

        await self.stubs[tor_id].ResumeFlow(request)

    async def resume_flow(self, schedule: np.ndarray) -> None:
        await asyncio.gather(
            *[self.__resume_flow_impl(tor_id, schedule) for tor_id in range(4)]
        )


Hosts = {
    "switch6-uranus": "10.0.13.23",
    "switch6-neptune": "10.0.13.24",
}


async def __main():
    cli = Client(url=Hosts["switch6-neptune"])

    start = time.time_ns()

    await asyncio.gather()

    await cli.pause_flow(util.default_schedule)
    await cli.resume_flow(util.default_schedule)

    end = time.time_ns()
    print(f"Time elapsed: {(end - start) / 1_000_000} ms")


if __name__ == "__main__":
    asyncio.run(__main())
