import grpc
import time

from concurrent.futures import ThreadPoolExecutor

from proto.rpc_pb2 import (
    PauseFlowRequest,
    SetAfcTableEntry,
    ResumeFlowRequest,
    HohoLookupSendSliceTableEntry,
    SliceToDirectTorIpTableEntry,
)
from proto.rpc_pb2_grpc import RpcStub
import stub.collect as collect


class Client:

    def __init__(self, url: str, do_benchmark: bool = False):
        self.url = url
        self.do_benchmark = do_benchmark

        self.channels, self.stubs = [], []
        for i in range(4):
            channel = grpc.insecure_channel(f"{url}:{4000 + i}")
            self.channels.append(channel)
            self.stubs.append(RpcStub(channel))

        self.executor = ThreadPoolExecutor(max_workers=4)

    def __del__(self):
        for channel in self.channels:
            channel.close()

        self.executor.shutdown()

    def __pause_flow_impl(self, tor_id: int) -> None:
        def __impl():
            stub = self.stubs[tor_id]
            set_afc_entries = collect.pause_flow_impl(tor_id)
            request = PauseFlowRequest(
                set_afc_table_entries=[
                    SetAfcTableEntry(**entry) for entry in set_afc_entries
                ]
            )
            stub.PauseFlow(request)

        if not self.do_benchmark:
            __impl()
            return

        start = time.time_ns()
        __impl()
        print(
            f"Pause flow for tor {tor_id} in {(time.time_ns() - start) / 1_000_000} ms"
        )

    def pause_flow(self) -> None:
        futures = []
        for tor_id in range(4):
            futures.append(self.executor.submit(self.__pause_flow_impl, tor_id))

        for future in futures:
            future.result()

    def __resume_flow_impl(self, tor_id: int) -> None:
        def __impl():
            stub = self.stubs[tor_id]
            hoho_lookup_send_slice_entries, slice_to_direct_tor_ip_entries = (
                collect.resume_flow_impl(tor_id)
            )

            request = ResumeFlowRequest(
                hoho_lookup_send_slice_table_entries=[
                    HohoLookupSendSliceTableEntry(**entry)
                    for entry in hoho_lookup_send_slice_entries
                ],
                slice_to_direct_tor_ip_table_entries=[
                    SliceToDirectTorIpTableEntry(**entry)
                    for entry in slice_to_direct_tor_ip_entries
                ],
            )
            stub.ResumeFlow(request)

        if not self.do_benchmark:
            __impl()
            return

        start = time.time_ns()
        __impl()
        print(
            f"Resume flow for tor {tor_id} in {(time.time_ns() - start) / 1_000_000} ms"
        )

    def resume_flow(self) -> None:
        futures = []
        for tor_id in range(4):
            futures.append(self.executor.submit(self.__resume_flow_impl, tor_id))

        for future in futures:
            future.result()


Hosts = {
    "switch6-neptune": "10.0.13.24",
}

if __name__ == "__main__":
    cli = Client(url=Hosts["switch6-neptune"], do_benchmark=True)
    cli.pause_flow()
    cli.resume_flow()
