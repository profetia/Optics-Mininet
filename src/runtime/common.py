import contextlib
import threading
import time


@contextlib.contextmanager
def Timer(name: str):
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()

    print(f"{name} took {(end - start) / 1_000_000} ms")


class Repeat(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
