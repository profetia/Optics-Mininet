import contextlib
import time

from threading import Timer


@contextlib.contextmanager
def timing(name: str):
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()

    print(f"{name} took {(end - start) / 1_000_000} ms")


class Repeat(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
