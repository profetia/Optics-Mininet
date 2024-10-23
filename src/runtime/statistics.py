import numpy as np
import numpy.typing as npt


from typing import Tuple


class RunningStatistics:

    def __init__(self, shape: Tuple[int, int]) -> None:
        self.__mean = np.zeros(shape, dtype=np.float32)
        self.__variance = np.zeros(shape, dtype=np.float32)
        self.__count = 0

    def update(self, matrix: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        self.__count += 1

        delta = matrix - self.__mean
        self.__mean += delta / self.__count
        self.__variance += delta * (matrix - self.__mean)

        return delta

    def mean(self) -> npt.NDArray[np.float32]:
        return self.__mean

    def variance(self) -> npt.NDArray[np.float32]:
        return self.__variance / self.__count

    def count(self) -> int:
        return self.__count

    def reset(self) -> None:
        self.__mean.fill(0)
        self.__variance.fill(0)
        self.__count = 0
