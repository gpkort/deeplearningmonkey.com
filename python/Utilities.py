import numpy as np


def get_mean(self, arr: list) -> float:
    mean = 0.0
    for i in arr:
        mean += i
    return mean / len(self._xmean)

