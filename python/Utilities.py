import numpy as np
import math


def get_mean(arr: list) -> float:
    if arr is None or len(arr) == 0:
        return None
    try:
        mean = 0.0
        for i in arr:
            mean += i
        return mean / len(arr)
    except Exception:
        return None


def get_std_dev(arr: list, mean=None):
    sigma = get_sigma(arr, mean)

    if sigma is not None:
        return sigma

    return None


def get_sigma(arr: list, mean=None):
    if arr is None or len(arr) == 0:
        return None
    try:
        mu = mean if mean is not None else get_mean(arr)
        err = 0.0

        for i in arr:
            err += (i - mu)**2

        return math.sqrt(err / len(arr))
    except Exception:
        return None

def get_sum(arr: list):
    if arr is None or len(arr) == 0:
        return None
    try:
        sum = 0.0
        for i in arr:
            sum += i
        return sum
    except Exception:
        return None
