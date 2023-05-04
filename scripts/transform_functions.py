import pandas as pd
import numpy as np

# EPS = 1e-8
EPS = 0

# --- universal functions --- #

class UnZScore:
    def __call__(self, data: np.ndarray, mean, std):
        new_std = std.copy()
        new_std[mean < 1] = 1
        data = data * (new_std + EPS) + mean
        return data


class ZScore:
    def __call__(self, data: np.ndarray, mean, std):
        new_std = std.copy()
        new_std[mean < 1] = 1 # ! fixme
        data = (data - mean) / (new_std + EPS)
        return data

