import datetime
import os
import shutil
import numpy as np


def setup_exp_dir(exp_dir: str, prefix: str = None, suffix: str = None):
    if prefix is not None:
        assert suffix is None
    if suffix is not None:
        assert prefix is None
    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if prefix is not None:
        exp_log_dir = os.path.join(exp_dir, prefix)
    elif suffix is not None:
        exp_log_dir = os.path.join(exp_dir, current_date_time + "_" + suffix)
    else:
        exp_log_dir = os.path.join(exp_dir, current_date_time)

    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_log_dir, "scripts"), exist_ok=True)
    source_files = os.listdir("./")
    for f in source_files:
        if f.endswith(".py") or f.endswith(".sh"):
            shutil.copy2(os.path.join("./", f), os.path.join(exp_log_dir, "scripts"))
    return exp_log_dir


class OutputValidator:
    @staticmethod
    def ndarray_in_dict(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            for k, v in result.items():
                if not isinstance(v, np.ndarray):
                    raise TypeError(f"{k} is not a numpy array!")
            return result

        return wrapper
