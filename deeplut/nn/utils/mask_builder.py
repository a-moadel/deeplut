import torch
import math
import numpy as np
import bisect
import random
import itertools
import copy

def rand_in_range_execlude(range_limit: int, must_have_val: int, count: int, device: str):
    """This function generate random 1d tensor where its values choosed from range[0, range_limit) without replacment and it make sure must_have_val is present.

    Args:
        range_limit (int): range limit [0, range_limit)
        must_have_val (int): element that should be represented in the final result
        count (int): count of element in the final result
        device (str): target device of the results

    Returns:
        torch.Tensor: 1d torch tensor contains random count-1 elements from range [0,range_limit) U {must_have_val}
    """
    range_exec_must_have = np.append(np.arange(0, must_have_val), np.arange(must_have_val + 1, range_limit))
    choices = np.random.choice(len(range_exec_must_have), count, replace=False)
    result_np = np.append(range_exec_must_have[choices], must_have_val)
    return torch.from_numpy(result_np).long().to(device)

