import torch
import math
import numpy as np
import bisect
import random
import itertools
import copy

def generate_truth_table(k, tables_count, device) -> torch.Tensor:
    table = torch.from_numpy(np.array(list(itertools.product([-1, 1], repeat=k)))).T
    return torch.vstack([table] * tables_count).to(device)

def generate_random(n, rmv, cnt, device):
    rng = np.append(np.arange(0, rmv), np.arange(rmv + 1, n))
    choices = np.random.choice(len(rng), cnt, replace=False)
    return torch.from_numpy(np.append(rng[choices], rmv)).long().to(device)


def generate_input_mask_expanded(k, in_size, out_size, device):
    expand_mask = torch.zeros((out_size * in_size * k, 1), dtype=torch.int64).to(device)
    for j in range(out_size):
        for i in range(in_size):
            _from = torch.arange(k) + (i * k) + (j * in_size * k)
            _to = generate_random(in_size, i, k - 1, device).reshape(-1, 1)
            expand_mask[_from] = _to
    return expand_mask

def generate_input_mask_minimal(k, in_size, out_size, device):
    tables_count = math.ceil(in_size/k)
    expand_mask = torch.from_numpy(np.arange(tables_count*k*out_size)%in_size).to(device)
    return expand_mask

def generate_input_mask_shallow(k, in_size, number_of_tables, device):
    expanded_mask_size = number_of_tables * k
    result = np.random.choice(in_size, expanded_mask_size)
    return torch.from_numpy(result).long().to(device)


def generate_input_mask(k, in_size, out_size, device, minimal):
  if minimal:
    return generate_input_mask_minimal(k, in_size, out_size, device)
  else:
    return generate_input_mask_expanded(k, in_size, out_size, device)

def reduce_truth_table(k, table, device):
    full_row_count = table.shape[1]
    reduced_row_count = int(full_row_count / k)
    result = torch.ones(table.shape[0], reduced_row_count, 2 ** k).to(device)
    for i in range(k):
        result = result * table[:, i::k]
    return result.to(device)