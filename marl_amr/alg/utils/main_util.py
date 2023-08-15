import os
import random

import numpy as np
import subprocess as sp
import tensorflow as tf


def set_global_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def convert_hyperparam_value_to_str(val):

    if isinstance(val, list):
        str_value = '_'.join([str(x) for x in val])
    elif isinstance(val, float):
        str_value = str(val).replace('.', 'p')
    else:
        str_value = str(val)

    return str_value


def create_log_dir(config_main):

    pass


def get_gpu_assignment(n_seeds, min_memory=512):
    """Assigns a gpu to each seed.

    Args:
        n_seeds: int
        min_memory: int

    Returns
        assignment: list of ints, assignment[i] is gpu number assigned to seed i
    """
    cmd = 'nvidia-smi --query-gpu=memory.free --format=csv'

    # e.g. b'memory.free [MiB]\n12208 MiB\n520 MiB\n12208 MiB\n12208 MiB\n'
    output = sp.check_output(cmd.split())
    # e.g. ['memory.free [MiB]', '12208 MiB', '520 MiB', '12208 MiB', '12208 MiB']
    output = output.decode('ascii').split('\n')[:-1]
    list_str = output[1:]
    list_int = [int(s.split(' ')[0]) for s in list_str]
    list_valid = [gpu_id for gpu_id, mem in enumerate(list_int)
                  if mem >= min_memory]

    num_available_gpu = len(list_valid)
    assignment = [0] * n_seeds
    for idx in range(n_seeds):
        # circular wrap
        assignment[idx] = list_valid[idx % num_available_gpu]

    return assignment
