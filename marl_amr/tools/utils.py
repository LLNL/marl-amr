"""Collection of commonly-used standalone functions in general."""

import numpy as np


def print_dict(d):
    """Prints a 1-level dictionary."""
    for k, v in d.items():
        if type(v) == float:
            print(k, '\t', '{.3e}'.format(v))
        elif type(v) == np.ndarray:
            with np.printoptions(precision=3):
                print(k, '\t', v)
        else:
            print(k, '\t', v)
