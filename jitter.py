import numpy as np


def jitter(data, jit=None):
    if jit is None:
        jit = 0.005

    a = data[0]
    b = data[1]

    duplicate_a = []
    duplicate_b = []
    jitter_a = [''] * len(a)
    jitter_b = [''] * len(b)
    jitter_a[0] = 0
    jitter_b[0] = 0

    for i in np.arange(1, len(a), 1):
        a_val = a.values[i]
        b_val = b.values[i]
        if a_val in a.values[0:i]:
            duplicate_a.append(a_val)
            val = jit * duplicate_a.count(a_val)
            jitter_a[i] = val
        else:
            jitter_a[i] = 0

        if b_val in b.values[0:i]:
            duplicate_b.append(b_val)
            val = jit * duplicate_b.count(b_val)
            jitter_b[i] = val
        else:
            jitter_b[i] = 0
    return jitter_a, jitter_b