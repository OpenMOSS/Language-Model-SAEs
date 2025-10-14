import io

import numpy as np


def np_to_bytes(arr):
    with io.BytesIO() as buffer:
        np.save(buffer, arr)
        return buffer.getvalue()


def bytes_to_np(b):
    with io.BytesIO(b) as buffer:
        return np.load(buffer)
