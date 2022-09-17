from typing import Tuple

import numba
import numpy as np


@numba.njit
def find_repeated(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find slices of repeated values, assuming the values are unique across slices.

    Returns three arrays containing the values, start indices and end indices, respectively.
    """
    num_unique = len(np.unique(array))
    values = np.zeros(num_unique, dtype=array.dtype)
    starts = np.full(num_unique, -1)
    ends = np.full(num_unique, -1)
    if num_unique == 0:
        return values, starts, ends

    slice_idx = 0
    starts[0] = 0
    values[0] = array[0]
    for i in range(len(array)):
        if i > 0 and array[i] != array[i - 1]:
            slice_idx += 1
            ends[slice_idx - 1] = starts[slice_idx] = i
            values[slice_idx] = array[i]
    ends[slice_idx] = len(array)

    return values, starts, ends


def split_ranges(
    starts: np.ndarray, ends: np.ndarray, max_len: int, min_len: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split given integer ranges to limit their length to the given maximum."""
    new_starts, new_ends, orig_starts = [], [], []
    assert len(starts) == len(ends)
    for start, end in zip(starts, ends):
        orig_start = start
        while start < end:
            length = min(end - start, max_len)
            if length >= min_len:
                orig_starts.append(orig_start)
                new_starts.append(start)
                new_ends.append(start + length)
            start += length
    return np.array(new_starts), np.array(new_ends), np.array(orig_starts)


@numba.njit
def nan_to_left(array):
    result = np.empty_like(array)
    last = np.nan
    for i in range(len(array)):
        if not np.isnan(array[i]):
            last = array[i]
        result[i] = last
    return result
