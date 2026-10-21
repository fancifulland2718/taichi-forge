"""Existing consecutive-run kernels and their cold specialization contract.

Prepared/JIT calls and the module/sequence adapters select from the same helper
family. The adapters retain their own storage projection, prefix and tail policy.
"""


def head_kernel(*, with_indices=False, ndarray_mode=True):
    from taichi_forge import _kernels

    if with_indices:
        return (
            _kernels.rle_mark_boundaries_and_starts_ndarray
            if ndarray_mode
            else _kernels.rle_mark_boundaries_and_starts_field
        )
    return _kernels.rle_mark_boundaries_ndarray if ndarray_mode else _kernels.rle_mark_boundaries_field


def unique_helper(dtype, with_indices):
    """Provide zero-size host metadata without allocating or running helpers."""
    import numpy as np

    exemplar = {
        "keys": np.empty(0, dtype={"i32": "int32", "u32": "uint32", "i64": "int64", "u64": "uint64"}[dtype]),
        "flags": np.empty(0, dtype="int32"),
    }
    if with_indices:
        exemplar["starts"] = exemplar["flags"]
    return head_kernel(with_indices=with_indices), exemplar
