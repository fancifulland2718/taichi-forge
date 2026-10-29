"""Cold coordinate transfers shared by ordinary Fields and component export."""

from taichi_forge.lang import impl
from taichi_forge.lang.util import to_numpy_type


def element_type(field):
    from taichi_forge.lang.matrix import MatrixType, VectorType
    from taichi_forge.lang.struct import StructField

    if isinstance(field, StructField):
        raise TypeError("Coordinate transfers require a scalar/vector/matrix member Field")
    if hasattr(field, "n"):
        return VectorType(field.n, field.dtype) if field.ndim == 1 else MatrixType(field.n, field.m, 2, field.dtype)
    return field.dtype


def transfer(field, coordinates, values=None):
    import numpy as np
    from taichi_forge._kernels import field_read_coordinates, field_write_coordinates

    impl.get_runtime().materialize()
    dtype = element_type(field)
    rank = len(field.shape)
    indices = np.asarray(coordinates)
    if indices.size == 0:
        indices = np.empty((len(indices), rank), dtype=np.int32)
    if indices.ndim != 2 or indices.shape[1] != rank or indices.dtype.kind not in "iu":
        raise ValueError("Field coordinates require an integer array of shape (count, field rank)")
    offsets = tuple(field.snode.ptr.offset) or (0,) * rank
    for axis, (offset, extent) in enumerate(zip(offsets, field.shape)):
        if np.any(indices[:, axis] < offset) or np.any(indices[:, axis] >= offset + extent):
            raise ValueError("Field coordinate is outside its logical shape/offset")
    if np.any(indices < -(2**31)) or np.any(indices > 2**31 - 1):
        raise ValueError("Field coordinates must fit the existing signed-i32 index ABI")
    count = len(indices)
    element_shape = tuple(getattr(dtype, "get_shape", lambda: ())())
    supplied = None
    if values is not None:
        if len({tuple(row) for row in indices.tolist()}) != count:
            raise ValueError("Coordinate writes require distinct indices")
        supplied = np.asarray(values, dtype=to_numpy_type(field.dtype))
        if supplied.shape != (count, *element_shape):
            raise ValueError("Field coordinate values differ from count or element shape")
    if count == 0:
        return None if values is not None else np.empty((0, *element_shape), dtype=to_numpy_type(field.dtype))
    index_array = impl.ndarray(impl.i32, shape=(count, max(rank, 1)))
    data_array = impl.ndarray(dtype, shape=count)
    try:
        index_array.from_numpy(np.ascontiguousarray(indices, dtype=np.int32) if rank else np.zeros((count, 1), np.int32))
        if supplied is not None:
            data_array.from_numpy(np.ascontiguousarray(supplied))
            field_write_coordinates(field, index_array, data_array)
            impl.get_runtime().sync()
            return None
        field_read_coordinates(field, index_array, data_array)
        return data_array.to_numpy()
    finally:
        # Managed ndarray destruction follows the ordinary runtime completion
        # leases; a read/write completes before these cold temporaries retire.
        del index_array, data_array
