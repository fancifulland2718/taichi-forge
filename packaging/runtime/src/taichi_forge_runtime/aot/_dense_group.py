"""Dense group instance ownership; all layout decisions belong to native code."""

import ctypes as C
import json
import math

from . import _c_api as abi
from ._runtime import NdArray, _Memory, _dtype, _shape


class DenseSNodeGroup:
    @classmethod
    def create(cls, runtime, schema, dimensions):
        if not isinstance(dimensions, dict) or any(
            not isinstance(key, str) or type(value) is not int for key, value in dimensions.items()
        ):
            raise TypeError("Dense group dimensions require explicit integer bindings")
        if not isinstance(schema, str):
            raise TypeError("Dense group schema must be serialized native schema JSON")
        with runtime._lock:
            runtime._check()
            api = runtime._api
            api.load_dense_snode_groups()
            handle = api.call(
                api.ti_create_dense_snode_group_ext,
                runtime._handle,
                schema.encode("utf8"),
                json.dumps(dimensions).encode("utf8"),
            )
            memory = None
            try:
                size = C.c_uint64()
                api.call(api.ti_get_dense_snode_group_layout_ext, runtime._handle, handle, C.byref(size), None)
                buffer = C.create_string_buffer(size.value)
                api.call(api.ti_get_dense_snode_group_layout_ext, runtime._handle, handle, C.byref(size), buffer)
                layout = json.loads(buffer.value)
                memory = _Memory._from_import(runtime, handle, layout["root_size"])
                result = cls()
                result._memory = memory
                result._layout_json = buffer.value
                return result
            except BaseException:
                if memory is not None:
                    memory.close()
                else:
                    api.call(api.ti_free_memory, runtime._handle, handle)
                raise

    @property
    def layout(self):
        self._memory._check()
        return json.loads(self._layout_json)

    @property
    def nbytes(self):
        return self._memory.size

    def member(self, paths, dtype, *, element_shape=()):
        """Bind declared scalar-place paths as one compact ndarray argument."""
        if (
            not isinstance(paths, (list, tuple))
            or not paths
            or any(not isinstance(path, (list, tuple)) or any(type(i) is not int for i in path) for path in paths)
        ):
            raise TypeError("Dense member requires explicit scalar-place child paths")
        code, _, width = _dtype(dtype)
        element_shape = _shape(element_shape)
        shape = abi.Shape(len(element_shape), (abi.UInt * 16)(*element_shape))
        runtime = self._memory._runtime
        with runtime._lock:
            self._memory._check()
            array = abi.NdArray()
            api = runtime._api
            api.call(
                api.ti_get_dense_snode_group_member_ext,
                runtime._handle,
                self._memory._handle,
                json.dumps(paths).encode("utf8"),
                code,
                C.byref(shape),
                C.byref(array),
            )
            memory = None
            try:
                index_shape = tuple(array.shape.dims[: array.shape.dim_count])
                size = math.prod(index_shape) * math.prod(element_shape) * width
                memory = _Memory._from_import(runtime, array.memory, size)
                result = NdArray(memory, dtype, code, index_shape, element_shape)
                # Python collection of the group must not retire live views.
                # Explicit group.close() still revokes the native family.
                result._group_owner = self
                return result
            except BaseException:
                if memory is not None:
                    memory.close()
                else:
                    api.call(api.ti_free_memory, runtime._handle, array.memory)
                raise

    def close(self):
        self._memory.close()

    def __enter__(self):
        self._memory._check()
        return self

    def __exit__(self, *_):
        self.close()
