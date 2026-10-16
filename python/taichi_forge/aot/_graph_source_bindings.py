"""Cold names for live arrays retained by otherwise opaque Graph operations."""

from collections.abc import Mapping

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError


class SourceArrayBindings:
    def __init__(self, bindings, fixed_bindings):
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            raise TypeError("source_bindings must map public names to producer ndarrays")
        self.arrays = dict(bindings)
        self.names = {}
        self.layouts = {}
        self.used = set()
        self.constraints = []
        for name, owner in self.arrays.items():
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("Source binding names must be nonempty strings without NUL")
            identity = self.identity(owner)
            if name in fixed_bindings and fixed_bindings[name] is not owner:
                raise ValueError("Source binding conflicts with private Graph storage")
            if identity in self.names:
                raise ValueError("One source allocation must have one public binding name")
            self.names[identity] = name
            dtype, shape = self.layout(owner)
            self.layouts[name] = (dtype, shape)
            scalar_bytes = core.data_type_size(dtype)
            self.constraints.append(
                dict(
                    kind="array_layout",
                    array=name,
                    dtype=str(dtype),
                    exact_shape=list(shape),
                    element_shape=[],
                    record_bytes=scalar_bytes,
                    alignment=scalar_bytes,
                )
            )

    @staticmethod
    def identity(owner):
        from taichi_forge.linalg.sparse_matrix import SparseMatrix

        if type(owner) is SparseMatrix:
            owner._ensure_valid()
            identity = owner._get_format_contract()["identity"]
            if (
                identity["backend_family"] != "cuda"
                or identity["storage_format"] not in ("csr", "bsr")
                or identity["dtype"] != "f32"
            ):
                raise TaichiRuntimeError("Sparse source bindings require f32 CUDA CSR/BSR values")
            # Keep the matrix owner alive in arrays; its native numeric storage
            # is distinct from the ndarray copied by from_pattern/update_values.
            return ("sparse_values", id(owner.matrix))
        if type(owner) is not ScalarNdarray:
            raise TypeError("Source bindings require scalar ndarrays or qualified SparseMatrix values")
        if owner.arr is None or owner._runtime_prog is not impl.get_runtime().prog:
            raise TaichiRuntimeError("Source binding belongs to a retired or different runtime")
        return owner._runtime_allocation_identity

    @staticmethod
    def layout(owner):
        if type(owner) is ScalarNdarray:
            return owner.dtype, owner.shape
        return owner.dtype, (owner._num_nonzero(),)

    def find(self, owner):
        from taichi_forge.linalg.sparse_matrix import SparseMatrix

        if type(owner) not in (ScalarNdarray, SparseMatrix):
            return None
        name = self.names.get(self.identity(owner))
        if name is not None:
            self.used.add(name)
        return name

    def require(self, owner):
        name = self.find(owner)
        if name is None:
            raise TaichiRuntimeError("AOT retained array needs a name in source_bindings")
        return name

    def validate_kernel(self, arguments, names=None):
        if names is None:
            names = [argument.name for argument in arguments]
        for argument, name in zip(arguments, names, strict=True):
            layout = self.layouts.get(name)
            if layout is not None and (
                argument.tag != core.ArgKind.NDARRAY
                or str(argument.dtype()) != str(layout[0])
                or argument.element_shape
                or argument.field_dim != len(layout[1])
            ):
                raise TaichiRuntimeError("Source binding does not match its compiled kernel argument")

    def validate_used(self, runtime_names):
        if self.arrays.keys() - self.used - set(runtime_names):
            raise ValueError("source_bindings contains arrays unused by the exported Graph")
