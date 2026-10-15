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
            scalar_bytes = core.data_type_size(owner.dtype)
            self.constraints.append(
                dict(
                    kind="array_layout",
                    array=name,
                    dtype=str(owner.dtype),
                    exact_shape=list(owner.shape),
                    element_shape=[],
                    record_bytes=scalar_bytes,
                    alignment=scalar_bytes,
                )
            )

    @staticmethod
    def identity(owner):
        if type(owner) is not ScalarNdarray:
            raise TypeError("Source bindings currently require scalar ndarrays")
        if owner.arr is None or owner._runtime_prog is not impl.get_runtime().prog:
            raise TaichiRuntimeError("Source binding belongs to a retired or different runtime")
        return owner._runtime_allocation_identity

    def find(self, owner):
        if type(owner) is not ScalarNdarray:
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
            owner = self.arrays.get(name)
            if owner is not None and (
                argument.tag != core.ArgKind.NDARRAY
                or str(argument.dtype()) != str(owner.dtype)
                or argument.element_shape
                or argument.field_dim != len(owner.shape)
            ):
                raise TaichiRuntimeError("Source binding does not match its compiled kernel argument")

    def validate_used(self, runtime_names):
        if self.arrays.keys() - self.used - set(runtime_names):
            raise ValueError("source_bindings contains arrays unused by the exported Graph")
