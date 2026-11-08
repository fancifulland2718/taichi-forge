"""Retained Host algorithm calls use the existing native execution provider.

No application kernels or numerical source are generated here. Resource shape,
workspace and device admission remain in ExecutionPlan preparation.
"""

from ._plan import ExecutionPlan
from ._runtime import NdArray


class SortWorkspace:
    def __init__(self, package, max_items=None, device=None):
        package._check()
        if max_items is not None and (type(max_items) is not int or max_items < 0):
            raise ValueError("SortWorkspace max_items must be a nonnegative integer or None")
        if device is not None:
            raise ValueError("Package sort uses its loaded Runtime device")
        self.package, self.max_items = package, max_items
        self._arrays, self._prepared = None, None
        package._arrays.append(self)

    def reserve(self, dtype=None, value_dtype=None, n=None):
        self.package._check()
        if n is not None and (n < 0 or self.max_items is not None and n > self.max_items):
            raise ValueError("Sort request exceeds its explicit workspace capacity")
        return self

    def clear(self):
        if self._prepared is not None:
            self._prepared.close()
        self._prepared, self._arrays = None, None

    close = clear


def sort_inputs(
    package,
    keys,
    values=None,
    *,
    stable=True,
    descending=False,
    method="auto",
    precision="exact",
    workspace=None,
    nan_policy="last",
):
    package._check()
    native = {"x64": "cpu_native", "arm64": "cpu_native", "cuda": "cuda_device", "vulkan": "vulkan_native_radix_u32"}[
        package.runtime.arch
    ]
    if stable is not True or descending is not False or precision != "exact" or nan_policy != "last":
        raise ValueError("Package native sort requires stable ascending exact order with NaNs last")
    if method not in ("auto", native):
        raise ValueError("Package sort requires the Runtime's native device method")
    keys = package.runtime._adapt_package_array(keys)
    values = package.runtime._adapt_package_array(values)
    arrays = (keys,) if values is None else (keys, values)
    for array in arrays:
        if not isinstance(array, NdArray):
            raise TypeError("Package sort requires device arrays or imported Field views")
        array._descriptor(package.runtime)
        if len(array.shape) != 1 or array.shape != keys.shape:
            raise ValueError("Package sort requires matching one-dimensional inputs")
    if keys.element_shape:
        raise ValueError("Package sort requires scalar keys")
    if workspace is not None:
        if not isinstance(workspace, SortWorkspace) or workspace.package is not package:
            raise ValueError("Sort workspace belongs to another loaded package")
        workspace.reserve(n=keys.shape[0])
    return arrays


class PrimitiveSequence:
    def __init__(self, package):
        self.package, self.calls = package, []

    def sort(self, keys, values=None, **options):
        arrays = sort_inputs(self.package, keys, values, **options)
        self.calls.append(arrays)
        return self


class Algorithms:
    def __init__(self, package):
        self.package = package

    def __getattr__(self, name):
        raise RuntimeError(f"Host algorithm ti.algorithms.{name} has no Runtime projection")

    def SortWorkspace(self, max_items=None, device=None):
        return SortWorkspace(self.package, max_items, device)

    def primitive_sequence(self):
        return PrimitiveSequence(self.package)

    def sort(
        self,
        keys,
        values=None,
        *,
        stable=True,
        descending=False,
        method="auto",
        precision="exact",
        workspace=None,
        nan_policy="last",
    ):
        arrays = sort_inputs(
            self.package,
            keys,
            values,
            stable=stable,
            descending=descending,
            method=method,
            precision=precision,
            workspace=workspace,
            nan_policy=nan_policy,
        )
        if workspace is not None and workspace._arrays == arrays:
            workspace._prepared.run()
            return
        operation = dict(kind="sort", keys="keys")
        bindings = dict(keys=keys)
        if values is not None:
            operation["values"], bindings["values"] = "values", values
        prepared = ExecutionPlan([operation]).prepare(self.package._module, bindings)
        if workspace is not None:
            workspace.clear()
            workspace._arrays, workspace._prepared = arrays, prepared
        # Runtime retains an in-flight preparation, including a one-shot call.
        prepared.run()
