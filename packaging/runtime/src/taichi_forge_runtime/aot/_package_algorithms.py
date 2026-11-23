"""Retained Host algorithm calls use the existing native execution provider.

No application kernels or numerical source are generated here. Resource shape,
workspace and device admission remain in ExecutionPlan preparation.
"""

import json

from ._plan import ExecutionPlan
from ._runtime import NdArray


class _Workspace:
    def __init__(self, package, max_items=None, device=None):
        package._check()
        if max_items is not None and (type(max_items) is not int or max_items < 0):
            raise ValueError("Workspace max_items must be a nonnegative integer or None")
        if device is not None:
            raise ValueError("Workspace uses its loaded Runtime device")
        self.package, self.max_items = package, max_items

    def reserve(self, dtype=None, value_dtype=None, n=None):
        self.package._check()
        if n is not None and (n < 0 or self.max_items is not None and n > self.max_items):
            raise ValueError("Primitive exceeds its explicit workspace capacity")
        return self


class SortWorkspace(_Workspace):
    def __init__(self, package, max_items=None, device=None):
        super().__init__(package, max_items, device)
        self._arrays, self._prepared = None, None
        package._arrays.append(self)

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


class CompactWorkspace(_Workspace):
    pass


class RunLengthWorkspace(_Workspace):
    pass


class PrimitiveSequence:
    def __init__(self, package):
        self.package, self.calls = package, []
        self._graph = None

    def _append(self, operations, bindings, resources=()):
        self.close()
        self.calls.append((operations, bindings, resources))
        return self

    def close(self):
        if self._graph is not None:
            self._graph.close()
            self._graph = None

    def sort(self, keys, values=None, **options):
        arrays = sort_inputs(self.package, keys, values, **options)
        bindings = dict(zip(("keys", "values"), arrays))
        return self._append([dict(kind="sort", **{port: port for port in bindings})], bindings)

    sort_by_key = sort

    def _arrays(self, arrays, workspace, workspace_type, method):
        self.package._check()
        if method != "auto":
            raise ValueError("Package primitive requires automatic native method selection")
        result = {}
        for name, value in arrays.items():
            array = self.package.runtime._adapt_package_array(value)
            if not isinstance(array, NdArray):
                raise TypeError("Package primitive requires ndarrays or imported Field views")
            array._descriptor(self.package.runtime)
            result[name] = array
        if workspace is not None:
            if type(workspace) is not workspace_type or workspace.package is not self.package:
                raise ValueError("Primitive requires its own workspace from the same package")
            first = next(iter(result.values()))
            if len(first.shape) != 1:
                raise ValueError("Primitive requires rank-one inputs")
            workspace.reserve(n=first.shape[0])
        return result

    def compact(self, values, flags, output, count, *, method="auto", workspace=None):
        arrays = self._arrays(dict(values=values, flags=flags, output=output, count=count),
                              workspace, CompactWorkspace, method)
        return self._append([dict(kind="compact", **{port: port for port in arrays})], arrays)

    def compact_columns(self, columns, flags, count):
        operations, arrays = [], {}
        for index, (source, destination) in enumerate(columns):
            values, output = f"values{index}", f"output{index}"
            operations.append(dict(values=values, output=output))
            arrays.update({values: source, output: destination})
        arrays.update(flags=flags, count=count)
        arrays = self._arrays(arrays, None, None, "auto")
        return self._append([dict(kind="compact", columns=operations, flags="flags", count="count")], arrays)

    def unique_by_key(self, keys, values, unique_keys, unique_values, count, *, mode="consecutive",
                      size=None, method="auto", workspace=None):
        arrays = self._arrays(dict(keys=keys, values=values, output=unique_keys,
                                   unique_values=unique_values, count=count), workspace, RunLengthWorkspace, method)
        keys = arrays["keys"]
        if mode != "consecutive" or keys.element_shape or keys.dtype not in ("u32", "i32", "u64", "i64"):
            raise ValueError("Package unique_by_key requires consecutive integral scalar keys")
        if len(keys.shape) != 1 or (size is not None and (type(size) is not int or not 0 <= size <= keys.shape[0])):
            raise ValueError("Invalid unique_by_key input size")
        for source, output in (("keys", "output"), ("values", "unique_values")):
            left, right = arrays[source], arrays[output]
            if (left.shape != keys.shape or right.shape != keys.shape
                    or left.element_shape != right.element_shape or left.dtype != right.dtype):
                raise ValueError("unique_by_key requires matching input and output columns")
        plan = self.package._module.execution_plan("__forge_package_unique_by_key_" + keys.dtype)
        document = json.loads(plan.to_json())
        if size is not None and size != keys.shape[0]:
            from ._runtime import Scalar
            size_name = "__forge_package_unique_prefix"
            document["resources"].append(Scalar("i32", size).resource(size_name))
            document["operations"][0]["arguments"][-2] = size_name
        return self._append(document["operations"], arrays, document["resources"])

    def run(self):
        if self._graph is None:
            from ._package_graph import GraphBuilder
            builder = GraphBuilder(self.package)
            builder.append_native(self)
            self._graph = builder.compile()
        self._graph.run({})


class PreparedPrimitive:
    def __init__(self, sequence):
        from ._package_graph import GraphBuilder
        self._sequence = sequence
        builder = GraphBuilder(sequence.package)
        builder.append_native(sequence)
        self._graph = builder.compile()
        self._call = self._graph.prepare({})
        self._closed = False
        sequence.package._arrays.append(self)

    def run(self):
        self._call.run()

    def record(self, *, persistent=False):
        if self._closed:
            raise RuntimeError("Prepared primitive is closed")
        if persistent:
            raise ValueError("Package primitive recordings use Graph-owned workspaces")
        return self._sequence

    def report(self):
        return dict(persistent_recording_available=False)

    def close(self):
        if not self._closed:
            self._call.close()
            self._graph.close()
            self._closed = True


class Algorithms:
    def __init__(self, package):
        self.package = package

    def __getattr__(self, name):
        raise RuntimeError(f"Host algorithm ti.algorithms.{name} has no Runtime projection")

    def SortWorkspace(self, max_items=None, device=None):
        return SortWorkspace(self.package, max_items, device)

    def CompactWorkspace(self, max_items=None, device=None):
        return CompactWorkspace(self.package, max_items, device)

    def RunLengthWorkspace(self, max_items=None, device=None):
        return RunLengthWorkspace(self.package, max_items, device)

    def prepare_compact(self, values, flags, output, count):
        return PreparedPrimitive(self.primitive_sequence().compact(values, flags, output, count))

    def prepare_compact_columns(self, columns, flags, count):
        return PreparedPrimitive(self.primitive_sequence().compact_columns(columns, flags, count))

    def prepare_sort(self, keys, values=None):
        return PreparedPrimitive(self.primitive_sequence().sort(keys, values))

    def prepare_unique_by_key(self, keys, values, unique_keys, unique_values, count, *, size=None):
        sequence = self.primitive_sequence().unique_by_key(keys, values, unique_keys, unique_values, count, size=size)
        return PreparedPrimitive(sequence)

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
