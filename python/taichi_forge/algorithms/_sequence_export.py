"""Expand qualified primitive sequences into the shared AOT native registry."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def _scan(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms._primitive_sequence_scan
        or type(call.workspace) is not algorithms.PrefixSumExecutor
        or set(call.kwargs) != {"workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 1
    ):
        raise TaichiRuntimeError("AOT scan requires its concrete PrefixSumExecutor")
    values = call.args[0]
    desc = packed_description(values)
    if len(desc.index_shape) != 1 or desc.element_shape:
        raise TaichiRuntimeError("AOT scan requires one-dimensional packed scalar storage")
    size = desc.index_shape[0]
    dtype = str(desc.scalar_type)
    if dtype not in {"i32", "u32", "f32", "i64", "u64", "f64"}:
        raise TaichiRuntimeError("AOT scan requires supported native scalar types")
    if call.workspace.sorting_length != size:
        raise TaichiRuntimeError("AOT scan executor length must match its input")
    arch = impl.current_cfg().arch
    backend, capability = {
        core.Arch.x64: ("cpu_native", "cpu_scan_available"),
        core.Arch.arm64: ("cpu_native", "cpu_scan_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_scan_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan_scan_available"),
    }.get(arch, (None, None))
    prog = impl.get_runtime().prog
    if backend is None or not getattr(prog, capability)():
        raise TaichiRuntimeError("AOT scan's selected native backend is unavailable")
    if arch == core.Arch.vulkan and not prog.vulkan_scan_value_type_available(
        algorithms._SCAN_VALUE_TYPE[values.dtype]
    ):
        raise TaichiRuntimeError("AOT scan dtype is unavailable on this Vulkan device")
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (values,), (size,))
        or call.plan.method_name not in {
            "cpu_inclusive_scan_ndarray", "cpu_inclusive_scan_dense_field",
            "cuda_device_inclusive_scan_ndarray", "cuda_device_inclusive_scan_dense_field",
            "vulkan_inclusive_scan_ndarray", "vulkan_inclusive_scan_dense_field",
        }
    ):
        raise TaichiRuntimeError("AOT scan cannot reconstruct the selected cached strategy")
    context.operations.append(dict(
        kind="native", provider="builtin", operation="scan", version=1,
        parameters=dict(size=size, dtype=dtype),
        bindings=dict(values=context.retained_arrays.require(values)),
    ))


def _reduce(call, context):
    from math import prod
    from taichi_forge.aot._graph_storage_bindings import field_description, packed_description
    from taichi_forge.lang._storage_view import describe_storage
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_reduce
        or type(call.workspace) is not algorithms.ReduceWorkspace
        or set(call.kwargs) != {"op", "method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 2
    ):
        raise TaichiRuntimeError("AOT reduce requires its concrete ReduceWorkspace")
    values, output = call.args
    members = any(type(a) is algorithms.StructNdarrayScalarMemberView for a in call.args)
    fields = all(type(a) is algorithms.ScalarField for a in call.args)
    tensors = all(type(a) is algorithms.MatrixField for a in call.args)
    if tensors:
        algorithms._check_matching_matrix_fields(
            "AOT reduce", values, output, require_same_shape=False)
        if (call.kwargs["op"] not in algorithms._SUPPORTED_REDUCE_OPS
                or call.kwargs["method"] not in algorithms._SUPPORTED_REDUCE_METHODS):
            raise TaichiRuntimeError("AOT tensor reduce requires a supported operation and method")
    else:
        algorithms._check_reduce_request(*call.args, **call.kwargs)
    strategy = "strided_scalar" if members else "scalar_field" if fields else "packed_field" if tensors else "packed_array"
    if members:
        if any(type(a) not in (algorithms.StructNdarrayScalarMemberView, algorithms.ScalarNdarray)
               for a in call.args):
            raise TaichiRuntimeError("AOT scalar member reduce requires scalar ndarray/member storage")
        descriptions = [describe_storage(a) for a in call.args]
        if any(not d.supported for d in descriptions):
            raise TaichiRuntimeError("AOT scalar member reduce requires qualified dense storage")
        source, destination = (d.descriptor for d in descriptions)
    elif fields:
        source, destination = (field_description(a) for a in call.args)
    else:
        source, destination = (packed_description(a) for a in call.args)
    element_shape = tuple(source.element_shape)
    if tensors:
        if (not element_shape or element_shape != tuple(destination.element_shape)
                or len(source.index_shape) != 1 or destination.index_shape
                or not source.index_shape[0]
                or any(a._native_dense_packed_plan() is None for a in call.args)):
            raise TaichiRuntimeError("AOT tensor reduce requires matching packed vector/matrix Fields")
    elif source.element_shape or destination.element_shape:
        raise TaichiRuntimeError("AOT reduce requires scalar storage")
    if strategy != "packed_array":
        if any(not d.properties["single_record_stride_compatible"] or d.properties["has_negative_stride"]
               for d in (source, destination)):
            raise TaichiRuntimeError("AOT scalar reduce requires a positive uniform record stride")
    elif len(source.index_shape) != 1 or len(destination.index_shape) > 1:
        raise TaichiRuntimeError("AOT reduce requires packed scalar storage")
    arch = impl.current_cfg().arch
    backend, capability = {
        core.Arch.x64: ("cpu_native", "cpu_reduce_available"),
        core.Arch.arm64: ("cpu_native", "cpu_reduce_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_reduce_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan_reduce_available"),
    }.get(arch, (None, None))
    prog = impl.get_runtime().prog
    if (backend is None or call.kwargs["method"] not in ("auto", backend)
            or not getattr(prog, capability)()):
        raise TaichiRuntimeError("AOT reduce requires its available native strategy")
    if arch == core.Arch.vulkan and not prog.vulkan_reduce_value_type_available(
        algorithms._REDUCE_VALUE_TYPE[values.dtype]
    ):
        raise TaichiRuntimeError("AOT reduce dtype is unavailable on this Vulkan device")
    op = call.kwargs["op"]
    methods = {
        "cpu_reduce_strided_ndarray", "cuda_device_reduce_strided_ndarray",
        "vulkan_reduce_strided_ndarray",
    } if members else {
        "cpu_reduce_dense_field_packed", "cuda_device_reduce_dense_field_packed",
        "vulkan_reduce_dense_field_packed",
    } if tensors else {
        "cpu_reduce_ndarray", "cpu_reduce_dense_field",
        "cuda_device_reduce_ndarray", "cuda_device_reduce_dense_field",
        "vulkan_reduce_ndarray", "vulkan_reduce_dense_field",
    }
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (values, output), (op,))
        or call.plan.method_name not in methods
    ):
        raise TaichiRuntimeError("AOT reduce cannot reconstruct the selected cached strategy")
    parameters = dict(size=prod(source.index_shape), dtype=str(source.scalar_type), op=op)
    if strategy != "packed_array":
        parameters["strategy"] = strategy
    if tensors:
        parameters["element_shape"] = list(element_shape)
    if backend == "cpu_native":
        parameters["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    context.operations.append(dict(
        kind="native", provider="builtin", operation="reduce", version=1,
        parameters=parameters,
        bindings=dict(values=context.retained_arrays.require(values),
                      output=context.retained_arrays.require(output)),
    ))


def _transform(call, context):
    from math import prod
    from taichi_forge.aot._graph_storage_bindings import field_description, packed_description
    from taichi_forge.lang._storage_view import describe_storage
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_transform
        or type(call.workspace) is not algorithms.TransformWorkspace
        or set(call.kwargs) != {"scale", "bias", "method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 2
    ):
        raise TaichiRuntimeError("AOT transform requires its concrete TransformWorkspace")
    src, dst = call.args
    packed_fields = any(type(a) is algorithms.MatrixField for a in call.args)
    members = any(type(a) is algorithms.StructNdarrayScalarMemberView for a in call.args)
    scalar_fields = any(type(a) is algorithms.ScalarField for a in call.args) and not packed_fields
    if packed_fields:
        # Match the whole-Field entry path, which accepts the native numeric
        # types before the narrower generic Field fallback is considered.
        algorithms._check_matching_matrix_fields("experimental_transform()", src, dst)
        call.workspace.check_shape(algorithms._shape_numel(src))
    else:
        algorithms._check_transform_request(src, dst, call.kwargs["method"], call.workspace)
    if members:
        # The selected JIT entry accepts scalar members and ordinary ndarrays.
        # Reconstruct the same parent/leaf mapping; never insert a packing pass.
        if any(type(a) not in (algorithms.StructNdarrayScalarMemberView, algorithms.ScalarNdarray)
               for a in call.args):
            raise TaichiRuntimeError("AOT scalar member transform requires scalar ndarray/member storage")
        descriptions = [describe_storage(a) for a in call.args]
        if any(not d.supported for d in descriptions):
            raise TaichiRuntimeError("AOT scalar member transform requires qualified dense storage")
        source, destination = (d.descriptor for d in descriptions)
    elif scalar_fields:
        source, destination = (field_description(a) for a in call.args)
    else:
        source, destination = (packed_description(a) for a in call.args)
    strided_fields = scalar_fields and any(
        len(d.index_shape) != 1
        or not all(d.properties[key] for key in ("compact_contiguous", "ndarray_abi_compatible"))
        for d in (source, destination)
    )
    if strided_fields and any(
        not d.properties["single_record_stride_compatible"] or d.properties["has_negative_stride"]
        for d in (source, destination)
    ):
        raise TaichiRuntimeError("AOT scalar Field transform requires a positive uniform record stride")
    if not strided_fields and any(len(d.index_shape) != 1 for d in (source, destination)):
        raise TaichiRuntimeError("AOT transform requires one-dimensional packed storage")
    element_shape = tuple(source.element_shape)
    tensor = bool(element_shape)
    if element_shape != tuple(destination.element_shape) or (tensor and any(
        type(a) not in (algorithms.VectorNdarray, algorithms.MatrixNdarray) for a in call.args
    ) and not packed_fields):
        raise TaichiRuntimeError("AOT tensor transform requires matching packed vector/matrix ndarrays or Fields")
    if packed_fields and any(a._native_dense_packed_plan() is None for a in call.args):
        raise TaichiRuntimeError("AOT tensor Field transform requires its native packed layout")
    scale, bias = algorithms._normalize_transform_args(src.dtype, call.kwargs["scale"], call.kwargs["bias"])
    arch = impl.current_cfg().arch
    backend, capability = {
        core.Arch.x64: ("cpu_native", "cpu_transform_available"),
        core.Arch.arm64: ("cpu_native", "cpu_transform_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_transform_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan_transform_available"),
    }.get(arch, (None, None))
    prog = impl.get_runtime().prog
    if (backend is None or call.kwargs["method"] not in ("auto", backend)
            or not getattr(prog, capability)()):
        raise TaichiRuntimeError("AOT transform requires its available native strategy")
    if arch == core.Arch.vulkan and not prog.vulkan_transform_value_type_available(
        algorithms._transform_value_type(src.dtype)
    ):
        raise TaichiRuntimeError("AOT transform dtype is unavailable on this Vulkan device")
    methods = {
        "cpu_transform_affine_strided_ndarray", "cuda_device_transform_affine_strided_ndarray",
        "vulkan_transform_affine_strided_ndarray", "vulkan_transform_affine_strided_ndarray_trusted",
    } if members else {"transform_affine_dense_field_packed"} if packed_fields else {
        "cpu_transform_affine_packed_strided_ndarray",
        "cuda_device_transform_affine_packed_strided_ndarray",
        "vulkan_transform_affine_packed_strided_ndarray",
    } if tensor else {
        "cpu_transform_affine_ndarray", "cpu_transform_affine_dense_field",
        "cuda_device_transform_affine_ndarray", "cuda_device_transform_affine_dense_field",
        "vulkan_transform_affine_ndarray", "vulkan_transform_affine_dense_field",
        "vulkan_transform_affine_ndarray_trusted", "vulkan_transform_affine_dense_field_trusted",
    }
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (src, dst), (scale, bias))
        or call.plan.method_name not in methods
    ):
        raise TaichiRuntimeError("AOT transform cannot reconstruct the selected cached strategy")
    context.operations.append(dict(
        kind="native", provider="builtin", operation="transform", version=1,
        parameters=dict(size=prod(source.index_shape), dtype=str(source.scalar_type),
                        scale_bits=Scalar("f64", float(scale)).resource("scale")["bits"],
                        bias_bits=Scalar("f64", float(bias)).resource("bias")["bits"],
                        **(dict(element_shape=list(element_shape)) if tensor else {}),
                        **(dict(strategy="strided_scalar") if members or strided_fields else
                           dict(strategy="packed_field") if packed_fields else {})),
        bindings=dict(src=context.retained_arrays.require(src),
                      dst=context.retained_arrays.require(dst)),
    ))


def _indexed_copy(call, context):
    from math import prod
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    if (
        call.func not in (algorithms.experimental_gather, algorithms.experimental_scatter)
        or type(call.workspace) is not algorithms.IndexedCopyWorkspace
        or set(call.kwargs) != {"method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 3
    ):
        raise TaichiRuntimeError("AOT indexed copy requires its concrete IndexedCopyWorkspace")
    src, indices, dst = call.args
    scatter = call.func is algorithms.experimental_scatter
    mode = "scatter" if scatter else "gather"
    algorithms._check_indexed_copy_request(
        src, indices, dst, call.kwargs["method"], call.workspace, f"experimental_{mode}()"
    )
    size = algorithms._indexed_copy_item_count(src, indices, dst, scatter)
    call.workspace.check_shape(size)
    source, index, destination = (packed_description(a) for a in call.args)
    if (any(len(d.index_shape) != 1 for d in (source, index, destination))
            or index.element_shape or str(index.scalar_type) != "i32"
            or source.scalar_type != destination.scalar_type
            or tuple(source.element_shape) != tuple(destination.element_shape)):
        raise TaichiRuntimeError("AOT indexed copy requires packed payloads and scalar i32 indices")
    arch = impl.current_cfg().arch
    backend, prefix, capability = {
        core.Arch.x64: ("cpu_native", "cpu", "cpu_indexed_copy_available"),
        core.Arch.arm64: ("cpu_native", "cpu", "cpu_indexed_copy_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device", "cuda_device_indexed_copy_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan", "vulkan_indexed_copy_available"),
    }.get(arch, (None, None, None))
    prog = impl.get_runtime().prog
    if (backend is None or call.kwargs["method"] not in ("auto", backend)
            or not getattr(prog, capability)()):
        raise TaichiRuntimeError("AOT indexed copy requires its available native strategy")
    item_bytes = core.data_type_size(source.scalar_type) * prod(source.element_shape)
    if arch == core.Arch.cuda and not prog.cuda_device_indexed_copy_payload_available(item_bytes):
        raise TaichiRuntimeError("AOT indexed-copy payload is unavailable on CUDA")
    methods = {
        f"{prefix}_{mode}_{suffix}" for suffix in (
            "ndarray", "strided_ndarray", "dense_field", "dense_field_indices_field",
            "dense_field_packed", "dense_field_packed_indices_field",
        )
    }
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (src, indices, dst), (scatter,))
        or call.plan.method_name not in methods
    ):
        raise TaichiRuntimeError("AOT indexed copy cannot reconstruct the selected cached strategy")
    context.operations.append(dict(
        kind="native", provider="builtin", operation="indexed_copy", version=1,
        parameters=dict(size=size, src_items=source.index_shape[0], dst_items=destination.index_shape[0],
                        dtype=str(source.scalar_type), element_shape=list(source.element_shape), op=mode),
        bindings=dict(src=context.retained_arrays.require(src), indices=context.retained_arrays.require(indices),
                      dst=context.retained_arrays.require(dst)),
    ))


def _scatter_add(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_scatter_add
        or type(call.workspace) is not algorithms.ScatterAddWorkspace
        or set(call.kwargs) != {"method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 3
    ):
        raise TaichiRuntimeError("AOT scatter-add requires its concrete ScatterAddWorkspace")
    src, indices, dst = call.args
    tensors = all(type(a) is algorithms.MatrixField for a in (src, dst))
    if tensors:
        algorithms._check_matching_matrix_fields(
            "AOT scatter-add", src, dst, require_same_shape=False)
    else:
        algorithms._check_scatter_add_request(*call.args, **call.kwargs)
    source, index, destination = (packed_description(a) for a in call.args)
    if (any(len(d.index_shape) != 1 for d in (source, index, destination))
            or index.element_shape or indices.dtype != algorithms.i32
            or source.index_shape != index.index_shape):
        raise TaichiRuntimeError("AOT scatter-add requires one-dimensional storage and matching scalar i32 indices")
    if tensors:
        if (not source.element_shape or source.element_shape != destination.element_shape
                or any(a._native_dense_packed_plan() is None for a in (src, dst))):
            raise TaichiRuntimeError("AOT tensor scatter-add requires matching packed vector/matrix Fields")
    elif source.element_shape or destination.element_shape:
        raise TaichiRuntimeError("AOT scatter-add requires scalar storage")
    n, bound = source.index_shape[0], destination.index_shape[0]
    call.workspace.check_shape(n, bound)
    arch = impl.current_cfg().arch
    backend, prefix, capability = {
        core.Arch.x64: ("cpu_native", "cpu", "cpu_scatter_add_available"),
        core.Arch.arm64: ("cpu_native", "cpu", "cpu_scatter_add_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device", "cuda_device_scatter_add_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan", "vulkan_scatter_add_available"),
    }.get(arch, (None, None, None))
    prog = impl.get_runtime().prog
    if (backend is None or call.kwargs["method"] not in ("auto", backend)
            or not getattr(prog, capability)()):
        raise TaichiRuntimeError("AOT scatter-add requires its available direct native strategy")
    value_type = algorithms._scatter_add_value_type(source.scalar_type)
    if arch == core.Arch.vulkan and not prog.vulkan_scatter_add_value_type_available(value_type):
        raise TaichiRuntimeError("AOT scatter-add dtype atomics are unavailable on Vulkan")
    methods = {
        "scatter_add_dense_field_packed", "scatter_add_dense_field_packed_indices_field",
    } if tensors else {
        f"{prefix}_scatter_add_{suffix}"
        for suffix in ("ndarray", "dense_field", "dense_field_indices_field")
    }
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (src, indices, dst), (value_type,))
        or call.plan.method_name not in methods
    ):
        raise TaichiRuntimeError("AOT scatter-add cannot reconstruct the selected cached strategy")
    parameters = dict(size=n, dst_items=bound, dtype=str(source.scalar_type))
    if tensors:
        parameters.update(strategy="packed_field", element_shape=list(source.element_shape))
    if backend == "cpu_native":
        parameters["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    context.operations.append(dict(
        kind="native", provider="builtin", operation="scatter_add", version=1,
        parameters=parameters,
        bindings=dict(src=context.retained_arrays.require(src), indices=context.retained_arrays.require(indices),
                      dst=context.retained_arrays.require(dst)),
    ))


def _grouped_reduce(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_grouped_reduce
        or type(call.workspace) is not algorithms.GroupedReduceWorkspace
        or set(call.kwargs) != {"op", "method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 3
    ):
        raise TaichiRuntimeError("AOT grouped reduce requires its concrete GroupedReduceWorkspace")
    keys, values, output = call.args
    algorithms._check_grouped_reduce_request(*call.args, **call.kwargs)
    descriptions = tuple(packed_description(a) for a in call.args)
    if any(len(d.index_shape) != 1 or d.element_shape for d in descriptions):
        raise TaichiRuntimeError("AOT grouped reduce requires one-dimensional packed scalar storage")
    n, groups = descriptions[0].index_shape[0], descriptions[2].index_shape[0]
    call.workspace.check_shape(n, groups)
    arch = impl.current_cfg().arch
    backend = call.workspace._native_grouped_reduce_backend_for_method(call.kwargs["method"])
    prefix, capability, accepted = {
        core.Arch.x64: ("cpu", "cpu_grouped_reduce_available", ("cpu_native_two_level",)),
        core.Arch.arm64: ("cpu", "cpu_grouped_reduce_available", ("cpu_native_two_level",)),
        core.Arch.cuda: ("cuda_device", "cuda_device_grouped_reduce_available",
                         ("cuda_device_atomic", "cuda_device_two_level")),
        core.Arch.vulkan: ("vulkan", "vulkan_grouped_reduce_available", ("vulkan_native_atomic",)),
    }.get(arch, (None, None, ()))
    prog = impl.get_runtime().prog
    if backend not in accepted or not getattr(prog, capability)():
        raise TaichiRuntimeError("AOT grouped reduce cannot reconstruct the requested native strategy")
    # CUDA aliases use the same atomic ndarray implementation. Their dense-field
    # request still falls back in JIT, so cold export must not silently change it.
    if backend == "cuda_device_two_level" and any(
        algorithms._primitive_view(a).is_dense_field for a in call.args
    ):
        raise TaichiRuntimeError("AOT grouped reduce cannot replace a CUDA segmented field fallback")
    value_type = algorithms._grouped_reduce_value_type(values.dtype)
    if arch == core.Arch.vulkan and not prog.vulkan_grouped_reduce_atomic_value_type_available(value_type):
        raise TaichiRuntimeError("AOT grouped reduce dtype atomics are unavailable on Vulkan")
    stem = f"{prefix}_grouped_reduce" + ("" if backend == "cpu_native_two_level" else "_atomic")
    methods = {
        f"{stem}_{suffix}"
        for suffix in ("ndarray", "dense_field", "member_ndarray", "strided_ndarray", "strided_keys_ndarray")
    }
    semantic = (value_type, algorithms._SUPPORTED_GROUPED_REDUCE_OPS[call.kwargs["op"]], n, groups)
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, call.args, semantic)
        or call.plan.method_name not in methods
    ):
        raise TaichiRuntimeError("AOT grouped reduce cannot reconstruct the selected cached strategy")
    parameters = dict(size=n, num_groups=groups, dtype=str(values.dtype))
    if backend == "cpu_native_two_level":
        parameters["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    context.operations.append(dict(
        kind="native", provider="builtin", operation="grouped_reduce", version=1,
        parameters=parameters,
        bindings=dict(keys=context.retained_arrays.require(keys), values=context.retained_arrays.require(values),
                      output=context.retained_arrays.require(output)),
    ))


def _histogram(call, context):
    import os
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_histogram
        or type(call.workspace) is not algorithms.HistogramWorkspace
        or set(call.kwargs) != {"method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 2
    ):
        raise TaichiRuntimeError("AOT histogram requires its concrete HistogramWorkspace")
    values, bins = call.args
    algorithms._check_histogram_request(*call.args, **call.kwargs)
    source, destination = (packed_description(a) for a in call.args)
    if any(len(d.index_shape) != 1 or d.element_shape for d in (source, destination)):
        raise TaichiRuntimeError("AOT histogram requires one-dimensional packed scalar storage")
    call.workspace.check_shape(source.index_shape[0], destination.index_shape[0])
    arch = impl.current_cfg().arch
    backend, capability = {
        core.Arch.x64: ("cpu_native", "cpu_histogram_available"),
        core.Arch.arm64: ("cpu_native", "cpu_histogram_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_histogram_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan_histogram_available"),
    }.get(arch, (None, None))
    prog = impl.get_runtime().prog
    if (backend is None
            or call.workspace._native_histogram_backend_for_method(call.kwargs["method"]) != backend
            or not getattr(prog, capability)()):
        raise TaichiRuntimeError("AOT histogram requires its available native strategy")
    semantic = (algorithms._histogram_value_type(values.dtype), algorithms._histogram_bin_type(bins.dtype))
    if arch == core.Arch.vulkan and not prog.vulkan_histogram_value_type_available(*semantic):
        raise TaichiRuntimeError("AOT histogram dtype is unavailable on this Vulkan device")
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (values, bins), semantic)
        or call.plan.method_name not in {
            "cpu_histogram_ndarray", "cpu_histogram_dense_field",
            "cuda_device_histogram_ndarray", "cuda_device_histogram_dense_field",
            "vulkan_histogram_ndarray", "vulkan_histogram_dense_field",
        }
    ):
        raise TaichiRuntimeError("AOT histogram cannot reconstruct the selected cached strategy")
    parameters = dict(size=source.index_shape[0], num_bins=destination.index_shape[0],
                      value_dtype=str(values.dtype), counter_dtype=str(bins.dtype))
    if backend == "cpu_native":
        parameters["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    elif backend == "vulkan_native":
        # Export a semantic recipe with cold export-time preferences, not the
        # producer's materialized command buffers. Consumer environments cannot
        # change this recipe, and replay never probes environment configuration.
        parameters["vulkan_policy"] = {
            key: max(0, int(os.environ.get("TI_VULKAN_HISTOGRAM_" + key.upper(), default)))
            for key, default in (("private_min_n", 65536), ("private_max_bins", 512),
                                 ("single_shared_max_n", 4096))
        }
    context.operations.append(dict(
        kind="native", provider="builtin", operation="histogram", version=1,
        parameters=parameters,
        bindings=dict(values=context.retained_arrays.require(values), bins=context.retained_arrays.require(bins)),
    ))


def _sort(call, context):
    from . import _algorithms as algorithms

    keys, values = call.args
    options = call.kwargs
    keyed = call.kind == "sort_by_key"
    if keyed:
        if isinstance(keys, (list, tuple)):
            if len(keys) != 1:
                from ._sort_export import export_key_parts

                return export_key_parts(call, context)
            keys = keys[0]
        if set(options) != {"stable", "order", "method", "workspace"} or options["order"] != "lexicographic":
            raise TaichiRuntimeError("AOT sequence sort_by_key requires exact lexicographic ordering")
        options = dict(options, descending=False, precision="exact", nan_policy="last")
        del options["order"]
    arch = impl.current_cfg().arch
    method, capability = {
        core.Arch.x64: ("cpu_native", "cpu_stable_sort_available"),
        core.Arch.arm64: ("cpu_native", "cpu_stable_sort_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_radix_sort_available"),
        core.Arch.vulkan: ("vulkan_native_radix_u32", "vulkan_radix_sort_available"),
    }.get(arch, (None, None))
    if (
        call.func is not (algorithms.sort_by_key if keyed else algorithms.sort)
        or call.plan is not None
        or method is None
        or options.get("method") not in ("auto", method)
        or options.get("stable") is not True
        or options.get("descending") is not False
        or options.get("precision") != "exact"
        or options.get("nan_policy") != "last"
    ):
        raise TaichiRuntimeError("AOT sequence sort requires its stable ascending native strategy with NaNs last")
    if set(options) != {"method", "stable", "descending", "precision", "nan_policy", "workspace"}:
        raise TaichiRuntimeError("AOT sequence sort has an unknown execution option")
    # This is the same native-first route used by public auto sort for compact
    # scalar keys and packed ndarray records. Never execute/prewarm to discover
    # it or silently replace an explicit CUB, shader or host strategy.
    if not getattr(impl.get_runtime().prog, capability)():
        raise TaichiRuntimeError("AOT sequence sort's selected native backend is unavailable")
    arrays = (keys,) if values is None else (keys, values)
    from taichi_forge.aot._graph_storage_bindings import packed_description

    descriptions = tuple(packed_description(a) for a in arrays)
    if descriptions[0].element_shape or any(len(desc.index_shape) != 1 for desc in descriptions):
        raise TaichiRuntimeError("AOT sequence sort requires one-dimensional packed storage and scalar keys")
    if values is not None and descriptions[1].element_shape:
        from taichi_forge.lang._ndarray import ScalarNdarray
        from taichi_forge.lang.matrix import MatrixNdarray, VectorNdarray

        # Dense tensor Fields and Struct members select different producer
        # closures. Only packed ndarray records use this direct raw-copy sort.
        if type(keys) is not ScalarNdarray or type(values) not in (VectorNdarray, MatrixNdarray):
            raise TaichiRuntimeError("AOT tensor sort requires scalar ndarray keys and vector/matrix ndarray payload")
    if not all(isinstance(a, algorithms.Ndarray) for a in arrays):
        # Keep storage qualification shared with ordinary sort, including
        # mixed scalar Field/ndarray pairs on CPU/CUDA/Vulkan.
        views = tuple(algorithms._primitive_view(a) for a in arrays)
        if not all(
            view is not None and view.is_dense_field and view.stride == core.data_type_size(desc.scalar_type)
            for view, desc in zip(views, descriptions)
        ) and not algorithms._native_mixed_sort_supported(
            keys, values, method=options["method"], descending=False, nan_policy="last"
        ):
            raise TaichiRuntimeError(
                "AOT sequence sort requires the producer's native storage strategy; "
                "use an explicit AOT sort description to select a different backend route"
            )
    algorithms._check_sort_request(keys, values, True, False, method, "exact", call.workspace, "last")
    if any(str(a.dtype) not in {"i32", "u32", "f32", "i64", "u64", "f64"} for a in arrays):
        raise TaichiRuntimeError("AOT sequence sort requires supported native scalar types")
    operation = dict(kind="sort", keys=context.retained_arrays.require(keys))
    if values is not None:
        operation["values"] = context.retained_arrays.require(values)
    context.operations.append(operation)


def _compact_method(method):
    backend, capability = {
        core.Arch.x64: ("cpu_native", "cpu_compact_available"),
        core.Arch.arm64: ("cpu_native", "cpu_compact_available"),
        core.Arch.cuda: ("cuda_device", "cuda_device_compact_available"),
        core.Arch.vulkan: ("vulkan_native", "vulkan_compact_available"),
    }.get(impl.current_cfg().arch, (None, None))
    if backend is None or method not in ("auto", backend) or not getattr(impl.get_runtime().prog, capability)():
        raise TaichiRuntimeError("AOT sequence compaction requires its available native backend")
    return backend


def _compact(call, context):
    from . import _algorithms as algorithms

    if call.func is not algorithms.experimental_compact or set(call.kwargs) != {"method", "workspace"}:
        raise TaichiRuntimeError("AOT compact requires its concrete public operation")
    backend = _compact_method(call.kwargs["method"])
    values, flags, output, count = call.args
    algorithms._check_compact_request(*call.args, **call.kwargs)
    if call.plan is not None:
        # Prewarmed sequences retain a direct native plan. Qualify that exact
        # backend/storage selection rather than ignoring its cached strategy.
        plan = call.plan
        if (
            type(plan) is not algorithms._NativePrimitivePlan
            or plan.backend != backend
            or not plan.matches_program(impl.get_runtime().prog)
            or len(plan.objects) != len(call.args)
            or any(a is not b for a, b in zip(plan.objects, call.args))
            or plan.method_name
            not in {
                "cpu_compact_ndarray",
                "cpu_compact_dense_field",
                "cuda_device_compact_ndarray",
                "cuda_device_compact_dense_field",
                "vulkan_compact_ndarray",
                "vulkan_compact_dense_field",
            }
        ):
            raise TaichiRuntimeError("AOT compact cannot reconstruct the selected cached strategy")
    context.operations.append(
        dict(
            kind="compact",
            **{
                name: context.retained_arrays.require(owner)
                for name, owner in zip(("values", "flags", "output", "count"), (values, flags, output, count))
            },
        )
    )


def _consecutive(call, context):
    from taichi_forge.aot._primitive_exports import unique_helper
    from taichi_forge.graph._graph import Arg, ArgKind
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    keyed = call.kind == "unique_by_key"
    rle = call.kind == "run_length_encode"
    expected = (algorithms.experimental_run_length_encode if rle else
                algorithms.experimental_unique_by_key if keyed else algorithms.experimental_unique)
    if (
        call.func is not expected
        or call.plan is not None
        or set(call.kwargs) != ({"size", "method", "workspace"} if rle else {"mode", "size", "method", "workspace"})
        or (not rle and call.kwargs["mode"] != "consecutive")
        or (rle and type(call.workspace) is not algorithms.RunLengthWorkspace)
    ):
        raise TaichiRuntimeError("AOT consecutive encoding requires its concrete public operation")
    _compact_method(call.kwargs["method"])
    if rle:
        keys, output, lengths, count = call.args
        capacity, size, ndarray_mode = algorithms._check_run_length_encode_request(
            *call.args, **call.kwargs)
    elif keyed:
        keys, values, output, unique_values, count = call.args
    else:
        keys, output, count = call.args
    if not rle:
        capacity, size, ndarray_mode = algorithms._check_rle_key_output_count(
            "AOT unique", keys, output, count, call.kwargs["method"],
            call.workspace, call.kwargs["size"],
        )
        inputs, outputs = (keys,), (output, count)
        if keyed:
            algorithms._check_rle_payload("AOT unique", keys, values, unique_values, ndarray_mode=ndarray_mode)
            inputs, outputs = (keys, values), (output, unique_values, count)
        algorithms._check_rle_storage_aliases("AOT unique", inputs, outputs)
    count_name = context.retained_arrays.require(count)
    if size == 0:
        context.operations.append(dict(kind="fill32", destination=count_name, value=0, size_bytes=4))
        return
    key_name = context.retained_arrays.require(keys)
    columns = [dict(values=key_name, output=context.retained_arrays.require(output))]
    if keyed:
        columns.append(
            dict(values=context.retained_arrays.require(values), output=context.retained_arrays.require(unique_values))
        )
    # One complete head overwrite and one native prefix shared by all columns.
    # Scratch is reconstructed, not sampled from the producer workspace.
    flags = context._resource(object(), dict(kind="ndarray", dtype="i32", shape=[capacity]))
    if rle:
        starts, run_starts = (
            context._resource(object(), dict(kind="ndarray", dtype="i32", shape=[capacity]))
            for _ in range(2)
        )
        columns.append(dict(values=starts, output=run_starts))
    size_name = context._resource(
        object(), {k: v for k, v in Scalar("i32", size).resource("size").items() if k != "name"}
    )
    capacity_name = context._resource(
        object(), {k: v for k, v in Scalar("i32", capacity).resource("capacity").items() if k != "name"}
    )
    helper, exemplar = unique_helper(str(keys.dtype), rle)
    kernel = helper._primal
    specialization = kernel.ensure_compiled(
        exemplar["keys"], exemplar["flags"],
        *([exemplar["starts"]] if rle else []), size, capacity, _for_aot=True,
    )
    context.kernel(
        kernel.compiled_kernels[specialization],
        (
            Arg(ArgKind.NDARRAY, key_name, keys.dtype, ndim=1),
            Arg(ArgKind.NDARRAY, flags, algorithms.i32, ndim=1),
            *([Arg(ArgKind.NDARRAY, starts, algorithms.i32, ndim=1)] if rle else []),
            Arg(ArgKind.SCALAR, size_name, algorithms.i32),
            Arg(ArgKind.SCALAR, capacity_name, algorithms.i32),
        ),
    )
    context.operations.append(dict(kind="compact", flags=flags, count=count_name, columns=columns))
    if rle:
        from taichi_forge._kernels import rle_finalize_lengths_ndarray

        # Scalar field counts and vector counts share the same one-element
        # helper ABI. The view retains its original owner and writes no tail.
        count_view = context._resource(object(), dict(
            kind="ndarray_view", array=count_name, offset=0, shape=[1]))
        kernel = rle_finalize_lengths_ndarray._primal
        specialization = kernel.ensure_compiled(
            exemplar["flags"], exemplar["flags"], exemplar["flags"],
            size, capacity, _for_aot=True,
        )
        context.kernel(kernel.compiled_kernels[specialization], (
            Arg(ArgKind.NDARRAY, run_starts, algorithms.i32, ndim=1),
            Arg(ArgKind.NDARRAY, context.retained_arrays.require(lengths), algorithms.i32, ndim=1),
            Arg(ArgKind.NDARRAY, count_view, algorithms.i32, ndim=1),
            Arg(ArgKind.SCALAR, size_name, algorithms.i32),
            Arg(ArgKind.SCALAR, capacity_name, algorithms.i32),
        ))


def _export_indexed_transform(sequence, context):
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    selected = sequence._fused_plan
    candidate = algorithms._try_build_vulkan_indexed_transform_sequence_plan(
        sequence._calls, impl.get_runtime().prog, invoke=False,
        respect_preferences=selected is None,
    )
    if selected is not None and (
        candidate is None
        or type(selected) is not algorithms._NativePrimitivePlan
        or selected.method_name != candidate.method_name
        or not selected.matches_program(impl.get_runtime().prog)
        or not algorithms._native_plan_request_matches(
            selected, candidate.backend, candidate.objects, candidate.semantic_key
        )
    ):
        raise TaichiRuntimeError("AOT sequence cannot reconstruct the selected fused strategy")
    if candidate is None:
        return False
    # Validate the public requests without prewarming or running any operation.
    transform, gather, scatter = sequence._calls
    if (type(transform.workspace) is not algorithms.TransformWorkspace
            or set(transform.kwargs) != {"scale", "bias", "method", "workspace"}
            or transform.kwargs["workspace"] is not transform.workspace):
        raise TaichiRuntimeError("AOT fusion requires its concrete TransformWorkspace")
    algorithms._check_transform_request(
        *transform.args, transform.kwargs["method"], transform.workspace
    )
    transform.workspace.check_shape(candidate.semantic_key[5])
    for call, name in ((gather, "gather"), (scatter, "scatter")):
        if (type(call.workspace) is not algorithms.IndexedCopyWorkspace
                or set(call.kwargs) != {"method", "workspace"}
                or call.kwargs["workspace"] is not call.workspace):
            raise TaichiRuntimeError("AOT fusion requires its concrete IndexedCopyWorkspace")
        algorithms._check_indexed_copy_request(
            *call.args, call.kwargs["method"], call.workspace, f"experimental_{name}()"
        )
        call.workspace.check_shape(candidate.n)
    _, _, scale, bias, index_size, src_size, dst_size = candidate.semantic_key
    context.operations.append(dict(
        kind="native", provider="builtin", operation="indexed_transform", version=1,
        parameters=dict(
            src_size=src_size, index_size=index_size, dst_size=dst_size,
            dtype=str(candidate.objects[0].dtype),
            scale_bits=Scalar("f64", scale).resource("scale")["bits"],
            bias_bits=Scalar("f64", bias).resource("bias")["bits"],
        ),
        bindings={name: context.retained_arrays.require(array) for name, array in zip(
            ("src", "transformed", "indices", "gathered", "dst"), candidate.objects
        )},
    ))
    return True


def export_sequence(sequence, context):
    from ._algorithms import PrimitiveSequence
    from ._diagnostic_export import export_check_call, export_metric_call
    from ._segmented_export import export_segmented_reduce, export_segmented_scan
    from ._bucket_export import export_bucket_builder

    if type(sequence) is not PrimitiveSequence:
        raise TaichiRuntimeError("AOT sequence requires its concrete selected native reconstruction")
    if _export_indexed_transform(sequence, context):
        return
    exporters = {
        "sort": _sort, "sort_by_key": _sort, "scan": _scan, "reduce": _reduce,
        "transform": _transform,
        "indexed_copy": _indexed_copy,
        "scatter_add": _scatter_add,
        "grouped_reduce": _grouped_reduce,
        "segmented_reduce": export_segmented_reduce,
        "segmented_scan": export_segmented_scan,
        "bucket_builder": export_bucket_builder,
        "histogram": _histogram,
        "check": export_check_call, "metric": export_metric_call,
        "compact": _compact, "unique": _consecutive, "unique_by_key": _consecutive,
        "run_length_encode": _consecutive,
    }
    for call in tuple(sequence._calls):
        exporter = exporters.get(call.kind)
        if exporter is None:
            raise TaichiRuntimeError(f"AOT primitive sequence cannot reconstruct {call.kind!r} yet")
        exporter(call, context)
