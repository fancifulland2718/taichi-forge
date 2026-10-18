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
    from taichi_forge.aot._graph_storage_bindings import packed_description
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
    algorithms._check_reduce_request(*call.args, **call.kwargs)
    source, destination = (packed_description(a) for a in call.args)
    if (
        len(source.index_shape) != 1 or source.element_shape
        or len(destination.index_shape) > 1 or destination.element_shape
    ):
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
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(call.plan, backend, (values, output), (op,))
        or call.plan.method_name not in {
            "cpu_reduce_ndarray", "cpu_reduce_dense_field",
            "cuda_device_reduce_ndarray", "cuda_device_reduce_dense_field",
            "vulkan_reduce_ndarray", "vulkan_reduce_dense_field",
        }
    ):
        raise TaichiRuntimeError("AOT reduce cannot reconstruct the selected cached strategy")
    parameters = dict(size=source.index_shape[0], dtype=str(source.scalar_type), op=op)
    if backend == "cpu_native":
        parameters["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    context.operations.append(dict(
        kind="native", provider="builtin", operation="reduce", version=1,
        parameters=parameters,
        bindings=dict(values=context.retained_arrays.require(values),
                      output=context.retained_arrays.require(output)),
    ))


def _sort(call, context):
    from . import _algorithms as algorithms

    keys, values = call.args
    options = call.kwargs
    keyed = call.kind == "sort_by_key"
    if keyed:
        if isinstance(keys, (list, tuple)):
            if len(keys) != 1:
                raise TaichiRuntimeError("AOT sequence sort_by_key requires one native key part")
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
    # scalar storage. Never execute/prewarm to discover it or silently replace
    # an explicit CUB, shader or host strategy.
    if not getattr(impl.get_runtime().prog, capability)():
        raise TaichiRuntimeError("AOT sequence sort's selected native backend is unavailable")
    arrays = (keys,) if values is None else (keys, values)
    from taichi_forge.aot._graph_storage_bindings import packed_description

    if any(len((desc := packed_description(a)).index_shape) != 1 or desc.element_shape for a in arrays):
        raise TaichiRuntimeError("AOT sequence sort requires one-dimensional packed scalar storage")
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


def _unique(call, context):
    from taichi_forge.aot._primitive_exports import unique_helper
    from taichi_forge.graph._graph import Arg, ArgKind
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    keyed = call.kind == "unique_by_key"
    expected = algorithms.experimental_unique_by_key if keyed else algorithms.experimental_unique
    if (
        call.func is not expected
        or call.plan is not None
        or set(call.kwargs) != {"mode", "size", "method", "workspace"}
        or call.kwargs["mode"] != "consecutive"
    ):
        raise TaichiRuntimeError("AOT unique requires its concrete consecutive operation")
    _compact_method(call.kwargs["method"])
    if keyed:
        keys, values, output, unique_values, count = call.args
    else:
        keys, output, count = call.args
    capacity, size, ndarray_mode = algorithms._check_rle_key_output_count(
        "AOT unique",
        keys,
        output,
        count,
        call.kwargs["method"],
        call.workspace,
        call.kwargs["size"],
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
    size_name = context._resource(
        object(), {k: v for k, v in Scalar("i32", size).resource("size").items() if k != "name"}
    )
    capacity_name = context._resource(
        object(), {k: v for k, v in Scalar("i32", capacity).resource("capacity").items() if k != "name"}
    )
    helper, exemplar = unique_helper(str(keys.dtype), False)
    kernel = helper._primal
    specialization = kernel.ensure_compiled(exemplar["keys"], exemplar["flags"], size, capacity, _for_aot=True)
    context.kernel(
        kernel.compiled_kernels[specialization],
        (
            Arg(ArgKind.NDARRAY, key_name, keys.dtype, ndim=1),
            Arg(ArgKind.NDARRAY, flags, algorithms.i32, ndim=1),
            Arg(ArgKind.SCALAR, size_name, algorithms.i32),
            Arg(ArgKind.SCALAR, capacity_name, algorithms.i32),
        ),
    )
    context.operations.append(dict(kind="compact", flags=flags, count=count_name, columns=columns))


def export_sequence(sequence, context):
    from ._algorithms import PrimitiveSequence

    if type(sequence) is not PrimitiveSequence or sequence._fused_plan is not None:
        raise TaichiRuntimeError("AOT sequence requires its concrete selected native reconstruction")
    exporters = {
        "sort": _sort, "sort_by_key": _sort, "scan": _scan, "reduce": _reduce,
        "compact": _compact, "unique": _unique, "unique_by_key": _unique,
    }
    for call in tuple(sequence._calls):
        exporter = exporters.get(call.kind)
        if exporter is None:
            raise TaichiRuntimeError(f"AOT primitive sequence cannot reconstruct {call.kind!r} yet")
        exporter(call, context)
