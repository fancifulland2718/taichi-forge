"""Cold bucket reconstruction through the shared native operation registry."""

import os
import re

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


def _producer_setting(suffix, default):
    name = "TI_VULKAN_BUCKET_BUILDER_" + suffix
    text = os.environ.get(name, str(default)).strip()
    # Freeze the export-time preference without imposing a host int-sized
    # budget. The native schema validates the policy's uint64 representation.
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise TaichiRuntimeError(f"{name} must be an integer for AOT export")
    return int(text)


def export_bucket_builder(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from . import _algorithms as algorithms

    if (
        call.func is not algorithms.experimental_bucket_builder
        or type(call.workspace) is not algorithms.BucketBuilderWorkspace
        or set(call.kwargs) != {"method", "workspace"}
        or call.kwargs["workspace"] is not call.workspace
        or len(call.args) != 4
    ):
        raise TaichiRuntimeError(
            "AOT bucket requires its concrete BucketBuilderWorkspace"
        )
    algorithms._check_bucket_builder_request(*call.args, **call.kwargs)
    keys, values, offsets, output = call.args
    if not all(isinstance(a, algorithms.Ndarray) for a in call.args):
        views = tuple(algorithms._primitive_view(a) for a in call.args)
        if any(
            view is None
            or not view.is_dense_field
            or view.stride != core.data_type_size(a.dtype)
            for a, view in zip(call.args, views)
        ):
            raise TaichiRuntimeError(
                "AOT bucket requires the native packed ndarray or scalar Field route"
            )
    descriptions = tuple(packed_description(a) for a in call.args)
    key, source, offset, target = descriptions
    if (
        any(len(d.index_shape) != 1 for d in descriptions)
        or key.element_shape
        or offset.element_shape
        or str(key.scalar_type) != "i32"
        or str(offset.scalar_type) != "i32"
        or str(source.scalar_type) not in {"i32", "u32", "f32", "i64", "u64", "f64"}
        or source.scalar_type != target.scalar_type
        or tuple(source.element_shape) != tuple(target.element_shape)
    ):
        raise TaichiRuntimeError(
            "AOT bucket requires packed numeric payloads and scalar i32 keys/offsets"
        )
    n, bins = key.index_shape[0], offset.index_shape[0] - 1
    call.workspace.check_shape(n, bins)
    arch = impl.current_cfg().arch
    backend, prefix, capability, methods = {
        core.Arch.x64: (
            "cpu_native_bucket_builder",
            "cpu",
            "cpu_bucket_builder_available",
            ("cpu_native", "cpu_two_level"),
        ),
        core.Arch.arm64: (
            "cpu_native_bucket_builder",
            "cpu",
            "cpu_bucket_builder_available",
            ("cpu_native", "cpu_two_level"),
        ),
        core.Arch.cuda: (
            "cuda_device_bucket_builder",
            "cuda_device",
            "cuda_device_bucket_builder_available",
            ("cuda_device", "cuda_two_level"),
        ),
        core.Arch.vulkan: (
            "vulkan_native_bucket_builder",
            "vulkan",
            "vulkan_bucket_builder_available",
            ("vulkan_native", "vulkan_two_level"),
        ),
    }.get(arch, (None, None, None, ()))
    prog = impl.get_runtime().prog
    if (
        backend is None
        or call.kwargs["method"] not in ("auto", "two_level", *methods)
        or not getattr(prog, capability)()
    ):
        raise TaichiRuntimeError("AOT bucket requires its available native strategy")
    value_type = algorithms._bucket_builder_value_type(values)
    if (
        arch == core.Arch.vulkan
        and not prog.vulkan_bucket_builder_value_type_available(value_type)
    ):
        raise TaichiRuntimeError(
            "AOT bucket payload is unavailable on this Vulkan device"
        )
    native_methods = {
        f"{prefix}_bucket_builder_{suffix}"
        for suffix in ("ndarray", "dense_field", "i32_ndarray")
    }
    if call.plan is not None and (
        type(call.plan) is not algorithms._NativePrimitivePlan
        or not algorithms._native_plan_request_matches(
            call.plan, backend, call.args, (value_type, n, bins)
        )
        or call.plan.method_name not in native_methods
    ):
        raise TaichiRuntimeError(
            "AOT bucket cannot reconstruct the selected cached strategy"
        )
    parameters = dict(
        size=n,
        num_bins=bins,
        dtype=str(source.scalar_type),
        element_shape=list(source.element_shape),
    )
    if arch in (core.Arch.x64, core.Arch.arm64):
        parameters["cpu_max_threads"] = max(1, impl.current_cfg().cpu_max_num_threads)
    elif arch == core.Arch.vulkan:
        parameters["vulkan_policy"] = dict(
            private_enabled=_producer_setting("PRIVATE", 1) != 0,
            private_min_items=max(0, _producer_setting("PRIVATE_MIN_N", 65536)),
            private_max_bins=max(0, _producer_setting("PRIVATE_MAX_BINS", 1024)),
            private_max_bytes=max(0, _producer_setting("PRIVATE_MAX_BYTES", 4 << 20)),
        )
    context.operations.append(
        dict(
            kind="native",
            provider="builtin",
            operation="bucket_builder",
            version=1,
            parameters=parameters,
            bindings={
                name: context.retained_arrays.require(value)
                for name, value in zip(
                    ("keys", "values", "offsets", "output"), call.args
                )
            },
        )
    )
