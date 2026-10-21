"""Export the existing native lexicographic sort/order/apply closure."""

from itertools import combinations

import numpy as np

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.matrix import MatrixNdarray, VectorNdarray

from ._primitive_definitions import IndexedCopyDefinition, TransformDefinition


def export_key_parts(call, context):
    from taichi_forge.aot._graph_storage_bindings import packed_description
    from taichi_forge.graph._graph import Arg, ArgKind
    from taichi_forge_runtime.aot import Scalar
    from . import _algorithms as algorithms

    parts, values = call.args
    options = call.kwargs
    if (
        call.func is not algorithms.sort_by_key
        or call.plan is not None
        or type(call.workspace) is not algorithms.SortWorkspace
        or set(options) != {"stable", "order", "method", "workspace"}
        or options["workspace"] is not call.workspace
        or options["stable"] is not True
        or options["order"] != "lexicographic"
        or len(parts) < 2
    ):
        raise TaichiRuntimeError(
            "AOT multi-key sort requires its concrete stable lexicographic call"
        )
    arch = impl.current_cfg().arch
    method, prefix = {
        core.Arch.x64: ("cpu_native", "cpu"),
        core.Arch.arm64: ("cpu_native", "cpu"),
        core.Arch.cuda: ("cuda_device", "cuda_device"),
        core.Arch.vulkan: ("vulkan_native_radix_u32", "vulkan"),
    }.get(arch, (None, None))
    if options["method"] == "auto" and arch in (core.Arch.x64, core.Arch.arm64):
        raise TaichiRuntimeError(
            "AOT multi-key sort cannot replace the CPU auto host strategy; select cpu_native"
        )
    if method is None or options["method"] not in ("auto", method):
        raise TaichiRuntimeError(
            "AOT multi-key sort requires the selected native sort/order/apply strategy"
        )
    targets = tuple(parts) + (() if values is None else (values,))
    if any(type(a) is not ScalarNdarray for a in parts) or (
        values is not None
        and type(values) not in (ScalarNdarray, VectorNdarray, MatrixNdarray)
    ):
        raise TaichiRuntimeError(
            "AOT multi-key sort requires packed scalar ndarray keys and scalar or tensor payload"
        )
    if len({id(a.arr) for a in targets}) != len(targets):
        raise TaichiRuntimeError(
            "AOT multi-key sort requires disjoint keys and payload"
        )
    descriptors = tuple(map(packed_description, targets))
    n = descriptors[0].index_shape[0]
    for desc in descriptors:
        if tuple(desc.index_shape) != (n,) or str(desc.scalar_type) not in {
            "i32",
            "u32",
            "f32",
            "i64",
            "u64",
            "f64",
        }:
            raise TaichiRuntimeError(
                "AOT multi-key sort requires matching one-dimensional packed storage"
            )
    for part in parts:
        algorithms._check_sort_request(
            part, None, True, False, method, "exact", call.workspace, "last"
        )
    prog = impl.get_runtime().prog
    sort_capability = {
        "cpu_native": "cpu_stable_sort_available",
        "cuda_device": "cuda_device_radix_sort_available",
        "vulkan_native_radix_u32": "vulkan_radix_sort_available",
    }[method]
    if not all(
        getattr(prog, name)()
        for name in (
            sort_capability,
            prefix + "_indexed_copy_available",
            prefix + "_transform_available",
        )
    ):
        raise TaichiRuntimeError(
            "AOT multi-key sort's native sort/order/apply backend is unavailable"
        )
    if arch == core.Arch.vulkan and any(
        not prog.vulkan_transform_value_type_available(
            algorithms._transform_value_type(a.dtype)
        )
        for a in targets
        if not a.element_shape
    ):
        raise TaichiRuntimeError("AOT multi-key sort dtype is unavailable on Vulkan")

    names = tuple(context.retained_arrays.require(a) for a in targets)
    for left, right in combinations(names, 2):
        context.binding_constraints.append(
            dict(kind="disjoint", left=left, right=right)
        )
    # The JIT native closure also returns without touching singleton data.
    # Existing zero-byte copies retain argument contracts but emit no device
    # work; native exporters and top-level plans require nonempty descriptions.
    if n <= 1:
        context.operations.extend(
            dict(kind="copy", source=name, destination=name, size_bytes=0)
            for name in names
        )
        return
    order = context._resource(object(), dict(kind="ndarray", dtype="i32", shape=[n]))
    count = context._resource(
        object(),
        {k: v for k, v in Scalar("i32", n).resource("count").items() if k != "name"},
    )
    # Specialize the existing helper from host metadata; do not allocate or
    # prewarm the producer workspace merely to discover its execution closure.
    kernel = algorithms.fill_i32_arange_ndarray._primal
    specialization = kernel.ensure_compiled(
        np.empty(0, dtype=np.int32), n, _for_aot=True
    )
    compiled = kernel.compiled_kernels[specialization]

    def emit_arange(ctx, destination):
        ctx.kernel(
            compiled,
            (
                Arg(ArgKind.NDARRAY, destination, algorithms.i32, ndim=1),
                Arg(ArgKind.SCALAR, count, algorithms.i32),
            ),
        )

    identity_order = None
    if values is not None and values.element_shape:
        identity_order = context._resource(
            object(), dict(kind="ndarray", dtype="i32", shape=[n])
        )
        # The immutable copy-back indices belong to this prepared frame and
        # are initialized once, using the same helper code as each sort pass.
        context.initialization.extend(context.region(emit_arange, identity_order))
    temporaries = {}
    for key_index in reversed(range(len(parts))):
        emit_arange(context, order)
        context.operations.append(
            dict(kind="sort", keys=names[key_index], values=order)
        )
        for index, (name, descriptor) in enumerate(zip(names, descriptors)):
            if index == key_index:
                continue
            dtype = str(descriptor.scalar_type)
            shape = tuple(descriptor.element_shape)
            key = (dtype, shape)
            if key not in temporaries:
                temporaries[key] = context._resource(
                    object(),
                    dict(
                        kind="ndarray",
                        dtype=dtype,
                        shape=[n],
                        element_shape=list(shape),
                    ),
                )
            temp = temporaries[key]
            gather = IndexedCopyDefinition(n, n, n, descriptor.scalar_type, shape, False)
            gather.bind(name, order, temp).lower(context)
            if shape:
                gather.bind(temp, identity_order, name).lower(context)
                continue
            # Preserve the JIT's affine identity operation, including its
            # floating-point semantics; a byte copy is not an equivalent swap.
            TransformDefinition(n, descriptor.scalar_type, 1, 0).bind(temp, name).lower(context)
