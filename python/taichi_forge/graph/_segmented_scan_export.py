"""Export the selected segmented-scan stages and their retained array closure."""

import numpy as np

from taichi_forge.lang.exception import TaichiRuntimeError


def export_segmented_scan(executable, context):
    from taichi_forge.graph._native_algorithm import _CHUNK_STRATEGIES, _GLOBAL_STRATEGY, _SERIAL_STRATEGY
    from taichi_forge.graph._graph import Arg, ArgKind
    from taichi_forge_runtime.aot import Scalar

    source, strategy = executable._source, executable._strategy
    if strategy not in (*_CHUNK_STRATEGIES, _GLOBAL_STRATEGY, _SERIAL_STRATEGY):
        raise TaichiRuntimeError("AOT segmented scan requires a qualified selected strategy")
    arrays = context.retained_arrays
    values, output = arrays.require(source.values), arrays.require(source.output)
    context.binding_constraints.append(dict(kind="disjoint", left=values, right=output))
    offsets = arrays.declare_data(source.layout._offsets, np.asarray(source.offsets, dtype=np.int32).tobytes())
    if strategy == _SERIAL_STRATEGY:
        call = executable._serial_call
        kernel = call._function._primal
        specialization = kernel.ensure_compiled(*call._args, _for_aot=True)
        constants = [
            context._resource(
                object(), {k: v for k, v in Scalar("i32", value).resource("unused").items() if k != "name"}
            )
            for value in (source.layout.num_segments, int(source.inclusive))
        ]
        context.kernel(
            kernel.compiled_kernels[specialization],
            (
                Arg(ArgKind.NDARRAY, values, source.values.dtype, ndim=1),
                Arg(ArgKind.NDARRAY, offsets, source.layout._offsets.dtype, ndim=1),
                Arg(ArgKind.NDARRAY, output, source.output.dtype, ndim=1),
                *(Arg(ArgKind.SCALAR, name, source.layout._offsets.dtype) for name in constants),
            ),
        )
        return
    if strategy == _GLOBAL_STRATEGY:
        arrays.declare_scratch(executable._scan_scratch)
        arrays.declare_scratch(executable._bases)
    if executable._bucket_indices:
        # These immutable indices were selected at materialization. Recreate
        # from the same frozen topology, never read user arrays or reselect.
        short, long = [], []
        for index, (begin, end) in enumerate(zip(source.offsets, source.offsets[1:])):
            (short if end - begin <= 32 else long).append(index)
        for owner, indices in zip(executable._bucket_indices, (short, long), strict=True):
            arrays.declare_data(owner, np.asarray(indices, dtype=np.int32).tobytes())
    context.bound_graph(executable._nested_graph, executable._nested_bindings)
