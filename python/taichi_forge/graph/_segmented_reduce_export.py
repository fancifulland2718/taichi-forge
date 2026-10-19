"""Export the selected reduction stages and only their live retained arrays."""

import numpy as np

from taichi_forge.lang.exception import TaichiRuntimeError


def export_segmented_reduce(executable, context):
    from taichi_forge.graph._segmented_reduce import _BLOCK, _PARTIAL, _SERIAL, _WARP

    source, strategy = executable._source, executable._manifest.strategy
    if strategy not in (_SERIAL, _WARP, _BLOCK, _PARTIAL):
        raise TaichiRuntimeError("AOT segmented reduction requires a qualified selected strategy")
    arrays = context.retained_arrays
    values, output = arrays.require(source.values), arrays.require(source.output)
    context.binding_constraints.append(dict(kind="disjoint", left=values, right=output))
    if strategy == _PARTIAL:
        # This selected graph never binds the original offsets or per-item IDs.
        # Reconstruct immutable boundaries from the materialized source, not
        # from device scratch or by selecting an algorithm again at load time.
        tiles, ends, partials = executable._owned
        tile_bounds, partial_bounds = source.partial_layout
        arrays.declare_data(tiles, np.asarray(tile_bounds, dtype=np.int32).tobytes())
        arrays.declare_data(ends, np.asarray(partial_bounds, dtype=np.int32).tobytes())
        arrays.declare_scratch(partials)
    else:
        arrays.declare_data(source.layout._offsets, np.asarray(source.offsets, dtype=np.int32).tobytes())
    context.bound_graph(executable._graph, executable._bindings)
