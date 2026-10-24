"""Cold bindings and selected dispatches for retained array/template helpers."""

from taichi_forge._lib import core
from taichi_forge.graph._graph import Arg, ArgKind
from taichi_forge.lang import impl
from taichi_forge.lang.field import Field
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.annotations import template
from taichi_forge.types.ndarray_type import NdarrayType


def _parameters(function, arguments):
    kernel = function._primal
    if len(arguments) != len(kernel.arguments):
        raise TaichiRuntimeError("Retained helper argument inventory changed")
    if any(not isinstance(p.annotation, (NdarrayType, template)) for p in kernel.arguments):
        raise TaichiRuntimeError("Retained helper lowering requires ndarray or template arguments")
    return kernel, tuple(zip(kernel.arguments, arguments, strict=True))


def freeze_array_template_call(function, arguments):
    """Compile once at producer creation and retain its actual ordinary dispatch."""
    kernel, parameters = _parameters(function, arguments)
    specialization = kernel.ensure_compiled(*arguments)
    symbols = tuple(
        Arg(ArgKind.NDARRAY, f"array_{i}", value.element_type, ndim=len(value.shape))
        for i, (parameter, value) in enumerate(parameters)
        if isinstance(parameter.annotation, NdarrayType)
    )
    builder = core.GraphBuilder()
    builder.dispatch(kernel.compiled_kernels[specialization], symbols)
    return builder.compile()


def lower_array_template_call(context, function, arguments, *, label="", selected=None):
    """Share bindings; adapters choose deployment specialization or a code loan."""
    kernel, parameters = _parameters(function, arguments)
    symbols = []
    template_trees = set()
    for parameter, value in parameters:
        if isinstance(parameter.annotation, NdarrayType):
            name = context.retained_arrays.require(value)
            symbols.append(Arg(ArgKind.NDARRAY, name, value.element_type, ndim=len(value.shape)))
        elif isinstance(value, Field):
            impl.get_runtime().materialize()
            template_trees.update(int(member.ptr.snode().get_snode_tree_id()) for member in value._get_field_members())
    if template_trees:
        context.field_state(impl.get_runtime().prog._snapshot_snode_tree_dependencies(sorted(template_trees)))
    context.retained_kernel(kernel, arguments, tuple(symbols), label, selected)
