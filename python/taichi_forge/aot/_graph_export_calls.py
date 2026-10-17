"""Cold export of retained helper calls with array and template arguments."""

from taichi_forge.graph._graph import Arg, ArgKind
from taichi_forge.lang import impl
from taichi_forge.lang.field import Field
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.annotations import template
from taichi_forge.types.ndarray_type import NdarrayType


def export_array_template_call(context, function, arguments, *, label=""):
    """Reuse normal AOT specialization; concrete templates remain code inputs."""
    kernel = function._primal
    if len(arguments) != len(kernel.arguments):
        raise TaichiRuntimeError("Retained helper argument inventory changed")
    symbols = []
    template_trees = set()
    for parameter, value in zip(kernel.arguments, arguments, strict=True):
        annotation = parameter.annotation
        if isinstance(annotation, NdarrayType):
            name = context.retained_arrays.require(value)
            symbols.append(Arg(ArgKind.NDARRAY, name, value.element_type, ndim=len(value.shape)))
        elif isinstance(annotation, template):
            if isinstance(value, Field):
                # Fail before publishing kernels when mutable Field snapshots
                # were not authorized. Actual compiled roots are sampled once
                # by the enclosing export transaction, including global Fields.
                impl.get_runtime().materialize()
                template_trees.update(
                    int(member.ptr.snode().get_snode_tree_id()) for member in value._get_field_members()
                )
        else:
            raise TaichiRuntimeError("Retained helper export requires ndarray or template arguments")
    if template_trees:
        context.field_state(impl.get_runtime().prog._snapshot_snode_tree_dependencies(sorted(template_trees)))
    specialization = kernel.ensure_compiled(*arguments, _for_aot=True)
    context.kernel(kernel.compiled_kernels[specialization], tuple(symbols), label)
