"""Private producer bridge for compiled DSL template constructors.

The producer compiles callable constructors together with its host module.
Consumers build fresh frontend nodes; no source lookup, serialized AST, or
second compiler is involved. This protocol is tied to the Forge frontend and
the producer's Python ABI, and is not a stable third-party serialization format.
"""

import ast
from dataclasses import dataclass
import inspect


def node(kind, fields, position=()):
    """Shared constructor to avoid repeating keyword-call code in host binaries."""
    value = getattr(ast, kind)(*fields)
    if position:
        value.lineno, value.col_offset, value.end_lineno, value.end_col_offset = position
    return value


@dataclass(frozen=True)
class CompiledTemplate:
    factory: object
    globals_getter: object
    closure_getter: object
    is_class: bool
    filename: str
    signature: object

    def scope(self):
        result = dict(self.globals_getter())
        result.update(self.closure_getter())
        return result

    def instantiate(self, name):
        tree = self.factory()
        if (
            not isinstance(tree, ast.Module)
            or len(tree.body) != 1
            or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != name
            or tree.body[0].decorator_list
        ):
            raise ValueError("Compiled DSL constructor must produce its undecorated function")
        lines = max((getattr(node, "end_lineno", 1) or 1 for node in ast.walk(tree)), default=1)
        # Preserve locations for diagnostics without retaining private source.
        return tree, ["<compiled DSL template>"] * lines


def bind(factory, *, globals_getter, closure_getter, is_class, filename, annotations):
    """Attach producer-owned constructors before the ordinary DSL decorator."""
    if not all(callable(value) for value in (factory, globals_getter, closure_getter)):
        raise TypeError("Compiled DSL templates require callable constructors and scopes")
    if type(is_class) is not bool or not isinstance(filename, str):
        raise TypeError("Compiled DSL templates require explicit class and location metadata")

    def decorate(function):
        # Compiled Python hosts may stringify annotations independently of
        # annotation_typing, including behind a staticmethod wrapper. Keep
        # evaluated DSL types in our signature rather than mutating reflection.
        original = inspect.signature(function)
        signature = original.replace(
            parameters=[
                p.replace(annotation=annotations.get(p.name, inspect.Parameter.empty))
                for p in original.parameters.values()
            ],
            return_annotation=annotations.get("return", inspect.Signature.empty),
        )
        template = CompiledTemplate(factory, globals_getter, closure_getter, is_class, filename, signature)
        function._taichi_compiled_template = template
        return function

    return decorate
