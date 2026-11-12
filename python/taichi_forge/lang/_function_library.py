"""Shared typed function boundaries for package builds and JIT compilation."""

import ast
import copy
import inspect


class FunctionLibrary:
    def __init__(self):
        self.functions = {}
        self._function_variants = {}
        self._function_templates = {}
        self.function_library = False
        self.thread_local_call = False

    def library_function(self, source, args, *, dependencies=None):
        """Retain typed device boundaries without mutating JIT definitions.

        Infer value parameters from DSL operands, never from a model sample.
        Unannotated returns are eligible only for a terminal projection of
        those value parameters. Their local copies retain the incoming types,
        including after assignment. Other return inference stays inline.
        """
        if not self.function_library or not self.thread_local_call:
            return None
        if source not in self._function_templates:
            tree, _ = source._definition.acquire(cache=True)
            returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
            projection = ()
            supported = not any(
                isinstance(node, ast.Attribute) and node.attr in ("simt", "SharedArray", "loop_config")
                for node in ast.walk(tree)
            )
            if source.return_type is None and any(node.value is not None for node in returns):
                body = tree.body[0].body
                value = returns[0].value
                values = value.elts if isinstance(value, ast.Tuple) else [value]
                parameters = {arg.name: i for i, arg in enumerate(source.arguments)}
                supported = (
                    supported
                    and len(returns) == 1
                    and returns[0] is body[-1]
                    and (not isinstance(value, ast.Tuple) or len(values) > 1)
                    and all(isinstance(node, ast.Name) and node.id in parameters for node in values)
                )
                if supported:
                    from taichi_forge.lang import ops
                    from taichi_forge.lang.ast.symbol_resolver import ASTResolver

                    projection = tuple(parameters[node.id] for node in values)
                    # Static assignment/iteration can replace a parameter's
                    # local value with a Python object. Do not change that ABI.
                    names = {node.id for node in values}
                    scope = source._definition.scope()
                    local_names = set(parameters) | {
                        node.id
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                    }
                    scope = {name: value for name, value in scope.items() if name not in local_names}
                    for node in ast.walk(tree):
                        targets = (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else ([node.target] if isinstance(node, (ast.AnnAssign, ast.For)) else [])
                        )
                        if any(
                            isinstance(item, ast.Name) and item.id in names
                            for target in targets
                            for item in ast.walk(target)
                        ):
                            expression = node.iter if isinstance(node, ast.For) else node.value
                            if isinstance(node, (ast.For, ast.AnnAssign)) or (
                                isinstance(expression, ast.Call)
                                and not any(
                                    ASTResolver.resolve_to(expression.func, allowed, scope)
                                    for allowed in (ops.min, ops.max, ops.cast)
                                )
                            ):
                                supported = False
            self._function_templates[source] = (supported, projection)
        supported, projection = self._function_templates[source]
        if not supported:
            return None

        from taichi_forge.lang._local_tensor import is_model_value
        from taichi_forge.lang.expr import Expr
        from taichi_forge.lang.kernel_impl import Func, TaichiCallableTemplateMapper
        from taichi_forge.lang.kernel_arguments import KernelArgument
        from taichi_forge.lang.matrix import MatrixType, VectorType
        from taichi_forge.types import primitive_types

        arguments = list(source.arguments)
        signature = []
        for index, (argument, value) in enumerate(zip(arguments, args)):
            if argument.annotation is not inspect.Parameter.empty:
                continue
            # Model extents carry provenance used by type/resource construction;
            # retain the existing inline path instead of discarding it here.
            if is_model_value(value) or not (isinstance(value, Expr) or type(value) in (int, float, bool)):
                return None
            value = Expr(value)
            if value.is_struct():
                return None
            dtype = value.element_type()
            if dtype not in primitive_types.all_types:
                return None
            dtype = next(item for item in primitive_types.all_types if item == dtype)
            shape = tuple(value.get_shape()) if value.is_tensor() else ()
            if len(shape) > 2:
                return None
            annotation = (
                VectorType(shape[0], dtype)
                if len(shape) == 1
                else MatrixType(shape[0], shape[1], 2, dtype) if shape else dtype
            )
            arguments[index] = KernelArgument(annotation, argument.name, argument.default)
            signature.append((index, str(dtype), shape))
        if not source._can_auto_promote(arguments):
            return None
        returns = source.return_type
        if projection:
            returns = tuple(arguments[index].annotation for index in projection)
            if any(
                id(dtype) not in primitive_types.type_ids and not isinstance(dtype, MatrixType) for dtype in returns
            ):
                return None
        identity = (source, tuple(signature)) if signature or projection else source
        if dependencies is not None:
            identity = (identity, dependencies)
        if identity not in self._function_variants:
            function = copy.copy(source)
            function.func_id = Func.function_counter
            Func.function_counter += 1
            function.arguments = arguments
            function.return_type = returns
            function.is_real_function = True
            function._copy_value_arguments = True
            function.compiled = None
            function.taichi_functions = {}
            function._compiled_program = None
            function._real_function_cache_variants = {}
            function._next_real_function_variant_id = 0
            function.mapper = TaichiCallableTemplateMapper(function.arguments, function.template_slot_locations)
            self._function_variants[identity] = function
            # The lifetime inventory keeps callable keys, including when a
            # source has several inferred value ABIs in the same transaction.
            self.functions[function if signature or projection or dependencies is not None else source] = function
        return self._function_variants[identity]
