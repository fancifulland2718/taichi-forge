"""Build-owned definitions, streamed into the existing AOT Module builder."""

import ast
import copy
import gc
import hashlib
import inspect
import json
from types import SimpleNamespace


def definition_symbol(code):
    options = {"opt_level": code.kernel.opt_level}
    identity = json.dumps([code.semantic_key, options], sort_keys=True).encode("utf8")
    return "definition_" + hashlib.sha256(identity).hexdigest()


class PackageCompilation:
    """Keep compiled artifacts and contracts, not every producer frontend AST.

    Ordinary Kernel/Component caches belong to callers. Only private copies
    participate in this transaction; metadata-only candidates can still prove
    sharing while walking further owner/layout choices. Native builders own
    their exported payload independently of the source-definition lease.
    """

    def __init__(self):
        self.module = None
        self.kernels = {}
        self.exported = {}
        self.resource_effects = {}
        self.functions = {}
        self._function_variants = {}
        self._function_templates = {}
        self.retired_function_definitions = 0
        self.function_library = False
        self.thread_local_call = False
        self.schedule_guard = None
        self.static_preparation = None
        self.static_policies = {}
        self._automatic_gc = None

    def __enter__(self):
        return self

    def attach(self, module, *, function_library=False):
        module._aot_builder._enable_streaming_exports()
        self.module = module
        self.function_library = function_library
        # Discovery has finished. Keep short-lived lowering cycles in the young
        # generations until each private definition is sealed. Otherwise the
        # automatic collector can promote them, requiring a scan of the entire
        # live engine declaration graph for every exported binding.
        self._automatic_gc = gc.isenabled()
        gc.disable()

    def entry(self, entry):
        from taichi_forge.lang.kernel_impl import Kernel, TaichiCallableTemplateMapper

        source = entry._primal
        if source not in self.kernels:
            # Preserve typed free/closure entry signatures and decorator policy,
            # which may be richer than re-reading func.__annotations__.
            kernel = copy.copy(source)
            kernel.kernel_counter = Kernel.counter
            Kernel.counter += 1
            kernel.reset()
            kernel.kernel_cpp = None
            kernel.mapper = TaichiCallableTemplateMapper(kernel.arguments, kernel.template_slot_locations)
            kernel._package_compilation = self
            kernel.runtime.kernels.add(kernel)
            self.kernels[source] = kernel
        return SimpleNamespace(_primal=self.kernels[source], _kernel_owner=getattr(entry, "_kernel_owner", None))

    def library_function(self, source, args):
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
                supported = supported and len(returns) == 1 and returns[0] is body[-1] and (
                    not isinstance(value, ast.Tuple) or len(values) > 1
                ) and all(
                    isinstance(node, ast.Name) and node.id in parameters for node in values
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
                        node.id for node in ast.walk(tree)
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                    }
                    scope = {name: value for name, value in scope.items() if name not in local_names}
                    for node in ast.walk(tree):
                        targets = node.targets if isinstance(node, ast.Assign) else (
                            [node.target] if isinstance(node, (ast.AnnAssign, ast.For)) else [])
                        if any(isinstance(item, ast.Name) and item.id in names
                               for target in targets for item in ast.walk(target)):
                            expression = node.iter if isinstance(node, ast.For) else node.value
                            if isinstance(node, (ast.For, ast.AnnAssign)) or (isinstance(expression, ast.Call) and not any(
                                ASTResolver.resolve_to(expression.func, allowed, scope)
                                for allowed in (ops.min, ops.max, ops.cast)
                            )):
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
            annotation = (VectorType(shape[0], dtype) if len(shape) == 1 else
                          MatrixType(shape[0], shape[1], 2, dtype) if shape else dtype)
            arguments[index] = KernelArgument(annotation, argument.name, argument.default)
            signature.append((index, str(dtype), shape))
        if not source._can_auto_promote(arguments):
            return None
        returns = source.return_type
        if projection:
            returns = tuple(arguments[index].annotation for index in projection)
            if any(id(dtype) not in primitive_types.type_ids and not isinstance(dtype, MatrixType) for dtype in returns):
                return None
        identity = (source, tuple(signature)) if signature or projection else source
        if identity not in self._function_variants:
            function = copy.copy(source)
            function.func_id = Func.function_counter
            Func.function_counter += 1
            function.arguments = arguments
            function.return_type = returns
            function.is_real_function = True
            function._aot_copy_value_arguments = True
            function.compiled = None
            function.taichi_functions = {}
            function._compiled_program = None
            function._real_function_cache_variants = {}
            function._next_real_function_variant_id = 0
            function.mapper = TaichiCallableTemplateMapper(function.arguments, function.template_slot_locations)
            self._function_variants[identity] = function
            # The lifetime inventory keeps callable keys, including when a
            # source has several inferred value ABIs in the same transaction.
            self.functions[function if signature or projection else source] = function
        return self._function_variants[identity]

    def _release_functions(self):
        for function in self.functions.values():
            if function is not None:
                # These are build-owned copies, never the ordinary JIT Func's
                # definitions. Program retains native shells; dropping Python
                # handles alone would keep every callee's frontend IR alive.
                for definition in function.taichi_functions.values():
                    definition._retire_definition()
                    self.retired_function_definitions += 1
                function.compiled = {}
                function.taichi_functions.clear()

    def compile(self, code):
        """Export each discovered implementation before exploring its neighbors."""
        identity = code.kernel, code.key
        if identity in self.exported:
            return
        if code.kernel not in self.kernels.values():
            raise RuntimeError("Package compilation cannot retire a caller-owned kernel")
        symbol = definition_symbol(code)
        code.export(self.module, symbol)
        self.resource_effects[identity] = self.module._aot_builder._kernel_resource_effects(symbol)
        self.module._aot_builder._seal_kernel_export(symbol)
        self.exported[identity] = symbol
        # Definition packages finish with archive(), not an execution-plan
        # builder. Retain the reserved name (and fail-closed collision check),
        # while releasing the Module's Python source handle. Public Module and
        # Component export keep their ordinary reusable source ownership.
        self.module._exported_kernels[symbol] = identity
        code.kernel.compiled_kernels.pop(code.key)
        code.kernel.kernel_cpp = None
        self._release_functions()
        # Lowering contexts can form Python cycles through AST/Expr objects.
        # Release their reference-internal builder leases before native GC.
        gc.collect(1)
        code.runtime.collect_unused_kernel_definitions(force=True)

    def symbol(self, code):
        return self.exported[(code.kernel, code.key)]

    def __exit__(self, *exception):
        try:
            self._release()
        finally:
            if self._automatic_gc is not None:
                if self._automatic_gc:
                    gc.enable()
                else:
                    gc.disable()

    def _release(self):
        from taichi_forge.lang import impl

        # Also release incomplete definitions after a failed build. This never
        # clears user caches or touches submitted executable handles.
        for kernel in self.kernels.values():
            kernel.compiled_kernels.clear()
            kernel.kernel_cpp = None
            kernel._component_candidates.clear()
        self.exported.clear()
        self.resource_effects.clear()
        self._release_functions()
        self.functions.clear()
        self._function_variants.clear()
        self._function_templates.clear()
        self.kernels.clear()
        self.static_preparation = None
        if self.module is not None:
            # Failed builds may have exported a native asset before its Python
            # contract validation failed. Drop the build-owned builder as well.
            self.module._aot_builder = None
            self.module._exported_kernels.clear()
            self.module._kernels.clear()
            self.module = None
        gc.collect()
        impl.get_runtime().collect_unused_kernel_definitions(force=True)
