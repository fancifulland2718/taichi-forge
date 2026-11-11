"""Build-owned definitions, streamed into the existing AOT Module builder."""

import ast
import copy
import gc
import hashlib
import json
from contextlib import contextmanager
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
        self.function_library = False
        self.thread_local_call = False
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

    def library_function(self, source):
        """Retain typed device boundaries without mutating JIT definitions.

        Untyped returns and arguments keep their existing inline semantics.
        This is an opt-in producer representation; prepared LLVM optimization
        remains responsible for profitable inlining after constants are known.
        """
        if not self.function_library or not self.thread_local_call or not source._can_auto_promote():
            return None
        if source not in self.functions:
            from taichi_forge.lang.kernel_impl import Func, TaichiCallableTemplateMapper

            tree, _ = source._definition.acquire(cache=True)
            if (source.return_type is None and any(
                isinstance(node, ast.Return) and node.value is not None for node in ast.walk(tree)
            )) or any(
                isinstance(node, ast.Attribute) and node.attr in ("simt", "SharedArray", "loop_config")
                for node in ast.walk(tree)
            ):
                self.functions[source] = None
            else:
                function = copy.copy(source)
                function.func_id = Func.function_counter
                Func.function_counter += 1
                function.is_real_function = True
                function.compiled = None
                function.taichi_functions = {}
                function._compiled_program = None
                function._real_function_cache_variants = {}
                function._next_real_function_variant_id = 0
                function.mapper = TaichiCallableTemplateMapper(function.arguments, function.template_slot_locations)
                self.functions[source] = function
        return self.functions[source]

    @contextmanager
    def function_call_scope(self, ctx):
        # Calls at kernel scope can contribute parallel offloads or CTA-local
        # storage. Turning that helper into a serial device function changes
        # execution semantics. Only calls already inside an iteration/function
        # may acquire a library boundary, including nested inline helpers.
        previous = self.thread_local_call
        self.thread_local_call = previous or ctx.is_real_function or any(
            not scope.is_static for scope in ctx.loop_scopes
        )
        try:
            yield
        finally:
            self.thread_local_call = previous

    def _release_functions(self):
        for function in self.functions.values():
            if function is not None:
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
