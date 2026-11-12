"""Build-owned definitions, streamed into the existing AOT Module builder."""

import copy
import gc
import hashlib
import json
from types import SimpleNamespace

from taichi_forge.lang._function_library import FunctionLibrary


def definition_symbol(code):
    options = {"opt_level": code.kernel.opt_level}
    identity = json.dumps([code.semantic_key, options], sort_keys=True).encode("utf8")
    return "definition_" + hashlib.sha256(identity).hexdigest()


class PackageCompilation(FunctionLibrary):
    """Keep compiled artifacts and contracts, not every producer frontend AST.

    Ordinary Kernel/Component caches belong to callers. Only private copies
    participate in this transaction; metadata-only candidates can still prove
    sharing while walking further owner/layout choices. Native builders own
    their exported payload independently of the source-definition lease.
    """

    def __init__(self):
        super().__init__()
        self.module = None
        self.kernels = {}
        self.exported = {}
        self.resource_effects = {}
        self.retired_function_definitions = 0
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

    def library_function(self, source, args, *, dependencies=None):
        if (self.static_preparation is not None
                and self.static_preparation.requires_task_context(source, args)):
            # Direct syntax checks miss SIMT operations in inline callees.
            # Keep the existing task boundary instead of extracting a wrapper
            # whose transitive body requires that task's execution context.
            return None
        return super().library_function(source, args, dependencies=dependencies)

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
