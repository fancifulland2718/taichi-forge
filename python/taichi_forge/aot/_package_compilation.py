"""Build-owned definitions, streamed into the existing AOT Module builder."""

import copy
import gc
import hashlib
import json
from collections import deque
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
        self.field_trees = []
        self._automatic_gc = None
        self._retained_functions = {}
        self._released = False

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

    def reuse_function(self, identity, context):
        state = self._retained_functions.get(identity)
        if state is not None and state["dependencies"].matches(context):
            return state
        return None

    def _retain_functions(self, code):
        # Each typed function family owns the variants needed by its latest
        # successful entry. Unrelated entries do not evict that working set;
        # revisiting a family replaces its historical owner/static/tier variants
        # rather than accumulating every definition explored by the build.
        # A single entry may legitimately need several variants of one family.
        states, code._function_states = code._function_states, {}
        owned = {function for function in self.functions.values() if function is not None}
        available = {**self._retained_functions, **states}
        current = {definition for state in states.values() for definition in state["definitions"]}
        touched = {function for function, _ in current}
        eligible = {
            (identity[0], state["key"].instance_id)
            for identity, state in available.items()
            if identity[0] in owned and not state["recursive"] and state["local_index"] is None
            and "dependencies" in state and state["dependencies"].reusable()
            and (identity[0] not in touched or (identity[0], state["key"].instance_id) in current)
        }
        # Definitions are transitive. Replacing a leaf also removes every old
        # parent that still calls it, including parents absent from this entry.
        # Reused parents carry their descendants even when those callees were
        # not visited separately by this capture.
        self._retained_functions = {
            identity: state for identity, state in available.items()
            if (identity[0], state["key"].instance_id) in eligible and state["definitions"] <= eligible
        }
        return self._retained_definitions()

    def _retained_definitions(self):
        return {
            definition for state in self._retained_functions.values() for definition in state["definitions"]
        }

    def _release_functions(self, keep=()):
        for function in self.functions.values():
            if function is not None:
                # These are build-owned copies, never the ordinary JIT Func's
                # definitions. Program retains native shells; dropping Python
                # handles alone would keep every callee's frontend IR alive.
                for key, definition in list(function.taichi_functions.items()):
                    if (function, key) in keep:
                        continue
                    definition._retire_definition()
                    self.retired_function_definitions += 1
                    del function.taichi_functions[key]
                # Callbacks own Python AST/Expr cycles. The retained native
                # Function already owns its body and needs no callback lease.
                function.compiled = {}

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
        self._release_functions(self._retain_functions(code))
        # Lowering contexts can form Python cycles through AST/Expr objects.
        # Release their reference-internal builder leases before native GC.
        gc.collect(1)
        code.runtime.collect_unused_kernel_definitions(force=True)

    def symbol(self, code):
        return self.exported[(code.kernel, code.key)]

    def discard_capture(self, kernel):
        """Retire build-owned IR after an inadmissible resource selection.

        The explorer must first drop the exception/traceback and retain only
        its dependency facts. Failed captures never reach compile()'s normal
        retirement path, while automatic cyclic GC is disabled for the build.
        """
        if kernel not in self.kernels.values():
            raise RuntimeError("Package compilation cannot retire a caller-owned kernel")
        kernel.kernel_cpp = None
        # Failed captures never publish their states. Keep the last successful
        # closure intact, retiring only definitions created by the failed try.
        self._release_functions(self._retained_definitions())
        gc.collect(1)
        kernel.runtime.collect_unused_kernel_definitions(force=True)

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

        if self._released:
            return
        # Also release incomplete definitions after a failed build. This never
        # clears user caches or touches submitted executable handles.
        for kernel in self.kernels.values():
            kernel.compiled_kernels.clear()
            kernel.kernel_cpp = None
            kernel._component_candidates.clear()
        self.exported.clear()
        self.resource_effects.clear()
        self._retained_functions.clear()
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
        # Only construct_fields registers these private layout witnesses.
        # Ordinary Module exports and caller-owned Fields retain their trees.
        # Retire after the builder and definition leases have been released.
        runtime = impl.get_runtime()
        retired_trees = False
        try:
            while self.field_trees:
                tree = self.field_trees[-1]
                if not tree.destroyed:
                    # All private code leases are gone. Each native retirement
                    # still validates/notifies its dependents; scan the shared
                    # Python kernel cache once after the owned batch, not once
                    # for every compile-only layout witness.
                    tree._destroy(clear_compiled_functions=False)
                    retired_trees = True
                self.field_trees.pop()
        finally:
            # A partial failure retains the remaining ownership for a retry.
            # Remove retired queued wrappers without materializing live trees.
            with runtime._kernel_compilation_lock:
                runtime._pending_aot_snode_trees = deque(
                    tree for tree in runtime._pending_aot_snode_trees if not tree.destroyed
                )
            if retired_trees:
                # Also release cache leases for the successful prefix if a
                # later tree failed. Remaining trees stay owned for retry.
                runtime.clear_compiled_functions()
        self._released = True
