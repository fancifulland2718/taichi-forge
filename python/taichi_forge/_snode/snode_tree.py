# The reason we import just the taichi_forge.core.util module, instead of the ti_python_core
# object within it, is that ti_python_core is stateful. While in practice ti_python_core is
# loaded during the import procedure, it's probably still good to delay the
# access to it.

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


class SNodeTree:
    def __init__(self, ptr):
        self.prog = impl.get_runtime().prog
        self.ptr = ptr
        self.destroyed = False
        # A wrapper can be retained by a data-oriented owner or a test/report
        # frame beyond ti.reset(). Clear its native references before Program
        # finalization rather than relying on later Python garbage collection.
        impl.get_runtime().register_runtime_object(self)

    def destroy(self):
        if self.destroyed:
            raise TaichiRuntimeError("SNode tree has been destroyed")
        if self.prog != impl.get_runtime().prog:
            # ti.reset() already finalized the owning Program. Do not retain or
            # dereference its native SNodeTree pointer from a later runtime.
            self._mark_destroyed()
            return
        runtime = impl.get_runtime()
        dependency = (int(self.ptr.id()), int(self.ptr.generation()))
        notified = runtime.begin_snode_tree_destroy(dependency)
        try:
            self.ptr.destroy_snode_tree(runtime.prog)
        except BaseException:
            runtime.cancel_snode_tree_destroy(dependency, notified)
            raise

        # Native destruction succeeded. Publish the wrapper state before
        # clearing Python compilation caches so an exceptional cache cleanup
        # cannot leave a live-looking wrapper around a destroyed native tree.
        self._mark_destroyed()

        # FieldExpression holds a SNode* to the place-SNode associated with a SNodeTree
        # Therefore, we have to recompile all the kernels after destroying a SNodeTree
        runtime.clear_compiled_functions()

    def _mark_destroyed(self):
        self.destroyed = True
        self.ptr = None
        self.prog = None

    def _invalidate_runtime(self):
        self._mark_destroyed()

    @property
    def id(self):
        if self.destroyed:
            raise TaichiRuntimeError("SNode tree has been destroyed")
        return self.ptr.id()

    @property
    def generation(self):
        if self.destroyed:
            raise TaichiRuntimeError("SNode tree has been destroyed")
        return self.ptr.generation()
